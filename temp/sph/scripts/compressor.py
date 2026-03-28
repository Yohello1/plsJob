import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import sys

# Constants based on SPH settings (Updated to 400x400)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
BUFFER_WIDTH = 400
BUFFER_HEIGHT = 400

# Normalization factors
DENSITY_NORM = 50.0 # Maps 0.02 max to 1.0
VELOCITY_NORM = 1.0 / 7.5 # Maps 7.5 max to 1.0

class SPHDataset(Dataset):
    def __init__(self, data_dirs, skip=10):
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        self.data_dirs = data_dirs
        self.samples = []
        self.skip = skip
        
        for d_dir in self.data_dirs:
            frame_indices = []
            d_files = glob.glob(os.path.join(d_dir, "*_d.bin"))
            for f in d_files:
                try:
                    idx = int(os.path.basename(f).split('_')[0])
                    frame_indices.append(idx)
                except (ValueError, IndexError):
                    continue
            frame_indices.sort()
            
            for i in range(len(frame_indices) - self.skip):
                if frame_indices[i+self.skip] == frame_indices[i] + self.skip:
                    # Store (directory, prev_idx, curr_idx)
                    self.samples.append((d_dir, frame_indices[i], frame_indices[i+self.skip]))

    def __len__(self):
        return len(self.samples)

    def load_bin(self, data_dir, frame_idx, suffix, dtype=np.float32):
        path = os.path.join(data_dir, f"{frame_idx}_{suffix}.bin")
        data = np.fromfile(path, dtype=dtype)
        data = data.reshape((BUFFER_HEIGHT, BUFFER_WIDTH))
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        
        # Apply normalization based on data type
        if suffix == "d":
            return tensor * DENSITY_NORM
        elif suffix.startswith("v"):
            return tensor * VELOCITY_NORM
        return tensor

    def __getitem__(self, idx):
        data_dir, prev_idx, curr_idx = self.samples[idx]
        prev_d = self.load_bin(data_dir, prev_idx, "d")
        prev_vx = self.load_bin(data_dir, prev_idx, "v_x")
        prev_vy = self.load_bin(data_dir, prev_idx, "v_y")
        curr_d = self.load_bin(data_dir, curr_idx, "d")
        return prev_d, torch.cat([prev_vx, prev_vy], dim=0), curr_d

class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        # 400 -> 200 -> 100 -> 50 -> 25
        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=2, padding=1),   # 200
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 100
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),# 50
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),# 25
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(512 * 25 * 25, latent_dim)

    def forward(self, x):
        return self.fc(self.conv(x))

class Decoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 25 * 25)
        self.prev_d_path = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1), # 200
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 100
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 50
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), # 25
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1), # 50
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 100
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 200
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1),     # 400
        )

    def forward(self, z, prev_d):
        z_feat = self.fc(z).view(-1, 512, 25, 25)
        d_feat = self.prev_d_path(prev_d)
        return self.deconv(torch.cat([z_feat, d_feat], dim=1))

class FullModel(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    def forward(self, p_d, p_v, c_d):
        z = self.encoder(torch.cat([p_d, p_v, c_d], dim=1))
        return self.decoder(z, p_d)

def train(requested_epochs=None):
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Detected {num_gpus} GPU(s). Using device: {device}")

    session_dirs = [os.path.join("data", d) for d in os.listdir("data") if os.path.isdir(os.path.join("data", d)) and d != "frames"] if os.path.exists("data") else []
    if not session_dirs:
        print("No simulation data found.")
        return

    # 1. Configuration & Constants
    LR = 2e-5
    LATENT_DIM = 512
    # The dataset default skip is 10, but we can access it from dataset if needed
    
    # 2. Setup Run Directory
    run_base = "attempts"
    os.makedirs(run_base, exist_ok=True)
    run_idx = 1
    while os.path.exists(os.path.join(run_base, f"run{run_idx}")):
        run_idx += 1
    run_dir = os.path.join(run_base, f"run{run_idx}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Starting {run_dir}...")

    # 3. Load Dataset
    dataset = SPHDataset(session_dirs)
    skip_val = dataset.skip
    
    # 4. Save Settings
    epochs = requested_epochs if requested_epochs else 10
    with open(os.path.join(run_dir, "settings.txt"), "w") as f:
        f.write(f"Attempt: {run_idx}\n")
        f.write(f"Learning Rate: {LR}\n")
        f.write(f"Latent Dim: {LATENT_DIM}\n")
        f.write(f"Skip Frames: {skip_val}\n")
        f.write(f"Epochs per Cycle: {epochs}\n")
        f.write(f"Loss Function: MSELoss\n")
        f.write(f"Normalizations: Density={DENSITY_NORM}, Velocity={VELOCITY_NORM}\n")

    # 5. Data Pipeline
    total_batch = max(1, 8 * num_gpus)
    dataloader = DataLoader(dataset, batch_size=total_batch, shuffle=True, num_workers=4, pin_memory=True)
    
    model = FullModel(LATENT_DIM).to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model)
        
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    # Prepare loss logging
    log_file = os.path.join(run_dir, "losses.csv")
    with open(log_file, "w") as f:
        f.write("epoch,loss,zero_loss,ident_loss,pred_var,gt_var\n")

    epochs = requested_epochs if requested_epochs else 50
    for epoch in range(epochs):
        model.train()
        total_zero_loss = 0
        total_ident_loss = 0
        total_pred_var = 0
        total_gt_var = 0
        for p_d, p_v, c_d in dataloader:
            p_d, p_v, c_d = p_d.to(device), p_v.to(device), c_d.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                output = model(p_d, p_v, c_d)
                loss = criterion(output, c_d)
                # Baseline 1: Loss if we predict only 0 (density fields are mostly empty)
                zero_loss = criterion(torch.zeros_like(c_d), c_d).item()
                # Baseline 2: Loss if we predict no change (previous density)
                ident_loss = criterion(p_d, c_d).item()
                
                total_zero_loss += zero_loss
                total_ident_loss += ident_loss
                total_pred_var += torch.var(output).item()
                total_gt_var += torch.var(c_d).item()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Reports normalized MSE
        current_loss = loss.item()
        avg_zero_loss = total_zero_loss / len(dataloader)
        avg_ident_loss = total_ident_loss / len(dataloader)
        avg_pred_var = total_pred_var / len(dataloader)
        avg_gt_var = total_gt_var / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {current_loss:.8f}, Baseline (Zero): {avg_zero_loss:.8f}, Baseline (Ident): {avg_ident_loss:.8f}")
        print(f"    - Var: [Pred: {avg_pred_var:.8f}, GT: {avg_gt_var:.8f}] | Max: [Pred: {output.max().item():.4f}, GT: {c_d.max().item():.4f}]")
        
        # Log to CSV
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{current_loss:.8f},{avg_zero_loss:.8f},{avg_ident_loss:.8f},{avg_pred_var:.8f},{avg_gt_var:.8f}\n")
            
        torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
        # Also keep a copy in main dir for continuity
        torch.save(model.state_dict(), "best_model.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train(args.epochs)
