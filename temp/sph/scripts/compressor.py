import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import sys
import random

# Constants based on SPH settings (Updated to 400x400)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
BUFFER_WIDTH = 400
BUFFER_HEIGHT = 400

# Normalization factors
DENSITY_NORM = 50.0 # Maps 0.02 max to 1.0
VELOCITY_NORM = 1.0 / 7.5 # Maps 7.5 max to 1.0
LATENT_DIM = 1024
# Activation Configuration
ACTIVATION_TYPE = "SiLU"
ACTIVATION_LOOKUP = {
    "ReLU": nn.ReLU,
    "SiLU": nn.SiLU,
    "LeakyReLU": lambda: nn.LeakyReLU(0.01),
    "ELU": nn.ELU
}
ACT = ACTIVATION_LOOKUP[ACTIVATION_TYPE]

class ResBlock(nn.Module):
    """Residual block to help deeper networks learn more effectively."""
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            ACT(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c)
        )
    def forward(self, x):
        return ACT()(x + self.conv(x))

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
    def __init__(self, latent_dim=1024):
        super().__init__()
        # 400 -> 200 -> 100 -> 50
        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=2, padding=1),   # 200
            ResBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 100
            ResBlock(128),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),# 50
            ResBlock(128),
            nn.Flatten()
        )
        self.fc = nn.Linear(128 * 50 * 50, latent_dim)

    def forward(self, x):
        return self.fc(self.conv(x))

class Decoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 50 * 50)
        
        # Downsamples prev_d to 50x50 bottleneck
        self.prev_d_path = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),   # 200
            ACT(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 100
            ACT(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),# 50
            ACT(),
        )
        
        # Upsamples from 50x50 back to 400x400
        self.deconv = nn.Sequential(
            nn.Upsample(size=(100,100), mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), 
            ResBlock(128),
            nn.Upsample(size=(200,200), mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            ResBlock(64),
            nn.Upsample(size=(400,400), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Softplus() # Ensures density is always positive
        )

    def forward(self, z, prev_d):
        z_feat = self.fc(z).view(-1, 128, 50, 50)
        d_feat = self.prev_d_path(prev_d)
        return self.deconv(torch.cat([z_feat, d_feat], dim=1))

class FullModel(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    def forward(self, p_d, p_v, c_d):
        z = self.encoder(torch.cat([p_d, p_v, c_d], dim=1))
        return self.decoder(z, p_d)

    def get_depth(self):
        count = 0
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                count += 1
        return count

def train(requested_epochs=None, data_dir="data", output_dir="attempts", model_filename="best_model.pth"):
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Detected {num_gpus} GPU(s). Using device: {device}")

    # Use specified data_dir
    session_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != "frames"] if os.path.exists(data_dir) else []
    if not session_dirs:
        print(f"No simulation data found in {data_dir}.")
        return

    # 1. Configuration & Constants
    LR = 1e-4
    LATENT_DIM_LOCAL = LATENT_DIM
    # The dataset default skip is 10, but we can access it from dataset if needed
    
    # 2. Setup Run Directory (unique suffix inside output_dir)
    os.makedirs(output_dir, exist_ok=True)
    run_idx = 1
    while os.path.exists(os.path.join(output_dir, f"run{run_idx}")):
        run_idx += 1
    run_dir = os.path.join(output_dir, f"run{run_idx}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Starting {run_dir}...")

    # 3. Split Dataset (90% Train, 10% Val)
    random.seed(42)
    random.shuffle(session_dirs)
    split_idx = max(1, int(len(session_dirs) * 0.9))
    if split_idx >= len(session_dirs) and len(session_dirs) > 1:
        split_idx = len(session_dirs) - 1
    
    train_dirs = session_dirs[:split_idx]
    val_dirs = session_dirs[split_idx:]
    print(f"Dataset split: {len(train_dirs)} training sessions, {len(val_dirs)} validation sessions.")

    train_dataset = SPHDataset(train_dirs)
    val_dataset = SPHDataset(val_dirs)
    skip_val = train_dataset.skip

    # Initialize model early to get depth for logging
    model = FullModel(LATENT_DIM_LOCAL).to(device)
    model_depth = model.get_depth()
    print(f"Model Depth: {model_depth} convolutional layers.")
    
    # 4. Save Settings
    epochs = requested_epochs if requested_epochs else 10
    with open(os.path.join(run_dir, "settings.txt"), "w") as f:
        f.write(f"Attempt: {run_idx}\n")
        f.write(f"Learning Rate: {LR} (Scheduled)\n")
        f.write(f"Activation: {ACTIVATION_TYPE}\n")
        f.write(f"Latent Dim: {LATENT_DIM_LOCAL}\n")
        f.write(f"Grid Res: 400x400\n")
        f.write(f"Bottleneck: 50x50\n")
        f.write(f"Skip Frames: {skip_val}\n")
        f.write(f"Epochs per Cycle: {epochs}\n")
        f.write(f"Train/Val Split: {len(train_dirs)}/{len(val_dirs)}\n")
        f.write(f"Loss Function: MSELoss\n")
        f.write(f"Normalizations: Density={DENSITY_NORM}, Velocity={VELOCITY_NORM}\n")
        f.write(f"Model Depth (Conv Layers): {model_depth}\n")

    # 5. Data Pipelines
    total_batch = max(1, 8 * num_gpus)
    train_loader = DataLoader(train_dataset, batch_size=total_batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=total_batch, shuffle=False, num_workers=4, pin_memory=True)
    
    if num_gpus > 1:
        model = nn.DataParallel(model)
        
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    # Prepare loss logging
    log_file = os.path.join(run_dir, "losses.csv")
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss,val_zero,val_ident,train_var,val_var\n")

    epochs = requested_epochs if requested_epochs else 50
    for epoch in range(epochs):
        # --- TRAINING LOOP ---
        model.train()
        total_train_loss = 0
        total_train_var = 0
        for p_d, p_v, c_d in train_loader:
            p_d, p_v, c_d = p_d.to(device), p_v.to(device), c_d.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                output = model(p_d, p_v, c_d)
                loss = criterion(output, c_d)
                total_train_loss += loss.item()
                total_train_var += torch.var(output).item()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # --- VALIDATION LOOP ---
        model.eval()
        total_val_loss = 0
        total_val_zero = 0
        total_val_ident = 0
        total_val_var = 0
        total_val_gt_var = 0
        with torch.no_grad():
            for p_d, p_v, c_d in val_loader:
                p_d, p_v, c_d = p_d.to(device), p_v.to(device), c_d.to(device)
                with torch.amp.autocast('cuda'):
                    output = model(p_d, p_v, c_d)
                    total_val_loss += criterion(output, c_d).item()
                    total_val_zero += criterion(torch.zeros_like(c_d), c_d).item()
                    total_val_ident += criterion(p_d, c_d).item()
                    total_val_var += torch.var(output).item()
                    total_val_gt_var += torch.var(c_d).item()

        # Reports normalized MSE
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_zero = total_val_zero / len(val_loader)
        avg_val_ident = total_val_ident / len(val_loader)
        avg_train_var = total_train_var / len(train_loader)
        avg_val_var = total_val_var / len(val_loader)
        avg_val_gt_var = total_val_gt_var / len(val_loader)
        
        # Step the scheduler
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"    Train Loss: {avg_train_loss:.8f} | Val Loss: {avg_val_loss:.8f}")
        print(f"    Val Baselines: [Zero: {avg_val_zero:.8f}, Ident: {avg_val_ident:.8f}]")
        print(f"    Variance: [Train: {avg_train_var:.8f}, Val: {avg_val_var:.8f}, Val_GT: {avg_val_gt_var:.8f}]")
        print(f"    Val Max: [Pred: {output.max().item():.4f}, GT: {c_d.max().item():.4f}]")
        
        # Log to CSV
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.8f},{avg_val_loss:.8f},{avg_val_zero:.8f},{avg_val_ident:.8f},{avg_train_var:.8f},{avg_val_var:.8f}\n")
            
        torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
        # Also keep a copy in output_dir (run-collection root) for current state
        torch.save(model.state_dict(), os.path.join(output_dir, model_filename))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="attempts")
    parser.add_argument("--model_name", type=str, default="best_model.pth")
    args = parser.parse_args()
    train(args.epochs, args.data_dir, args.output_dir, args.model_name)
