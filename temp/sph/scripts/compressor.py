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
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.frame_indices = []
        d_files = glob.glob(os.path.join(data_dir, "*_d.bin"))
        for f in d_files:
            try:
                idx = int(os.path.basename(f).split('_')[0])
                self.frame_indices.append(idx)
            except (ValueError, IndexError):
                continue
        self.frame_indices.sort()
        self.samples = []
        for i in range(len(self.frame_indices) - 1):
            if self.frame_indices[i+1] == self.frame_indices[i] + 1:
                self.samples.append((self.frame_indices[i], self.frame_indices[i+1]))

    def __len__(self):
        return len(self.samples)

    def load_bin(self, frame_idx, suffix, dtype=np.float32):
        path = os.path.join(self.data_dir, f"{frame_idx}_{suffix}.bin")
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
        prev_idx, curr_idx = self.samples[idx]
        prev_d = self.load_bin(prev_idx, "d")
        prev_vx = self.load_bin(prev_idx, "v_x")
        prev_vy = self.load_bin(prev_idx, "v_y")
        curr_d = self.load_bin(curr_idx, "d")
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

    dataset = SPHDataset(session_dirs[0]) # Simplified; in practice you'd combine sessions
    # Multi-session data loading logic could be added here
    
    # Scale batch size
    batch_per_gpu = 8 # Reduced from 16 to save memory
    total_batch = max(1, batch_per_gpu * num_gpus)
    dataloader = DataLoader(dataset, batch_size=total_batch, shuffle=True, num_workers=4, pin_memory=True)
    
    model = FullModel().to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model)
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    epochs = requested_epochs if requested_epochs else 50
    for epoch in range(epochs):
        model.train()
        for p_d, p_v, c_d in dataloader:
            p_d, p_v, c_d = p_d.to(device), p_v.to(device), c_d.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                output = model(p_d, p_v, c_d)
                loss = criterion(output, c_d)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Reports normalized MSE
        print(f"Epoch {epoch+1}/{epochs}, Normalized Loss: {loss.item():.8f}")
        torch.save(model.state_dict(), "best_model.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train(args.epochs)
