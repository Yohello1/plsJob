import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import sys
import random
import gc
from torch.utils.checkpoint import checkpoint

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
        # Using GroupNorm instead of BatchNorm for better stability (especially with batch_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c), # 8 groups is a robust default
            ACT(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.GroupNorm(8, c)
        )
    def _inner_forward(self, x):
        return ACT()(x + self.conv(x))
        
    def forward(self, x):
        if self.training:
            return checkpoint(self._inner_forward, x, use_reentrant=False)
        return self._inner_forward(x)

class SPHDataset(Dataset):
    def __init__(self, data_dirs, skip=10):
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        self.data_dirs = data_dirs
        self.samples = []
        self.skip = skip
        
        self.frame_size = 4 * BUFFER_WIDTH * BUFFER_HEIGHT * 4 # 4 fields * N * 4 bytes
        self.field_size = BUFFER_WIDTH * BUFFER_HEIGHT * 4
        self.order = {"d": 0, "v_x": 1, "v_y": 2, "m": 3}

        for d_dir in self.data_dirs:
            bin_file = os.path.join(d_dir, "sim_data.bin")
            if not os.path.exists(bin_file):
                continue
            
            file_size = os.path.getsize(bin_file)
            num_frames = file_size // self.frame_size
            
            for i in range(num_frames - self.skip):
                # Store (directory, prev_idx, curr_idx)
                self.samples.append((d_dir, i, i + self.skip))

    def __len__(self):
        return len(self.samples)

    def load_bin(self, data_dir, frame_idx, suffix, dtype=np.float32):
        field_offset = self.order[suffix] * self.field_size
        offset = frame_idx * self.frame_size + field_offset
        
        path = os.path.join(data_dir, "sim_data.bin")
        # Optimization: keep files open or use memmap if needed, 
        # but seek + fromfile is a good start for IOPS improvement
        with open(path, "rb") as f:
            f.seek(offset)
            data = np.fromfile(f, dtype=dtype, count=BUFFER_WIDTH * BUFFER_HEIGHT)
            
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
        
        # Load Previous Frame
        p_d = self.load_bin(data_dir, prev_idx, "d")
        p_v = torch.cat([self.load_bin(data_dir, prev_idx, "v_x"), 
                       self.load_bin(data_dir, prev_idx, "v_y")], dim=0)
        
        # Load Current Frame
        c_d = self.load_bin(data_dir, curr_idx, "d")
        c_v = torch.cat([self.load_bin(data_dir, curr_idx, "v_x"), 
                       self.load_bin(data_dir, curr_idx, "v_y")], dim=0)
        
        # Load Solid Mask
        mask = self.load_bin(data_dir, prev_idx, "m")
        
        return p_d, p_v, c_d, c_v, mask

class Encoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        # 400x7 -> 200x64 -> 100x128 -> 50x128
        self.conv = nn.Sequential(
            nn.Conv2d(7, 64, 3, stride=2, padding=1),   # Input channels: p_d(1), p_v(2), c_d(1), c_v(2), mask(1)
            ResBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), 
            ResBlock(128),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
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
        
        # Downsamples prev_d + mask to 50x50 bottleneck
        self.prev_context_path = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=2, padding=1),   # 200 (prev_d + mask)
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
            # Final output will be added to prev_d, so no activation here
        )
        self.final_act = nn.Sigmoid()

    def forward(self, z, prev_d, mask):
        z_feat = self.fc(z).view(-1, 128, 50, 50)
        # Contextual input: tell the decoder where the fluid was AND where the walls are
        context_feat = self.prev_context_path(torch.cat([prev_d, mask], dim=1))
        
        # Absolute Prediction: Reconstruction using both latent info and spatial context
        raw_output = self.deconv(torch.cat([z_feat, context_feat], dim=1))
        return self.final_act(raw_output)

class FullModel(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    def forward(self, p_d, p_v, c_d, c_v, mask):
        z = self.encoder(torch.cat([p_d, p_v, c_d, c_v, mask], dim=1))
        return self.decoder(z, p_d, mask)

    def get_depth(self):
        count = 0
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                count += 1
        return count

def find_max_batch_size(model, device, is_bf16=False):
    """Auto-detects the largest power-of-2 (or multiple) batch size that fits in VRAM."""
    print("Auto-detecting maximum possible batch size...")
    torch.cuda.empty_cache()
    gc.collect()
    
    # Mock inputs matching FullModel.forward(p_d, p_v, c_d, c_v, mask)
    p_d = torch.randn(1, 1, BUFFER_HEIGHT, BUFFER_WIDTH).to(device)
    p_v = torch.randn(1, 2, BUFFER_HEIGHT, BUFFER_WIDTH).to(device)
    c_d = torch.randn(1, 1, BUFFER_HEIGHT, BUFFER_WIDTH).to(device)
    c_v = torch.randn(1, 2, BUFFER_HEIGHT, BUFFER_WIDTH).to(device)
    mask = torch.randn(1, 1, BUFFER_HEIGHT, BUFFER_WIDTH).to(device)
    
    if is_bf16:
        p_d, p_v, c_d, c_v, mask = [t.to(torch.bfloat16) for t in [p_d, p_v, c_d, c_v, mask]]

    model.train()
    found_batch = 1
    # Try common tensor-core friendly batch sizes
    candidates = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    
    for b in candidates:
        try:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16 if is_bf16 else torch.float16):
                # Using expand avoids actual memory copy until the forward pass
                out = model(p_d.expand(b, -1, -1, -1), 
                            p_v.expand(b, -1, -1, -1),
                            c_d.expand(b, -1, -1, -1),
                            c_v.expand(b, -1, -1, -1),
                            mask.expand(b, -1, -1, -1))
                loss = out.sum()
            loss.backward()
            model.zero_grad(set_to_none=True)
            found_batch = b
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                break
            else:
                raise e
    
    print(f"Max batch size found: {found_batch}")
    torch.cuda.empty_cache()
    gc.collect()
    return found_batch

def train(requested_epochs=None, data_dir="data", output_dir="attempts", model_filename="best_model.pth", fluid_weight=50.0, mass_loss_weight=0.0, args=None):
    # Determine effective mass loss weight based on curriculum
    current_cycle = args.cycle if args and hasattr(args, 'cycle') else 1
    start_cycle = args.mass_loss_start_cycle if args and hasattr(args, 'mass_loss_start_cycle') else 1
    
    effective_mass_weight = mass_loss_weight if current_cycle >= start_cycle else 0.0
    if effective_mass_weight != mass_loss_weight:
        print(f"Curriculum: Mass loss weight delayed (current cycle {current_cycle} < start cycle {start_cycle}) | Effective Weight: {effective_mass_weight}")
    else:
        print(f"Curriculum: Mass loss weight active (Cycle {current_cycle} >= {start_cycle}) | Effective Weight: {effective_mass_weight}")

    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Detected {num_gpus} GPU(s). Using device: {device}")

    # Use specified data_dir
    session_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != "frames"] if os.path.exists(data_dir) else []
    if not session_dirs:
        print(f"No simulation data found in {data_dir}.")
        return

    # 1. Configuration & Constants
    LR = 5e-5
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
    
    # Precision Control: BF16 saves ~2.4GB on weights/grads for this model
    is_bf16 = args.bf16 if args and hasattr(args, 'bf16') else False
    if is_bf16 and torch.cuda.is_bf16_supported():
        model.to(torch.bfloat16)
        # Keep normalization layers in float32 for stability and type compatibility
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
                m.float()
        print("Enabled BFloat16 Precision (Norm layers kept in Float32 for stability)")
    else:
        is_bf16 = False
    
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
        f.write(f"Loss Function: WeightedMSE (Fluid Weight: {fluid_weight})\n")
        f.write(f"Normalizations: Density={DENSITY_NORM}, Velocity={VELOCITY_NORM}\n")
        f.write(f"Cycle: {current_cycle}\n")
        f.write(f"Mass Loss Weight (Target): {mass_loss_weight}\n")
        f.write(f"Mass Loss Start Cycle: {start_cycle}\n")
        f.write(f"Effective Mass Loss Weight: {effective_mass_weight}\n")
        f.write(f"Model Depth (Conv Layers): {model_depth}\n")

    # 5. Data Pipelines
    # Automatically handle batch size for local training
    batch_size = args.batch_size if args and hasattr(args, 'batch_size') else 1
    effective_batch = args.effective_batch_size if args and hasattr(args, 'effective_batch_size') else 8
    
    if batch_size == 0:
        # Find limit on a single GPU first, then scale
        found_limit = find_max_batch_size(model, device, is_bf16)
        batch_size = found_limit * max(1, num_gpus)
        print(f"Final training batch size set to: {batch_size} ({found_limit} per GPU)")
        # If batch size is already large enough, skip accumulation
        effective_batch = max(effective_batch, batch_size)
    
    accumulation_steps = max(1, effective_batch // batch_size)
    
    print(f"Batch Size: {batch_size} | Effective Batch: {effective_batch} (Accumulation Steps: {accumulation_steps})")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    if num_gpus > 1:
        model = nn.DataParallel(model)
        
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # Optional: Use Adam with lower precision or specific flags if still OOM
    # optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-4) 
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    def hybrid_loss(input, target, f_weight=fluid_weight, m_weight=effective_mass_weight):
        # Create mask for fluid vs background
        fluid_mask = (target > 0.05).float()
        background_mask = 1.0 - fluid_mask

        # Calculate MSE for both regions separately
        mse_fluid = torch.sum(fluid_mask * (input - target) ** 2) / (fluid_mask.sum() + 1e-6)
        mse_bg = torch.sum(background_mask * (input - target) ** 2) / (background_mask.sum() + 1e-6)

        # Combine them: f_weight now acts as a relative importance ratio
        # (e.g., 1.0 means fluid and background are equally important)
        mse_total = (f_weight * mse_fluid) + mse_bg

        # Global Mass Conservation (Normalized to Mean to match MSE scale)
        if m_weight > 0:
            mass_input = torch.mean(input, dim=(1, 2, 3))
            mass_target = torch.mean(target, dim=(1, 2, 3))
            mass_loss = torch.mean((mass_input - mass_target) ** 2)
            return mse_total + m_weight * mass_loss

        return mse_total

    criterion = hybrid_loss
    
    # BF16 doesn't need scaling
    if is_bf16 and torch.cuda.is_bf16_supported():
        scaler = None
    else:
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
        optimizer.zero_grad(set_to_none=True)
        
        for i, (p_d, p_v, c_d, c_v, mask) in enumerate(train_loader):
            p_d, p_v, c_d, c_v, mask = p_d.to(device), p_v.to(device), c_d.to(device), c_v.to(device), mask.to(device)
            
            if is_bf16 and torch.cuda.is_bf16_supported():
                p_d, p_v, c_d, c_v, mask = p_d.to(torch.bfloat16), p_v.to(torch.bfloat16), c_d.to(torch.bfloat16), c_v.to(torch.bfloat16), mask.to(torch.bfloat16)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16 if is_bf16 else torch.float16):
                output = model(p_d, p_v, c_d, c_v, mask)
                # Use float32 for loss stability
                loss = criterion(output.to(torch.float32), c_d.to(torch.float32))
                loss = loss / accumulation_steps
                
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
            total_train_loss += (loss.item() * accumulation_steps)
            total_train_var += torch.var(output).item()
            
            # Periodically Clear Fragments
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Handle leftover gradients if dataset size not divisible
        if (len(train_loader) % accumulation_steps) != 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # --- VALIDATION LOOP ---
        model.eval()
        total_val_loss = 0
        total_val_zero = 0
        total_val_ident = 0
        total_val_var = 0
        total_val_gt_var = 0
        with torch.no_grad():
            for p_d, p_v, c_d, c_v, mask in val_loader:
                p_d, p_v, c_d, c_v, mask = p_d.to(device), p_v.to(device), c_d.to(device), c_v.to(device), mask.to(device)
                
                if is_bf16:
                   p_d, p_v, c_d, c_v, mask = p_d.to(torch.bfloat16), p_v.to(torch.bfloat16), c_d.to(torch.bfloat16), c_v.to(torch.bfloat16), mask.to(torch.bfloat16)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16 if is_bf16 else torch.float16):
                    output = model(p_d, p_v, c_d, c_v, mask)
                    total_val_loss += criterion(output.float(), c_d.float()).item()
                    total_val_zero += criterion(torch.zeros_like(c_d).float(), c_d.float()).item()
                    total_val_ident += criterion(p_d.float(), c_d.float()).item()
                    total_val_var += torch.var(output.float()).item()
                    total_val_gt_var += torch.var(c_d.float()).item()
                
                torch.cuda.empty_cache()

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
    parser.add_argument("--cycle", type=int, default=1, help="Current active learning cycle index")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="attempts")
    parser.add_argument("--model_name", type=str, default="best_model.pth")
    parser.add_argument("--fluid_weight", type=float, default=25.0)
    parser.add_argument("--mass_loss_weight", type=float, default=0.0)
    parser.add_argument("--mass_loss_start_cycle", type=int, default=5, help="At what cycle to begin applying mass loss weight")
    parser.add_argument("--batch_size", type=int, default=0, help="0 = Auto-detect maximum for GPU, >0 = fixed size")
    parser.add_argument("--effective_batch_size", type=int, default=8, help="Target batch size for optimization steps (achieved via accumulation)")
    parser.add_argument("--bf16", action="store_true", help="Use BFloat16 precision for memory savings")
    args = parser.parse_args()
    train(args.epochs, args.data_dir, args.output_dir, args.model_name, args.fluid_weight, args.mass_loss_weight, args)

