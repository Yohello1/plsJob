import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
import numpy as np
import os
import glob
import sys
import random
import gc
import inspect

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

# Constants based on SPH settings (Updated to 400x400)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
BUFFER_WIDTH = 400
BUFFER_HEIGHT = 400

# Normalization factors (default)
DENSITY_NORM = 1.0 / 0.020 
VELOCITY_NORM = 1.0 / 7.5 
LATENT_DIM = 1024
# Activation Configuration\
# Note: Try sigmoid & tanh, I lowkey think they will owkr better than any of these
# I think relu (and varients) loos too much data 
ACTIVATION_TYPE = "SiLU"
ACTIVATION_LOOKUP = {
    "ReLU": nn.ReLU,
    "SiLU": nn.SiLU,
    "LeakyReLU": lambda: nn.LeakyReLU(0.01),
    "ELU": nn.ELU
}
ACT = ACTIVATION_LOOKUP[ACTIVATION_TYPE]

# coord matrix thingy
def get_coord_grid(batch_size, h, w, device):
    yy = torch.linspace(-1, 1, h, device=device)
    xx = torch.linspace(-1, 1, w, device=device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=0) # [2, H, W]
    return grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

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
        return self._inner_forward(x)

class SPHDataset(Dataset):
    def __init__(self, data_dirs, skip=10, n_steps=1, skip_initial=1):
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        self.data_dirs = data_dirs
        self.samples = []
        self.skip = skip
        self.n_steps = n_steps
        
        self.frame_size = 4 * BUFFER_WIDTH * BUFFER_HEIGHT * 4 # 4 fields * N * 4 bytes
        self.field_size = BUFFER_WIDTH * BUFFER_HEIGHT * 4
        self.order = {"d": 0, "v_x": 1, "v_y": 2, "m": 3}
        self.handles = {}
        self.density_norm = 1.0
        self.velocity_norm = 1.0

        for d_dir in self.data_dirs:
            bin_file = os.path.join(d_dir, "sim_data.bin")
            if not os.path.exists(bin_file):
                continue
            
            file_size = os.path.getsize(bin_file)
            num_frames = file_size // self.frame_size
            
            for i in range(num_frames - self.skip * self.n_steps):
                self.samples.append((d_dir, i))
        
        if skip_initial > 1:
            self.samples = self.samples[::skip_initial]

    def set_norms(self, d_norm, v_norm):
        self.density_norm = d_norm
        self.velocity_norm = v_norm

    # literally half of this was to handle the shitty hardware Im on
    def _get_handle(self, data_dir):
        if data_dir not in self.handles:
            path = os.path.join(data_dir, "sim_data.bin")
            self.handles[data_dir] = open(path, "rb")
        return self.handles[data_dir]

    def load_frame_data(self, data_dir, frame_idx):
        handle = self._get_handle(data_dir)
        offset = frame_idx * self.frame_size
        handle.seek(offset)
        
        raw_data = np.fromfile(handle, dtype=np.float32, count=4 * BUFFER_WIDTH * BUFFER_HEIGHT)
        raw_data = raw_data.reshape((4, BUFFER_HEIGHT, BUFFER_WIDTH))
        
        # Unpack fields (no more multiple seeks!)
        d = torch.from_numpy(raw_data[0]).unsqueeze(0)
        v = torch.from_numpy(raw_data[1:3]) # v_x, v_y
        m = torch.from_numpy(raw_data[3]).unsqueeze(0)
        
        return d, v, m

    def __len__(self):
        return len(self.samples)

    # The last data bender did some data bending (augmentation)
    def __getitem__(self, idx):
        data_dir, start_idx = self.samples[idx]
        
        # Load Sequence Data
        frames_d = []
        frames_v = []
        
        # Random Physical Symmetry Flips (Data Augmentation)
        flip_h = random.random() > 0.5
        flip_v = random.random() > 0.5
        
        for step in range(self.n_steps + 1):
            f_idx = start_idx + step * self.skip
            d, v, m = self.load_frame_data(data_dir, f_idx)
            
            # Normalization
            d = d * self.density_norm
            v = v * self.velocity_norm
            
            # Apply physical symmetry with velocity correction
            if flip_h:
                d = torch.flip(d, [-1])
                v = torch.flip(v, [-1])
                v = torch.cat([-v[0:1], v[1:2]], dim=0) # Negate X-velocity for horizontal flip out-of-place
                m = torch.flip(m, [-1])
            if flip_v:
                d = torch.flip(d, [-2])
                v = torch.flip(v, [-2])
                v = torch.cat([v[0:1], -v[1:2]], dim=0) # Negate Y-velocity for vertical flip out-of-place
                m = torch.flip(m, [-2])
            
            frames_d.append(d)
            frames_v.append(v)
            if step == 0:
                mask = m.float()
        
        p_d = frames_d[0]
        p_v = frames_v[0]
        c_ds = torch.stack(frames_d[1:]) # [N, 1, H, W]
        c_vs = torch.stack(frames_v[1:]) # [N, 2, H, W]
        
        return p_d, p_v, c_ds, c_vs, mask

class Encoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(9, 64, 3, stride=2, padding=1),   
            ResBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), 
            ResBlock(128),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            ResBlock(128),
            nn.Flatten()
        )
        self.fc = nn.Linear(128 * 50 * 50, latent_dim)

    def forward(self, x):
        x = checkpoint(self.conv, x, use_reentrant=False)
        return self.fc(x)

class Decoder(nn.Module):
    # I lowkey just vibe coded parts of this for the sake of my brain energy
    def __init__(self, latent_dim=1024):
        super().__init__()
        # Bottleneck mapping
        self.fc = nn.Linear(latent_dim, 128 * 50 * 50)
        
        # STREAMING-FRIENDLY CONTEXT: Borrow sharp edges + absolute coordinates
        # Input channels: prev_d(1), mask(1), coord_x(1), coord_y(1) = 4
        self.context_400 = nn.Sequential(nn.Conv2d(4, 16, 3, padding=1), ACT())
        self.context_200 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), ACT())
        self.context_100 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), ACT())
        self.context_50  = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), ACT())
        
        # Sub-pixel Convolution (PixelShuffle) with context injection
        # 50x50 Stage
        self.up_50_to_100 = nn.Sequential(
            nn.Conv2d(128 + 128, 512, 3, padding=1), # (z + context_50)
            nn.PixelShuffle(2),                      # Output: 128 channels, 100x100
            ResBlock(128)
        )
        # 100x100 Stage
        self.up_100_to_200 = nn.Sequential(
            nn.Conv2d(128 + 64, 256, 3, padding=1), # (up_128 + context_100)
            nn.PixelShuffle(2),                      # Output: 64 channels, 200x200
            ResBlock(64)
        )
        # 200x200 Stage
        self.up_200_to_400 = nn.Sequential(
            nn.Conv2d(64 + 32, 12, 3, padding=1),  # 3 output channels (Density, Vx, Vy) * 2^2
            nn.PixelShuffle(2)                       # Output: 3 channels, 400x400
        )

    #check pointing like hell so it stops crashing mid run
    # might highkey just move to cpu training 
    def forward(self, z, prev_d, mask, coords):
        x = self.fc(z).view(-1, 128, 50, 50)
        
        c400 = checkpoint(self.context_400, torch.cat([prev_d, mask, coords], dim=1), use_reentrant=False)
        c200 = checkpoint(self.context_200, c400, use_reentrant=False)
        c100 = checkpoint(self.context_100, c200, use_reentrant=False)
        c50  = checkpoint(self.context_50,  c100, use_reentrant=False)
        
        x = checkpoint(self.up_50_to_100,  torch.cat([x, c50],   dim=1), use_reentrant=False)
        x = checkpoint(self.up_100_to_200, torch.cat([x, c100],  dim=1), use_reentrant=False)
        x = checkpoint(self.up_200_to_400, torch.cat([x, c200],  dim=1), use_reentrant=False)
        
        d = torch.sigmoid(x[:, 0:1])
        v = torch.tanh(x[:, 1:3]) # scaled Tanh activation for velocity
        
        return torch.cat([d, v], dim=1)

class FullModel(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    def forward(self, p_d, p_v, c_d, c_v, mask, noise_std=0.0):
        # Noise injection to improve stability against drift
        if self.training and noise_std > 0:
            p_d = p_d + torch.randn_like(p_d) * noise_std
            p_v = p_v + torch.randn_like(p_v) * noise_std

        coords = get_coord_grid(p_d.size(0), BUFFER_HEIGHT, BUFFER_WIDTH, p_d.device).to(p_d.dtype)
        
        # 9 channels: p_d(1), p_v(2), c_d(1), c_v(2), mask(1), coords(2)
        # We cat everything here; checkpointing the passes below will handle the rest
        x = torch.cat([p_d, p_v, c_d, c_v, mask, coords], dim=1)
        z = self.encoder(x)
        return self.decoder(z, p_d, mask, coords)

    def get_depth(self):
        count = 0
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                count += 1
        return count

def find_max_batch_size(model, device, n_steps=1, is_bf16=False):
    """Auto-detects the largest power-of-2 (or multiple) batch size that fits in VRAM, accounting for AR steps."""
    print(f"Auto-detecting maximum possible batch size (for {n_steps} AR steps)...")
    torch.cuda.empty_cache()
    gc.collect()
    
    # Mock inputs matching FullModel.forward(p_d, p_v, c_d, c_v, mask)
    p_d = torch.randn(1, 1, BUFFER_HEIGHT, BUFFER_WIDTH).to(device)
    p_v = torch.randn(1, 2, BUFFER_HEIGHT, BUFFER_WIDTH).to(device)
    c_ds = torch.randn(1, n_steps, 1, BUFFER_HEIGHT, BUFFER_WIDTH).to(device)
    c_vs = torch.randn(1, n_steps, 2, BUFFER_HEIGHT, BUFFER_WIDTH).to(device)
    mask = torch.randn(1, 1, BUFFER_HEIGHT, BUFFER_WIDTH).to(device)
    
    if is_bf16:
        p_d, p_v, c_ds, c_vs, mask = [t.to(torch.bfloat16) for t in [p_d, p_v, c_ds, c_vs, mask]]

    model.train()
    found_batch = 1
    # Try common tensor-core friendly batch sizes
    candidates = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    
    for b in candidates:
        try:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16 if is_bf16 else torch.float16):
                # Simulate the training loop AR rollout
                curr_p_d = p_d.expand(b, -1, -1, -1)
                curr_p_v = p_v.expand(b, -1, -1, -1)
                curr_mask = mask.expand(b, -1, -1, -1)
                
                total_loss = 0
                for s in range(n_steps):
                    c_d = c_ds.expand(b, -1, -1, -1, -1)[:, s]
                    c_v = c_vs.expand(b, -1, -1, -1, -1)[:, s]
                    
                    out = model(curr_p_d, curr_p_v, c_d, c_v, curr_mask)
                        
                    total_loss += out.sum()
                    curr_p_d = out[:, 0:1]
                    curr_p_v = out[:, 1:3]
                    
                loss = total_loss / n_steps
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

def get_global_stats(data_dirs):
    """Scans all binary files to find the maximum density for normalization."""
    print("Scanning dataset for global normalization factors...")
    max_density = 1e-6
    max_velocity = 1e-6
    
    for d_dir in data_dirs:
        bin_file = os.path.join(d_dir, "sim_data.bin")
        if not os.path.exists(bin_file):
            continue
            
        try:
            # Use memory mapping for high-speed scanning without loading into RAM
            m = np.memmap(bin_file, dtype=np.float32, mode='r')
            # Each frame is 4 fields of BUFFER_WIDTH * BUFFER_HEIGHT
            frame_elements = 4 * BUFFER_WIDTH * BUFFER_HEIGHT
            num_frames = m.size // frame_elements
            if num_frames == 0:
                continue
                
            m = m[:num_frames * frame_elements].reshape(-1, 4, BUFFER_HEIGHT, BUFFER_WIDTH)
            
            # Density is the first field (index 0)
            local_max_d = m[:, 0, :, :].max()
            max_density = max(max_density, local_max_d)
            
            # Velocity fields are index 1 and 2
            local_max_v = np.abs(m[:, 1:3, :, :]).max()
            max_velocity = max(max_velocity, local_max_v)
            
            del m # Close the memory map
        except Exception as e:
            print(f"Warning: Could not scan {bin_file} ({e})")
            
    print(f"Scan complete. Max Density: {max_density:.4f}, Max Velocity: {max_velocity:.4f}")
    return max_density, max_velocity

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
    
    # 2. Setup Run Directory
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

    # Determine current AR n_steps based on curriculum
    n_steps_limit = args.n_steps if args and hasattr(args, 'n_steps') else 1
    ar_start = args.ar_start_cycle if args and hasattr(args, 'ar_start_cycle') else 2
    ar_interval = args.ar_increment_interval if args and hasattr(args, 'ar_increment_interval') else 3
    
    current_n_steps = 1
    if current_cycle >= ar_start:
        current_n_steps = 1 + (current_cycle - ar_start) // ar_interval
    current_n_steps = min(current_n_steps, n_steps_limit)
    
    print(f"Curriculum: n_steps = {current_n_steps} (Limit: {n_steps_limit}, Cycle: {current_cycle})")

    # Calculate global normalization factors
    max_d, max_v = get_global_stats(session_dirs)
    density_norm = 1.0 / max_d
    velocity_norm = 1.0 / max_v
    print(f"Normalization: Density Scale={density_norm:.6f}, Velocity Scale={velocity_norm:.6f}")

    train_dataset = SPHDataset(train_dirs, skip=args.skip_frames if args else 10, n_steps=current_n_steps, skip_initial=args.skip_initial if args else 1)
    val_dataset = SPHDataset(val_dirs, skip=args.skip_frames if args else 10, n_steps=current_n_steps)
    
    train_dataset.set_norms(density_norm, velocity_norm)
    val_dataset.set_norms(density_norm, velocity_norm)
    
    skip_val = train_dataset.skip

    # Initialize model
    model = FullModel(LATENT_DIM_LOCAL).to(device)
    
    # Precision Control
    is_bf16 = args.bf16 if args and hasattr(args, 'bf16') else False
    if is_bf16 and torch.cuda.is_bf16_supported():
        model.to(torch.bfloat16)
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
                m.float()
        print("Enabled BFloat16 Precision (Norm layers kept in Float32 for stability)")
    else:
        is_bf16 = False

    # Load Persistent Weights
    pretrained_path = os.path.join(output_dir, model_filename)
    if os.path.exists(pretrained_path):
        try:
            print(f"Loading existing weights from {pretrained_path} (CPU -> GPU Transfer)...")
            state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            if is_bf16:
                state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            del state_dict 
        except Exception as e:
            print(f"Warning: Could not load pretrained weights ({e}). Starting from scratch.")
    else:
        print("No previous model found. Starting training from random initialization.")
    
    model_depth = model.get_depth()
    print(f"Model Depth: {model_depth} convolutional layers.")
    
    # Save Settings
    epochs = requested_epochs if requested_epochs else 10
    with open(os.path.join(run_dir, "settings.txt"), "w") as f:
        f.write(f"Attempt: {run_idx}\n")
        f.write(f"Learning Rate: {LR} (Scheduled)\n")
        f.write(f"Activation: {ACTIVATION_TYPE}\n")
        f.write(f"Latent Dim: {LATENT_DIM_LOCAL}\n")
        f.write(f"Grid Res: 400x400\n")
        f.write(f"Bottleneck: 50x50\n")
        f.write(f"Skip Frames: {skip_val}\n")
        f.write(f"AR Steps: {current_n_steps}\n")
        f.write(f"Noise Std: {args.noise_std if args else 0}\n")
        f.write(f"Epochs per Cycle: {epochs}\n")
        f.write(f"Train/Val Split: {len(train_dirs)}/{len(val_dirs)}\n")
        f.write(f"Loss Function: WeightedMSE (Fluid Weight: {fluid_weight})\n")
        f.write(f"Normalizations: Density={density_norm}, Velocity={velocity_norm}\n")
        f.write(f"Cycle: {current_cycle}\n")
        f.write(f"Mass Loss Weight (Target): {mass_loss_weight}\n")
        f.write(f"Mass Loss Start Cycle: {start_cycle}\n")
        f.write(f"Effective Mass Loss Weight: {effective_mass_weight}\n")
        f.write(f"Model Depth (Conv Layers): {model_depth}\n")

    # Data Pipelines
    batch_size = args.batch_size if args and hasattr(args, 'batch_size') else 1
    effective_batch = args.effective_batch_size if args and hasattr(args, 'effective_batch_size') else 8
    
    if batch_size == 0:
        found_limit = find_max_batch_size(model, device, current_n_steps, is_bf16)
        batch_size = found_limit * max(1, num_gpus)
        print(f"Final training batch size set to: {batch_size} ({found_limit} per GPU)")
        effective_batch = max(effective_batch, batch_size)
    
    accumulation_steps = max(1, effective_batch // batch_size)
    print(f"Batch Size: {batch_size} | Effective Batch: {effective_batch} (Accumulation Steps: {accumulation_steps})")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    if num_gpus > 1:
        model = nn.DataParallel(model)

    use_fused = 'fused' in inspect.signature(optim.AdamW).parameters
    print(f"Using standard AdamW optimizer (Fused={use_fused}).")
    optimizer = optim.AdamW(model.parameters(), lr=LR, fused=use_fused)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    def hybrid_loss(input, target, f_weight=fluid_weight, m_weight=effective_mass_weight, grad_weight=15.0, mean_weight=25.0):
        fluid_mask = (target[:, 0:1] > 0.05).float()
        background_mask = 1.0 - fluid_mask

        mse_fluid = torch.sum(fluid_mask * (input - target) ** 2) / (fluid_mask.sum() * input.shape[1] + 1e-6)
        mse_bg = torch.sum(background_mask * (input - target) ** 2) / (background_mask.sum() * input.shape[1] + 1e-6)
        mse_total = (f_weight * mse_fluid) + mse_bg

        # Penalise zero-collapse: force fluid-region mean to match target mean
        pred_fluid_mean = torch.sum(fluid_mask * input[:, 0:1]) / (fluid_mask.sum() + 1e-6)
        gt_fluid_mean   = torch.sum(fluid_mask * target[:, 0:1]) / (fluid_mask.sum() + 1e-6)
        mean_match_loss = (pred_fluid_mean - gt_fluid_mean) ** 2
        mse_total = mse_total + mean_weight * mean_match_loss

        in_dx = input[:, :, 1:, :] - input[:, :, :-1, :]
        in_dy = input[:, :, :, 1:] - input[:, :, :, :-1]
        tg_dx = target[:, :, 1:, :] - target[:, :, :-1, :]
        tg_dy = target[:, :, :, 1:] - target[:, :, :, :-1]
        f_mask_dx = fluid_mask[:, :, 1:, :]
        f_mask_dy = fluid_mask[:, :, :, 1:]
        
        grad_loss = (torch.sum(f_mask_dx * (in_dx - tg_dx)**2) / (f_mask_dx.sum() * input.shape[1] + 1e-6) +
                     torch.sum(f_mask_dy * (in_dy - tg_dy)**2) / (f_mask_dy.sum() * input.shape[1] + 1e-6))

        total_loss = mse_total + (grad_weight * grad_loss)

        if m_weight > 0:
            mass_input = torch.mean(input[:, 0:1], dim=(1, 2, 3))
            mass_target = torch.mean(target[:, 0:1], dim=(1, 2, 3))
            mass_loss = torch.mean((mass_input - mass_target) ** 2)
            total_loss += m_weight * mass_loss

        return total_loss

    criterion = hybrid_loss
    
    scaler = torch.amp.GradScaler('cuda')

    # Prepare loss logging
    log_file = os.path.join(run_dir, "losses.csv")
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss,val_zero,val_ident,train_var,val_var,grad_norm\n")

    epochs = requested_epochs if requested_epochs else 50
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # --- TRAINING LOOP (UPDATED FOR FULL BPTT) ---
        model.train()
        total_train_loss = 0
        total_train_var = 0
        total_grad_norm = 0
        grad_norm_steps = 0
        optimizer.zero_grad(set_to_none=True)
        
        for i, (p_d, p_v, c_ds, c_vs, mask) in enumerate(train_loader):
            p_d, p_v, c_ds, c_vs, mask = p_d.to(device), p_v.to(device), c_ds.to(device), c_vs.to(device), mask.to(device)
            
            if is_bf16:
                p_d, p_v, c_ds, c_vs, mask = p_d.to(torch.bfloat16), p_v.to(torch.bfloat16), c_ds.to(torch.bfloat16), c_vs.to(torch.bfloat16), mask.to(torch.bfloat16)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16 if is_bf16 else torch.float16):
                # Autoregressive Rollout Loop
                total_sequence_loss = 0
                noise_std = getattr(args, 'noise_std', 0.0)
                
                # Rollout sequence without detaching intermediate steps
                for step in range(current_n_steps):
                    curr_d_gt = c_ds[:, step]
                    curr_v_gt = c_vs[:, step]
                    
                    output = model(p_d, p_v, curr_d_gt, curr_v_gt, mask, noise_std=noise_std)
                        
                    pred_d = output[:, 0:1]
                    pred_v = output[:, 1:3]
                    target_combined = torch.cat([curr_d_gt, curr_v_gt], dim=1)
                    
                    # Density + Velocity Loss (Hybrid)
                    # Note: Divide by current_n_steps * accumulation_steps for correct averaging
                    step_loss = criterion(output.float(), target_combined.float()) / (current_n_steps * accumulation_steps)
                    
                    total_sequence_loss = total_sequence_loss + step_loss
                    
                    # Update for next step in rollout (AR)
                    # BPTT: Pass un-detached predictions
                    p_d = pred_d
                    
                    if step + 1 < current_n_steps:
                        p_v = pred_v
                
                loss_val = total_sequence_loss.item()
            
            # Backpropagate sequence dependencies globally
            if scaler:
                scaler.scale(total_sequence_loss).backward()
            else:
                total_sequence_loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                total_grad_norm += grad_norm.item()
                grad_norm_steps += 1
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
            total_train_loss += (loss_val * accumulation_steps * current_n_steps)
            total_train_var += torch.var(pred_d).item()
            
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        if (len(train_loader) % accumulation_steps) != 0:
            if scaler:
                scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_grad_norm += grad_norm.item()
            grad_norm_steps += 1
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # --- VALIDATION LOOP ---
        model.eval()
        total_val_loss = 0
        total_val_loss_steps = [0.0] * current_n_steps
        total_val_zero = 0
        total_val_ident = 0
        total_val_var = 0
        total_val_gt_var = 0
        
        with torch.no_grad():
            for p_d, p_v, c_ds, c_vs, mask in val_loader:
                p_d, p_v, c_ds, c_vs, mask = p_d.to(device), p_v.to(device), c_ds.to(device), c_vs.to(device), mask.to(device)
                if is_bf16:
                   p_d, p_v, c_ds, c_vs, mask = p_d.to(torch.bfloat16), p_v.to(torch.bfloat16), c_ds.to(torch.bfloat16), c_vs.to(torch.bfloat16), mask.to(torch.bfloat16)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16 if is_bf16 else torch.float16):
                    p_d_init = p_d.clone()
                    p_v_init = p_v.clone()
                    batch_total_val = 0
                    
                    for step in range(current_n_steps):
                        curr_d_gt = c_ds[:, step]
                        curr_v_gt = c_vs[:, step]
                        
                        output = model(p_d, p_v, curr_d_gt, curr_v_gt, mask)
                        pred_d = output[:, 0:1]
                        pred_v = output[:, 1:3]
                        target_combined = torch.cat([curr_d_gt, curr_v_gt], dim=1)
                        
                        step_loss = criterion(output.float(), target_combined.float()).item()
                        total_val_loss_steps[step] += step_loss
                        batch_total_val += step_loss
                        # Prepare for next step (AR Rollout)
                        p_d = pred_d
                        if step + 1 < current_n_steps:
                            p_v = pred_v
                    
                    total_val_loss += (batch_total_val / current_n_steps)
                    
                    # Baselines use full 3-channel target (density + velocity) to match model loss
                    target_combined_0 = torch.cat([c_ds[:, 0], c_vs[:, 0]], dim=1)
                    zero_pred = torch.zeros_like(target_combined_0)
                    ident_pred = torch.cat([p_d_init, p_v_init], dim=1)
                    total_val_zero += criterion(zero_pred.float(), target_combined_0.float()).item()
                    total_val_ident += criterion(ident_pred.float(), target_combined_0.float()).item()
                    total_val_var += torch.var(pred_d.float()).item()
                    total_val_gt_var += torch.var(c_ds[:, 0].float()).item()
                
                torch.cuda.empty_cache()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_zero = total_val_zero / len(val_loader)
        avg_val_ident = total_val_ident / len(val_loader)
        avg_train_var = total_train_var / len(train_loader)
        avg_val_var = total_val_var / len(val_loader)
        avg_val_gt_var = total_val_gt_var / len(val_loader)
        avg_grad_norm = total_grad_norm / max(1, grad_norm_steps)
        
        avg_val_steps = [s / len(val_loader) for s in total_val_loss_steps]
        step_error_str = ", ".join([f"S{i+1}: {err:.4f}" for i, err in enumerate(avg_val_steps)])
        
        print("-" * 90)
        print(f"Epoch {epoch+1:02d}/{epochs:02d}")
        print(f"  > LOSSES:   Train: {avg_train_loss:.4f}  |  Val Rollout: {avg_val_loss:.4f}")
        print(f"  > STEPS:    {step_error_str}")
        print(f"  > BASELINES: Zero-Field Baseline: {avg_val_zero:.4f}  |  Identity (Static) Baseline: {avg_val_ident:.4f}")
        print(f"  > VARIANCE: Target GT Var: {avg_val_gt_var:.5f}  |  Model Output Var: {avg_val_var:.5f}  (Train Var: {avg_train_var:.5f})")
        print(f"  > GRAD NORM: {avg_grad_norm:.4f}")
        print("-" * 90)

        scheduler.step(avg_val_loss)
        
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_val_loss:.6f},{avg_val_zero:.6f},{avg_val_ident:.6f},{avg_train_var:.6f},{avg_val_var:.6f},{avg_grad_norm:.6f}\n")
            
        # Only save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if num_gpus > 1:
                torch.save(model.module.state_dict(), os.path.join(output_dir, model_filename))
            else:
                torch.save(model.state_dict(), os.path.join(output_dir, model_filename))
            
    print("Training Cycle Finished.")

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
    parser.add_argument("--skip_frames", type=int, default=10, help="Temporal skip between frames in sequences")
    parser.add_argument("--n_steps", type=int, default=1, help="Maximum number of auto-regressive steps to train on")
    parser.add_argument("--ar_start_cycle", type=int, default=2, help="Cycle index to begin multi-step AR curriculum")
    parser.add_argument("--ar_increment_interval", type=int, default=3, help="How many cycles to wait between increasing AR step count")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Standard deviation of Gaussian noise injected into inputs during training")
    parser.add_argument("--skip_initial", type=int, default=1, help="If > 1, sparsely samples frames during initialization to speed up cycles")
    parser.add_argument("--use_8bit_adam", action="store_true", default=True, help="Use BitsAndBytes 8-bit AdamW optimizer for VRAM savings")
    args = parser.parse_args()
    train(args.epochs, args.data_dir, args.output_dir, args.model_name, args.fluid_weight, args.mass_loss_weight, args)

