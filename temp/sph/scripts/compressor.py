import torch
import torch.nn as nn
import torch.nn.functional as F
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

def get_coord_grid(batch_size, h, w, device):
    """Generates X and Y coordinate channels ranging from -1 to 1."""
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
        if self.training:
            return checkpoint(self._inner_forward, x, use_reentrant=False)
        return self._inner_forward(x)

class SPHDataset(Dataset):
    def __init__(self, data_dirs, skip=10, n_steps=1):
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        self.data_dirs = data_dirs
        self.samples = []
        self.skip = skip
        self.n_steps = n_steps
        
        self.frame_size = 4 * BUFFER_WIDTH * BUFFER_HEIGHT * 4 # 4 fields * N * 4 bytes
        self.field_size = BUFFER_WIDTH * BUFFER_HEIGHT * 4
        self.order = {"d": 0, "v_x": 1, "v_y": 2, "m": 3}
        self.handles = {} # Local handle cache to avoid NFS re-opening
        self.skip_val = False

        for d_dir in self.data_dirs:
            bin_file = os.path.join(d_dir, "sim_data.bin")
            if not os.path.exists(bin_file):
                continue
            
            file_size = os.path.getsize(bin_file)
            num_frames = file_size // self.frame_size
            
            # Ensure we have enough frames for the AR rollout
            max_start = num_frames - (self.n_steps * self.skip)
            for i in range(max_start):
                self.samples.append((d_dir, i))
        
        if not self.samples:
            self.skip_val = True

    def _get_handle(self, data_dir):
        if data_dir not in self.handles:
            path = os.path.join(data_dir, "sim_data.bin")
            self.handles[data_dir] = open(path, "rb")
        return self.handles[data_dir]

    def load_frame_data(self, data_dir, frame_idx):
        """Optimized: Reads all 4 fields (p_d, v_x, v_y, mask) in one go if possible"""
        handle = self._get_handle(data_dir)
        offset = frame_idx * self.frame_size
        handle.seek(offset)
        
        # Read the entire 4-field block into memory at once
        raw_data = np.fromfile(handle, dtype=np.float32, count=4 * BUFFER_WIDTH * BUFFER_HEIGHT)
        raw_data = raw_data.reshape((4, BUFFER_HEIGHT, BUFFER_WIDTH))
        
        # Unpack fields (no more multiple seeks!)
        d = torch.from_numpy(raw_data[0]).unsqueeze(0)
        v = torch.from_numpy(raw_data[1:3]) # v_x, v_y
        m = torch.from_numpy(raw_data[3]).unsqueeze(0)
        
        return d, v, m

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
        data_dir, start_idx = self.samples[idx]
        
        densities = []
        velocities = []
        
        # Load n_steps + 1 frames
        for step in range(self.n_steps + 1):
            frame_idx = start_idx + step * self.skip
            d, v, m = self.load_frame_data(data_dir, frame_idx)
            
            # Apply normalization
            d = d * DENSITY_NORM
            v = v * VELOCITY_NORM
            
            densities.append(d)
            velocities.append(v)
            if step == 0:
                mask = m.float()
        
        # Return as tensors: [T+1, 1, H, W] and [T+1, 2, H, W]
        return torch.stack(densities), torch.stack(velocities), mask

class Encoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        # Input channels: p_d(1), p_v(2), c_d(1), c_v(2), mask(1) + COORD_X(1), COORD_Y(1) = 9
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
        return self.fc(self.conv(x))

class Decoder(nn.Module):
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
            nn.Conv2d(64 + 32, 4, 3, padding=1),   # (up_64 + context_200)
            nn.PixelShuffle(2)                       # Output: 1 channel, 400x400
        )
        
        self.final_act = nn.Sigmoid()

    def forward(self, z, prev_d, mask, coords):
        # 1. Map bottleneck to 50x50
        x = self.fc(z).view(-1, 128, 50, 50)
        
        # 2. Extract Context "Blueprints" from Previous Frame + Mask + Coords
        ctx_400 = self.context_400(torch.cat([prev_d, mask, coords], dim=1))
        ctx_200 = self.context_200(ctx_400)
        ctx_100 = self.context_100(ctx_200)
        ctx_50  = self.context_50(ctx_100)
        
        # 3. Upsample while injecting high-res context at each step
        x = self.up_50_to_100(torch.cat([x, ctx_50], dim=1))
        x = self.up_100_to_200(torch.cat([x, ctx_100], dim=1))
        # Inject the 200x200 details right before the final 400x400 expansion
        x = self.up_200_to_400(torch.cat([x, ctx_200], dim=1))
        
        return self.final_act(x)

class FullModel(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    def forward(self, p_d, p_v, c_d, c_v, mask):
        coords = get_coord_grid(p_d.size(0), BUFFER_HEIGHT, BUFFER_WIDTH, p_d.device).to(p_d.dtype)
        # 9 channels: p_d(1), p_v(2), c_d(1), c_v(2), mask(1), coords(2)
        z = self.encoder(torch.cat([p_d, p_v, c_d, c_v, mask, coords], dim=1))
        return self.decoder(z, p_d, mask, coords)

    def get_depth(self):
        count = 0
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                count += 1
        return count

def find_max_batch_size(model, device, is_bf16=False, ar_steps=1):
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
                # Simulate AR steps to check memory limit (BPTT style)
                context_d = p_d.expand(b, -1, -1, -1)
                total_loss = 0
                for _ in range(ar_steps):
                    out = model(context_d, 
                                p_v.expand(b, -1, -1, -1),
                                c_d.expand(b, -1, -1, -1),
                                c_v.expand(b, -1, -1, -1),
                                mask.expand(b, -1, -1, -1))
                    context_d = out 
                    total_loss += out.sum()
                
                total_loss.backward()
            
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
    
    # AR Curriculum Logic
    ar_target_steps = args.ar_steps if args and hasattr(args, 'ar_steps') else 1
    ar_start_cycle = args.ar_start_cycle if args and hasattr(args, 'ar_start_cycle') else 1
    ar_interval = args.ar_increment_interval if args and hasattr(args, 'ar_increment_interval') else 3
    
    if current_cycle < ar_start_cycle:
        effective_ar_steps = 1
    else:
        # Increment every N cycles: 1 + (cycles_since_start // interval)
        effective_ar_steps = 1 + (current_cycle - ar_start_cycle) // ar_interval
        effective_ar_steps = min(effective_ar_steps, ar_target_steps)

    if effective_mass_weight != mass_loss_weight:
        print(f"Curriculum: Mass loss weight delayed (current cycle {current_cycle} < start cycle {start_cycle}) | Effective Weight: {effective_mass_weight}")
    else:
        print(f"Curriculum: Mass loss weight active (Cycle {current_cycle} >= {start_cycle}) | Effective Weight: {effective_mass_weight}")
        
    print(f"Curriculum: AR steps (Cycle {current_cycle}) | Effective Steps: {effective_ar_steps} (Target: {ar_target_steps})")

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

    ar_steps = effective_ar_steps # Use curriculum-determined steps
    train_dataset = SPHDataset(train_dirs, n_steps=ar_steps)
    val_dataset = SPHDataset(val_dirs, n_steps=ar_steps)
    skip_val = train_dataset.skip

    # Initialize model
    model = FullModel(LATENT_DIM_LOCAL).to(device)
    
    # NEW: LOAD PERSISTENT WEIGHTS
    # This allows Active Learning to actually "build" on previous cycles
    pretrained_path = os.path.join(output_dir, model_filename)
    if os.path.exists(pretrained_path):
        try:
            print(f"Loading existing weights from {pretrained_path} (Incremental Learning)...")
            state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Could not load pretrained weights ({e}). Starting from scratch.")
    else:
        print("No previous model found. Starting training from random initialization.")
    
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
        f.write(f"AR Steps: {ar_steps}\n")
        f.write(f"Noise Std: {args.noise_std if args else 0.0}\n")

    # 5. Data Pipelines
    # Automatically handle batch size for local training
    batch_size = args.batch_size if args and hasattr(args, 'batch_size') else 1
    effective_batch = args.effective_batch_size if args and hasattr(args, 'effective_batch_size') else 8
    
    if batch_size == 0:
        # Find limit on a single GPU first, then scale
        found_limit = find_max_batch_size(model, device, is_bf16, ar_steps=ar_steps)
        batch_size = found_limit * max(1, num_gpus)
        print(f"Final training batch size set to: {batch_size} ({found_limit} per GPU)")
        # If batch size is already large enough, skip accumulation
        effective_batch = max(effective_batch, batch_size)
    
    accumulation_steps = max(1, effective_batch // batch_size)
    
    print(f"Batch Size: {batch_size} | Effective Batch: {effective_batch} (Accumulation Steps: {accumulation_steps})")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    if num_gpus > 1:
        model = nn.DataParallel(model)
        
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # Optional: Use Adam with lower precision or specific flags if still OOM
    # optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-4) 
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    def hybrid_loss(input, target, f_weight=fluid_weight, m_weight=effective_mass_weight, grad_weight=5.0, prev_frame=None):
        # 1. Spatial Masks: Separate the fluid from the background
        fluid_mask = (target > 0.05).float()
        background_mask = 1.0 - fluid_mask

        # 2. Weighted MSE: Focus on getting the fluid reconstruction right
        mse_f = torch.sum(fluid_mask * (input - target) ** 2) / (fluid_mask.sum() + 1e-6)
        mse_b = torch.sum(background_mask * (input - target) ** 2) / (background_mask.sum() + 1e-6)
        mse_total = (f_weight * mse_f) + mse_b

        # 3. Gradient Consistency: Forces the model to match sharp edges (High-frequency details)
        in_dx = input[:, :, 1:, :] - input[:, :, :-1, :]
        in_dy = input[:, :, :, 1:] - input[:, :, :, :-1]
        tg_dx = target[:, :, 1:, :] - target[:, :, :-1, :]
        tg_dy = target[:, :, :, 1:] - target[:, :, :, :-1]
        f_mask_dx = fluid_mask[:, :, 1:, :]
        f_mask_dy = fluid_mask[:, :, :, 1:]
        
        grad_l = (torch.sum(f_mask_dx * (in_dx - tg_dx)**2) / (f_mask_dx.sum() + 1e-6) +
                  torch.sum(f_mask_dy * (in_dy - tg_dy)**2) / (f_mask_dy.sum() + 1e-6))

        # 4. Global Mass Conservation: Ensures fluid doesn't disappear or appear from nowhere
        mass_l = torch.tensor(0.0, device=input.device)
        if m_weight > 0:
            mass_input = torch.mean(input, dim=(1, 2, 3))
            mass_target = torch.mean(target, dim=(1, 2, 3))
            mass_l = torch.mean((mass_input - mass_target) ** 2)

        # 5. Baselines (For human readability)
        with torch.no_grad():
            zero_l = F.mse_loss(torch.zeros_like(target), target).item()
            ident_l = F.mse_loss(prev_frame, target).item() if prev_frame is not None else 0.0

        # 6. Total Weighted Loss
        total_loss = mse_total + (grad_weight * grad_l) + (m_weight * mass_l)

        # 7. Metrics Dictionary for human-readable logging
        metrics = {
            "mse_fluid": mse_f.item(),
            "mse_bg": mse_b.item(),
            "grad": grad_l.item(),
            "mass": mass_l.item(),
            "zero": zero_l,
            "ident": ident_l
        }

        return total_loss, metrics

    criterion = hybrid_loss
    
    # BF16 doesn't need scaling
    if is_bf16 and torch.cuda.is_bf16_supported():
        scaler = None
    else:
        scaler = torch.amp.GradScaler('cuda')

    # Prepare loss logging
    log_file = os.path.join(run_dir, "losses.csv")
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss,mse_f,mse_b,grad,mass\n")

    best_val_loss = float('inf')
    epochs = requested_epochs if requested_epochs else 50
    for epoch in range(epochs):
        # --- TRAINING LOOP ---
        model.train()
        total_train_loss = 0
        epoch_metrics = {"mse_fluid": 0, "mse_bg": 0, "grad": 0, "mass": 0, "zero": 0, "ident": 0}
        optimizer.zero_grad(set_to_none=True)
        
        for i, (densities, velocities, mask) in enumerate(train_loader):
            densities, velocities, mask = densities.to(device), velocities.to(device), mask.to(device)
            
            # --- Physical Symmetry Augmentation (Flips) ---
            # Horizontal Flip
            if random.random() > 0.5:
                densities = torch.flip(densities, [-1])
                velocities = torch.flip(velocities, [-1])
                # Flip X-velocity component (index 0)
                velocities[:, :, 0] *= -1
                mask = torch.flip(mask, [-1])
            # Vertical Flip
            if random.random() > 0.5:
                densities = torch.flip(densities, [-2])
                velocities = torch.flip(velocities, [-2])
                # Flip Y-velocity component (index 1)
                velocities[:, :, 1] *= -1
                mask = torch.flip(mask, [-2])
            
            if is_bf16 and torch.cuda.is_bf16_supported():
                densities, velocities, mask = densities.to(torch.bfloat16), velocities.to(torch.bfloat16), mask.to(torch.bfloat16)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16 if is_bf16 else torch.float16):
                loss = 0
                context_d = densities[:, 0] # Initial previous density
                
                # Noise Injection: Add noise to the initial context
                if args and args.noise_std > 0:
                    context_d = context_d + torch.randn_like(context_d) * args.noise_std

                for step in range(ar_steps):
                    # Inputs for this step
                    p_v = velocities[:, step]
                    c_d = densities[:, step + 1]
                    c_v = velocities[:, step + 1]
                    
                    # Noise Injection for GT inputs
                    if args and args.noise_std > 0:
                        p_v = p_v + torch.randn_like(p_v) * args.noise_std
                        c_d_in = c_d + torch.randn_like(c_d) * args.noise_std
                        c_v_in = c_v + torch.randn_like(c_v) * args.noise_std
                    else:
                        c_d_in = c_d
                        c_v_in = c_v

                    output = model(context_d, p_v, c_d_in, c_v_in, mask)
                    
                    # Pass previous frame (densities[:, step]) for Identity baseline calculation
                    step_loss, step_metrics = criterion(output.to(torch.float32), c_d.to(torch.float32), prev_frame=densities[:, step])
                    loss += step_loss
                    for k in epoch_metrics:
                        epoch_metrics[k] += step_metrics[k]
                    
                    # Feedback for next step (Full BPTT: No detaching)
                    context_d = output
                    
                    # Optional: Add noise to context for next step to simulate drift
                    if step < ar_steps - 1 and args and args.noise_std > 0:
                        context_d = context_d + torch.randn_like(context_d) * args.noise_std

                loss = loss / (accumulation_steps * ar_steps)
                
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
            total_train_loss += (loss.item() * accumulation_steps * ar_steps)
            
            # Periodically Clear Fragments
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Handle leftover gradients if dataset size not divisible
        if (len(train_loader) % accumulation_steps) != 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # --- VALIDATION LOOP ---
        model.eval()
        total_val_loss = 0
        val_metrics = {"mse_fluid": 0, "mse_bg": 0, "grad": 0, "mass": 0, "zero": 0, "ident": 0}
        with torch.no_grad():
            for densities, velocities, mask in val_loader:
                densities, velocities, mask = densities.to(device), velocities.to(device), mask.to(device)
                
                if is_bf16:
                   densities, velocities, mask = densities.to(torch.bfloat16), velocities.to(torch.bfloat16), mask.to(torch.bfloat16)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16 if is_bf16 else torch.float16):
                    context_d = densities[:, 0]
                    batch_loss = 0
                    for step in range(ar_steps):
                        p_v = velocities[:, step]
                        c_d = densities[:, step + 1]
                        c_v = velocities[:, step + 1]
                        
                        output = model(context_d, p_v, c_d, c_v, mask)
                        step_loss, step_metrics = criterion(output.float(), c_d.float(), prev_frame=densities[:, step])
                        batch_loss += step_loss.item()
                        for k in val_metrics:
                            val_metrics[k] += step_metrics[k]
                        
                        context_d = output
                    
                    total_val_loss += (batch_loss / ar_steps)
                
                torch.cuda.empty_cache()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Calculate metric averages (normalized by number of batches and AR steps)
        for k in epoch_metrics: epoch_metrics[k] /= (len(train_loader) * ar_steps)
        for k in val_metrics: val_metrics[k] /= (len(val_loader) * ar_steps)

        # Step the scheduler
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"    Train Loss: {avg_train_loss:.8f} | Val Loss: {avg_val_loss:.8f}")
        print(f"    Metrics (Val): [Fluid_MSE: {val_metrics['mse_fluid']:.6f}, BG_MSE: {val_metrics['mse_bg']:.6f}, Grad: {val_metrics['grad']:.6f}, Mass: {val_metrics['mass']:.6f}]")
        print(f"    Baselines (Val): [Zero: {val_metrics['zero']:.6f}, Ident: {val_metrics['ident']:.6f}]")
        
        # Log to CSV
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.8f},{avg_val_loss:.8f},{val_metrics['mse_fluid']:.8f},{val_metrics['mse_bg']:.8f},{val_metrics['grad']:.8f},{val_metrics['mass']:.8f},{val_metrics['zero']:.8f},{val_metrics['ident']:.8f}\n")

        # Save Best Model logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"    *** New Best Val Loss: {best_val_loss:.8f} (Saved) ***")
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            # Also keep a copy in output_dir (run-collection root) for current state
            torch.save(model.state_dict(), os.path.join(output_dir, model_filename))
        else:
            print(f"    (Best Val Loss remained: {best_val_loss:.8f})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, default=1, help="Current active learning cycle index")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="attempts")
    parser.add_argument("--model_name", type=str, default="best_model.pth")
    parser.add_argument("--fluid_weight", type=float, default=12.0)
    parser.add_argument("--mass_loss_weight", type=float, default=0.0)
    parser.add_argument("--mass_loss_start_cycle", type=int, default=5, help="At what cycle to begin applying mass loss weight")
    parser.add_argument("--batch_size", type=int, default=0, help="0 = Auto-detect maximum for GPU, >0 = fixed size")
    parser.add_argument("--effective_batch_size", type=int, default=8, help="Target batch size for optimization steps (achieved via accumulation)")
    parser.add_argument("--bf16", action="store_true", help="Use BFloat16 precision for memory savings")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Standard deviation of Gaussian noise to inject during training")
    parser.add_argument("--ar_steps", type=int, default=1, help="Target number of autoregressive steps to train for (maximum)")
    parser.add_argument("--ar_start_cycle", type=int, default=1, help="Cycle at which to start increasing AR steps")
    parser.add_argument("--ar_increment_interval", type=int, default=3, help="How many cycles to wait between increasing AR steps")
    args = parser.parse_args()
    train(args.epochs, args.data_dir, args.output_dir, args.model_name, args.fluid_weight, args.mass_loss_weight, args)

