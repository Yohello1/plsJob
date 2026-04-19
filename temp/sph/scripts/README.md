# SPH Autoencoder-Compression Scripts

This directory contains the tools and scripts for the active learning loop, data generation, and model evaluation of the SPH fluid simulation compression project.

## 🚀 Main Workflow

### `active_train_parallel.sh`
The primary entry point for the **Active Learning Loop**.
-   **Function**: Orchestrates multiple cycles of parallel SPH simulations followed by a model training session.
-   **Slurm Ready**: Configured for the Slurm scheduler (`sbatch`).
-   **Key Settings**: Adjust `ITERATIONS`, `RUNS_PER_ITERATION`, and `EPOCHS_PER_CLEAN` within the script.
-   **Data Paths**: Automatically stores simulation results in `data/run_TIMESTAMP/`.

---

## 🏗️ Model & Training

### `compressor.py`
The core **PyTorch Autoencoder** implementation.
-   **Architecture**: Uses an **Encoder**-**Decoder** structure with **Residual Blocks** and a massive **1024-dimension latent vector**.
-   **Absolute Prediction**: Predicts the entire next density grid from scratch (instead of deltas) to ensure stability at high skip-counts.
-   **Weighted MSE**: Uses a **50.0x fluid weight** multiplier to force the model to prioritize actual fluid particles over empty vacuum pixels.
-   **Activation**: Uses **Sigmoid** for the final output to bound density perfectly between 0.0 and 1.0.

### `spawn_random.sh`
Individual data generation script.
-   **Function**: A wrapper for the C++ SPH binary.
-   **Randomization**: Spawns randomized fluid boxes into the simulation each time it's called, providing diverse training data.

---

## 🔍 Validation & Visualization

### `test_reconstruct.py`
Quickly visualize model performance.
-   **Command**: `./env/bin/python test_reconstruct.py --model attempts/... --data data/...`
-   **Outputs**: Generates a 4-panel image (`reconstruction.png`) showing:
    1.  Previous Frame (Ref)
    2.  Ground Truth
    3.  Model Prediction
    4.  Absolute Error Map (Heatmap of mistakes)

### `check_model.py`
Numerical sanity check.
-   **Function**: Compares the current model's MSE against two baselines:
    -   **Zero Baseline**: Simple all-zero prediction.
    -   **Identity Baseline**: Copying the previous frame exactly.
-   **Goal**: The model should be significantly better (lower MSE) than both baselines.

### `grapher.py`
Performance analyzer.
-   **Function**: Parses simulation logs to track and plot the **time per frame**.
-   **Outputs**: Saves a performance analysis graph (`frame_time_analysis.png`).

---

## 🛠️ Environment

-   **`requirements.txt`**: Minimal requirements (`torch`, `numpy`, `matplotlib`).
-   **`env/`**: Your local virtual environment (recommended).

## 📂 Output Structure
1.  **`data/run_.../`**: Stores raw binary simulation data.
2.  **`logs/run_.../`**: Stores individual simulation logs and slurm outputs.
3.  **`attempts/run_.../`**: Stores model checkpoints (`best_model.pth`), loss curves (`losses.csv`), and training settings.

---

## 📊 Data Analysis

### 📈 Reading Losses (`losses.csv`)
The `losses.csv` file is updated every epoch. You can monitor it using `tail -f` or plot it with `matplotlib`.
- **`train_loss` / `val_loss`**: The total weighted loss (Hybrid Loss).
- **`mse_f`**: Mean Squared Error on **fluid pixels only**. This is your main quality metric.
- **`mse_b`**: MSE on the background. Should stay very low.
- **`grad`**: Gradient consistency loss (edge sharpness).
- **`mass`**: Physics violation penalty (fluid disappearance/creation).
- **`zero`**: Baseline MSE if the model simply predicted nothing.
- **`ident`**: Baseline MSE if the model simply copied the previous frame.
> [!TIP]
> **Success Criteria**: `mse_f` must be significantly lower than `zero` and `ident`. If `mse_f` approaches `zero`, the model is collapsing/blurring.

### 💾 Reading Raw Simulation Data (`sim_data.bin`)
Simulation data is stored in a flat binary format for maximum IOPS. Each frame consists of **4 contiguous float32 fields** of size **400x400**.

**Field Order**:
1.  **Density** (`[0, 1]`)
2.  **Velocity X**
3.  **Velocity Y**
4.  **Fluid Mask** (0 or 1)

**Python Reading Example**:
```python
import numpy as np

# Dimensions
W, H = 400, 400
frame_size = 4 * W * H * 4 # 4 fields * 400 * 400 * 4 bytes

with open("sim_data.bin", "rb") as f:
    # Read frame 0
    data = np.fromfile(f, dtype=np.float32, count=4 * W * H)
    data = data.reshape((4, H, W))
    
    density = data[0]
    velocity_x = data[1]
    velocity_y = data[2]
    mask = data[3]
```
