import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
import argparse

sys.path.append(os.getcwd())
from compressor import FullModel, SPHDataset, LATENT_DIM, get_coord_grid, BUFFER_HEIGHT, BUFFER_WIDTH

def run_iterative_test(run_name, model_path, data_path, steps=6, skip=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_filename = os.path.basename(model_path)

    model = FullModel(LATENT_DIM).to(device)
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return
    print(f"Loading {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    if not os.path.exists(data_path):
        print(f"ERROR: Data path not found at {data_path}")
        return

    if os.path.exists(os.path.join(data_path, "sim_data.bin")):
        session_dirs = [data_path]
    else:
        session_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path)
                        if os.path.isdir(os.path.join(data_path, d)) and d != "frames"]

    if not session_dirs:
        print(f"ERROR: No valid simulations found in {data_path}")
        return

    dataset = SPHDataset(session_dirs, skip=skip, n_steps=1, augment=False)

    current_idx = 0
    p_d, p_v, c_ds, c_vs, mask = dataset[current_idx]

    # context_d starts as the first frame's density; context_v as its velocity
    context_d = p_d.unsqueeze(0).to(device)
    context_v = p_v.unsqueeze(0).to(device)
    mask_in   = mask.unsqueeze(0).to(device)

    predictions = []
    ground_truths = []

    print(f"Running {steps} iterative steps (Interval: {skip} frames)...")
    with torch.no_grad():
        for s in range(steps):
            p_d_b, p_v_b, c_ds_b, c_vs_b, _ = dataset[current_idx]

            # Ground-truth next frame (step 0 of the 1-step sequence)
            c_d_gt = c_ds_b[0].unsqueeze(0).to(device).float()
            c_v_gt = c_vs_b[0].unsqueeze(0).to(device).float()

            context_d = context_d.float()
            context_v = context_v.float()
            mask_in   = mask_in.float()

            # FullModel.forward: (p_d, p_v, c_d, c_v, mask) -> [B, 3, H, W]
            output = model(context_d, context_v, c_d_gt, c_v_gt, mask_in)

            pred_d = output[:, 0:1]   # density
            pred_v = output[:, 1:3]   # velocity

            predictions.append(pred_d.cpu().squeeze().numpy())
            ground_truths.append(c_d_gt.cpu().squeeze().numpy())

            # Autoregressive feedback
            context_d = pred_d
            context_v = pred_v
            current_idx += 1

    # --- Plotting ---
    fig, axes = plt.subplots(steps, 3, figsize=(18, 4 * steps))

    custom_cmap = copy.copy(plt.get_cmap('viridis'))
    custom_cmap.set_under('white')

    main_title = (f"Iterative Stability Test: {run_name}\n"
                  f"Model: {model_filename} | Steps: {steps} | Interval: {skip} Frames")
    plt.suptitle(main_title, fontsize=20, fontweight='bold', y=0.98)

    cols = ["Ground Truth", "Iterative Prediction", "Difference (Error)"]

    for i in range(steps):
        im0 = axes[i, 0].imshow(ground_truths[i], cmap=custom_cmap, vmin=1e-3, vmax=0.3)
        im1 = axes[i, 1].imshow(predictions[i],   cmap=custom_cmap, vmin=1e-3, vmax=0.3)
        diff = ground_truths[i] - predictions[i]
        im2 = axes[i, 2].imshow(diff, cmap='RdBu_r', vmin=-0.2, vmax=0.2)

        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04).set_label('Density', fontsize=9)
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04).set_label('Density', fontsize=9)
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04).set_label('Error',   fontsize=9)

        axes[i, 0].set_ylabel(f"Step {i+1}\n(Dataset idx {i})", fontsize=12, fontweight='bold')
        axes[i, 2].set_title(f"Max Resid: {np.abs(diff).max():.4f}", fontsize=11)

        if i == 0:
            for ax, col in zip(axes[0], cols):
                ax.set_title(col, fontsize=15, pad=15, fontweight='bold')

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    out_name = f"stability_{run_name.replace('/', '_')}.png"
    plt.savefig(out_name, dpi=150)
    print(f"Visualization saved to: {os.path.join(os.getcwd(), out_name)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",        type=str, default="SPH_Reconstruction", help="Title for the plot")
    parser.add_argument("--model_path", type=str, required=True,                help="Path to best_model.pth")
    parser.add_argument("--data_path",  type=str, required=True,                help="Path to data folder")
    parser.add_argument("--steps",      type=int, default=6,                    help="Number of iterative steps to visualize")
    parser.add_argument("--skip",       type=int, default=10,                   help="Frame skip used in dataset pairing")
    args = parser.parse_args()

    run_iterative_test(args.run, args.model_path, args.data_path, args.steps, args.skip)
