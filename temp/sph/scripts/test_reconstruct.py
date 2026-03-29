import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import random
from compressor import SPHDataset, FullModel

def visualize_reconstruction(model_path, data_dir, output_png="reconstruction.png", frame_idx=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Dataset
    session_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                    if os.path.isdir(os.path.join(data_dir, d)) and d != "frames"]
    if not session_dirs:
        print(f"No simulation sessions found in {data_dir}")
        return

    dataset = SPHDataset(session_dirs)
    if len(dataset) == 0:
        print("Dataset is empty. Check data path.")
        return

    # 2. Pick a frame
    if frame_idx is None:
        idx = random.randint(0, len(dataset) - 1)
    else:
        idx = min(frame_idx, len(dataset) - 1)
    
    p_d, p_v, c_d, c_v, mask = dataset[idx]
    
    # 3. Load Model
    # Note: FullModel defaults to latent_dim=1024 as per latest compressor.py
    model = FullModel(latent_dim=1024).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        # Strip DataParallel prefix if present
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    # 4. Inference
    with torch.no_grad():
        # Add batch dimension
        p_d_in = p_d.unsqueeze(0).to(device)
        p_v_in = p_v.unsqueeze(0).to(device)
        c_d_in = c_d.unsqueeze(0).to(device)
        c_v_in = c_v.unsqueeze(0).to(device)
        mask_in = mask.unsqueeze(0).to(device)
        
        pred_d = model(p_d_in, p_v_in, c_d_in, c_v_in, mask_in)
        pred_d = pred_d.squeeze().cpu().numpy()
        gt_d = c_d.squeeze().numpy()
        prev_d = p_d.squeeze().numpy()

    # 5. Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    titles = ["Previous Frame", "Ground Truth", "Model Prediction", "Diff (GT - Pred)"]
    datas = [prev_d, gt_d, pred_d, gt_d - pred_d]
    cmaps = ["viridis", "viridis", "viridis", "RdBu_r"]

    for i, (ax, data, title, cmap) in enumerate(zip(axes, datas, titles, cmaps)):
        im = ax.imshow(data, cmap=cmap, origin="lower")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")

    plt.suptitle(f"Reconstruction Test | Sample Index: {idx} | Path: {os.path.basename(model_path)}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    print(f"Saved visualization to {output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to weights")
    parser.add_argument("--data", type=str, default="data", help="Root data directory")
    parser.add_argument("--out", type=str, default="test_results.png", help="Output filename")
    parser.add_argument("--idx", type=int, default=None, help="Specific dataset index to test")
    
    args = parser.parse_args()
    
    visualize_reconstruction(args.model, args.data, args.out, args.idx)
