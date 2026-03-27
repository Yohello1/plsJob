import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
from compressor import SPHDataset, FullModel

def sanity_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    session_dirs = [os.path.join("data", d) for d in os.listdir("data") if os.path.isdir(os.path.join("data", d)) and d != "frames"] if os.path.exists("data") else []
    if not session_dirs:
        print("No simulation data found.")
        return

    dataset = SPHDataset(session_dirs)
    if len(dataset) == 0:
        print("Dataset empty.")
        return
        
    dataloader = DataLoader(dataset, batch_size=4)
    model = FullModel().to(device)
    if os.path.exists("best_model.pth"):
        try:
            state_dict = torch.load("best_model.pth", map_location=device)
            # Handle DataParallel prefix
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            print("Loaded best_model.pth")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("best_model.pth not found.")
        return

    model.eval()
    with torch.no_grad():
        p_d, p_v, c_d = next(iter(dataloader))
        p_d, p_v, c_d = p_d.to(device), p_v.to(device), c_d.to(device)
        output = model(p_d, p_v, c_d)
        
        mse = torch.mean((output - c_d)**2).item()
        zero_mse = torch.mean(c_d**2).item()
        ident_mse = torch.mean((p_d - c_d)**2).item()
        
        print("\n--- Model Sanity Check ---")
        print(f"Current Model MSE: {mse:.8f}")
        print(f"Zero-Guess MSE:    {zero_mse:.8f}")
        print(f"Identity-Guess MSE: {ident_mse:.8f}")
        
        print(f"\nStats for one batch:")
        print(f"Pred Max: {output.max().item():.4f}, GT Max: {c_d.max().item():.4f}")
        print(f"Pred Min: {output.min().item():.4f}, GT Min: {c_d.min().item():.4f}")
        print(f"Pred Var: {torch.var(output).item():.8f}, GT Var: {torch.var(c_d).item():.8f}")
        
        if mse < zero_mse:
            print("\nRESULT: Model is performing BETTER than predicting all-zeros.")
        else:
            print("\nRESULT: Model is NOT performing better than predicting all-zeros.")
            
        if mse < ident_mse:
            print("RESULT: Model is performing BETTER than predicting previous frame.")
        else:
            print("RESULT: Model is NOT performing better than predicting previous frame.")

if __name__ == "__main__":
    sanity_check()
