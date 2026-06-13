#!/usr/bin/env python3
"""Train the model on a pre-generated dataset folder. Can be run multiple times."""

import os
import subprocess
import sys
import datetime
import argparse

DEFAULT_EPOCHS = 20
DEFAULT_MASS_LOSS_WEIGHT = 2.5
DEFAULT_FLUID_WEIGHT = 150.0
DEFAULT_NOISE_STD = 0.01
DEFAULT_AR_STEPS = 5

def main():
    parser = argparse.ArgumentParser(description="Train SPH autoencoder on a fixed dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset folder (from generate_data.py)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--output_dir", type=str, default=None, help="Where to save model/logs (default: attempts/TIMESTAMP)")
    parser.add_argument("--fluid_weight", type=float, default=DEFAULT_FLUID_WEIGHT)
    parser.add_argument("--mass_loss_weight", type=float, default=DEFAULT_MASS_LOSS_WEIGHT)
    parser.add_argument("--noise_std", type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument("--ar_steps", type=int, default=DEFAULT_AR_STEPS)
    parser.add_argument("--use_8bit_adam", action="store_true", default=True)
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        print(f"Error: data_dir '{data_dir}' does not exist.")
        sys.exit(1)

    attempts_dir = os.path.abspath(
        args.output_dir or f"attempts/train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(attempts_dir, exist_ok=True)

    print(f"Training on: {data_dir}")
    print(f"Output:      {attempts_dir}")

    train_cmd = [
        sys.executable, "compressor.py",
        "--cycle", "1",
        "--epochs", str(args.epochs),
        "--data_dir", data_dir,
        "--output_dir", attempts_dir,
        "--model_name", "best_model.pth",
        "--mass_loss_weight", str(args.mass_loss_weight),
        "--mass_loss_start_cycle", "1",
        "--fluid_weight", str(args.fluid_weight),
        "--batch_size", "0",
        "--effective_batch_size", "8",
        "--skip_frames", "5",
        "--n_steps", str(args.ar_steps),
        "--ar_start_cycle", "1",
        "--ar_increment_interval", "999",
        "--noise_std", str(args.noise_std),
        "--skip_initial", "5",
    ]

    if args.use_8bit_adam:
        train_cmd.append("--use_8bit_adam")

    subprocess.run(train_cmd, check=True)
    print(f"Training complete. Model saved to: {attempts_dir}/best_model.pth")

if __name__ == "__main__":
    main()
