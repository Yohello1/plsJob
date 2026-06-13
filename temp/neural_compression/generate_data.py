#!/usr/bin/env python3
"""Generate N SPH simulations into a dataset folder. Run once."""

import os
import subprocess
import time
import datetime
import argparse

DEFAULT_N = 40
DEFAULT_MAX_PARALLEL = 3
DEFAULT_FRAMES_PER_RUN = 750

def main():
    parser = argparse.ArgumentParser(description="Generate SPH simulation dataset")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of simulations to generate")
    parser.add_argument("--max_parallel", type=int, default=DEFAULT_MAX_PARALLEL)
    parser.add_argument("--frames_per_run", type=int, default=DEFAULT_FRAMES_PER_RUN)
    parser.add_argument("--output_dir", type=str, default=None, help="Dataset output directory (default: data/dataset_TIMESTAMP)")
    args = parser.parse_args()

    ld_path = "/scratch/s23adhik/acpp/lib/x86_64-unknown-linux-gnu"
    os.environ["LD_LIBRARY_PATH"] = f"{ld_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    dataset_name = args.output_dir or f"data/dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_dir = os.path.abspath(dataset_name)
    log_dir = os.path.abspath(os.path.join("logs", os.path.basename(data_dir)))

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.environ["SPH_DATA_ROOT"] = data_dir

    print(f"Generating {args.n} simulations -> {data_dir}")

    active_processes = []
    for r in range(1, args.n + 1):
        log_file = open(os.path.join(log_dir, f"sim_{r}.log"), "w")
        p = subprocess.Popen(["./spawn_random.sh", str(args.frames_per_run)],
                             stdout=log_file, stderr=subprocess.STDOUT)
        active_processes.append((p, log_file))

        while len([proc for proc, _ in active_processes if proc.poll() is None]) >= args.max_parallel:
            time.sleep(1)

        print(f"  Launched sim {r}/{args.n}", end="\r")

    print(f"\nWaiting for remaining simulations...")
    for proc, log_file in active_processes:
        proc.wait()
        log_file.close()

    print(f"Done. Dataset saved to: {data_dir}")

if __name__ == "__main__":
    main()
