import os
import subprocess
import time
import datetime
import sys
import shutil
import glob
import argparse

# --- Configuration & Defaults ---
DEFAULT_ITERATIONS = 30
DEFAULT_RUNS_PER_ITERATION = 15
DEFAULT_MAX_PARALLEL = 3
DEFAULT_MAX_SESSIONS = 20
DEFAULT_FRAMES_PER_RUN = 750
DEFAULT_MASS_LOSS_START_CYCLE = 2
DEFAULT_MASS_LOSS_WEIGHT = 2.5
DEFAULT_FLUID_LOSS_WEIGHT = 35.0
DEFAULT_NOISE_STD = 0.01
DEFAULT_AR_STEPS = 5
DEFAULT_AR_START_CYCLE = 2
DEFAULT_AR_INCREMENT_INTERVAL = 3
DEFAULT_SKIP_INITIAL = 5
DEFAULT_USE_8BIT_ADAM = True

def run_simulations(iteration, run_name, runs_per_iteration, max_parallel, frames_per_run, data_dir, log_dir):
    """Launches simulations in parallel with a throttling limit."""
    print(f"Launching {runs_per_iteration} simulations ({max_parallel} at a time)...")
    
    active_processes = []
    
    for r in range(1, runs_per_iteration + 1):
        log_path = os.path.join(log_dir, f"sim_c{iteration}_r{r}.log")
        
        # Open log file
        log_file = open(log_path, "w")
        
        # Start simulation process
        # Note: We assume spawn_random.sh is in the current directory
        p = subprocess.Popen(["./spawn_random.sh", str(frames_per_run)], 
                             stdout=log_file, stderr=subprocess.STDOUT)
        active_processes.append((p, log_file))
        
        # Throttle: Wait if we reached max_parallel
        while len([proc for proc, _ in active_processes if proc.poll() is None]) >= max_parallel:
            time.sleep(1)
            
    # Wait for all remaining processes
    print("Waiting for final simulations to complete...")
    for proc, log_file in active_processes:
        proc.wait()
        log_file.close()
    
    print(f"All simulations for cycle {iteration} complete.")

def cleanup_storage(data_dir, max_sessions):
    """Keeps only the most recent N simulation folders."""
    print(f"Cleaning up old simulation data in {data_dir} (keeping top {max_sessions})...")
    
    # Get all subdirectories in data_dir
    subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Sort by modification time (newest first)
    subdirs.sort(key=os.path.getmtime, reverse=True)
    
    # Remove old ones
    if len(subdirs) > max_sessions:
        for old_dir in subdirs[max_sessions:]:
            try:
                shutil.rmtree(old_dir)
            except Exception as e:
                print(f"Error removing {old_dir}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Parallel Active Learning Loop for SPH Neural Compression")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--runs_per_iteration", type=int, default=DEFAULT_RUNS_PER_ITERATION)
    parser.add_argument("--max_parallel", type=int, default=DEFAULT_MAX_PARALLEL)
    parser.add_argument("--max_sessions", type=int, default=DEFAULT_MAX_SESSIONS)
    parser.add_argument("--frames_per_run", type=int, default=DEFAULT_FRAMES_PER_RUN)
    parser.add_argument("--mass_loss_weight", type=float, default=DEFAULT_MASS_LOSS_WEIGHT)
    parser.add_argument("--mass_loss_start_cycle", type=int, default=DEFAULT_MASS_LOSS_START_CYCLE)
    parser.add_argument("--fluid_weight", type=float, default=DEFAULT_FLUID_LOSS_WEIGHT)
    parser.add_argument("--noise_std", type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument("--ar_steps", type=int, default=DEFAULT_AR_STEPS)
    parser.add_argument("--ar_start_cycle", type=int, default=DEFAULT_AR_START_CYCLE)
    parser.add_argument("--ar_increment_interval", type=int, default=DEFAULT_AR_INCREMENT_INTERVAL)
    parser.add_argument("--skip_initial", type=int, default=DEFAULT_SKIP_INITIAL)
    parser.add_argument("--use_8bit_adam", action="store_true", default=DEFAULT_USE_8BIT_ADAM)
    parser.add_argument("--no_build", action="store_true", help="Skip building fluid sim")
    args = parser.parse_args()

    # --- Setup Environment ---
    ld_path = "/scratch/s23adhik/acpp/lib/x86_64-unknown-linux-gnu"
    os.environ["LD_LIBRARY_PATH"] = f"{ld_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    # --- Unique Run Setup ---
    run_name = args.run_name if args.run_name else f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_dir = os.path.abspath(os.path.join("data", run_name))
    log_dir = os.path.abspath(os.path.join("logs", run_name))
    attempts_dir = os.path.abspath(os.path.join("attempts", run_name))
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(attempts_dir, exist_ok=True)
    
    os.environ["SPH_DATA_ROOT"] = data_dir
    
    print(f"Setting up unique run: {run_name}")
    print(f"Data Dir: {data_dir}")

    # --- Build Step ---
    if not args.no_build:
        print("Checking fluid sim build...")
        sph_dir = os.path.abspath("../sph")
        if os.path.exists(sph_dir):
            try:
                subprocess.run(["make", "clean"], cwd=sph_dir, check=True)
                subprocess.run(["make", "-j"], cwd=sph_dir, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Build failed: {e}")
                sys.exit(1)
        else:
            print(f"Warning: SPH directory not found at {sph_dir}. Skipping build.")

    print(f"Starting Parallel Active Learning Loop for {run_name}...")

    # --- Main Active Learning Loop ---
    for i in range(1, args.iterations + 1):
        print("----------------------------------------")
        print(f" Cycle {i} of {args.iterations} (Run: {run_name})")
        print("----------------------------------------")

        # 0. Dynamic Epoch Calculation
        if i < 5:
            current_epochs = 5
        elif i < 15:
            current_epochs = 10
        else:
            current_epochs = 20

        # 1. Generate Data
        run_simulations(i, run_name, args.runs_per_iteration, args.max_parallel, 
                        args.frames_per_run, data_dir, log_dir)

        # 2. Storage Cleanup
        cleanup_storage(data_dir, args.max_sessions)

        # 3. Train on remaining data
        print(f"Starting training session with {current_epochs} epochs...")
        
        # Build the command for compressor.py
        # We run it as a separate process to ensure memory is fully cleared between cycles
        train_cmd = [
            sys.executable, "compressor.py",
            "--cycle", str(i),
            "--epochs", str(current_epochs),
            "--data_dir", data_dir,
            "--output_dir", attempts_dir,
            "--model_name", "best_model.pth",
            "--mass_loss_weight", str(args.mass_loss_weight),
            "--mass_loss_start_cycle", str(args.mass_loss_start_cycle),
            "--fluid_weight", str(args.fluid_weight),
            "--batch_size", "0",
            "--effective_batch_size", "8",
            "--skip_frames", "5",
            "--n_steps", str(args.ar_steps),
            "--ar_start_cycle", str(args.ar_start_cycle),
            "--ar_increment_interval", str(args.ar_increment_interval),
            "--noise_std", str(args.noise_std),
            "--skip_initial", str(args.skip_initial),
            "--bf16"
        ]
        
        if args.use_8bit_adam:
            train_cmd.append("--use_8bit_adam")
            
        try:
            subprocess.run(train_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Training failed at cycle {i}: {e}")
            # Optional: decide whether to exit or continue
            # sys.exit(1)

        print(f"Cycle {i} complete.")

    print(f"Active Training Loop Finished for {run_name}.")

if __name__ == "__main__":
    main()
