import os
import subprocess
import time
import datetime
import sys
import shutil
import glob
import argparse
import random  # Memory-efficient random sampling

# --- Full Configuration Domain ---
DOMAIN_ITERATIONS = [5]
DOMAIN_RUNS_PER_ITERATION = [3, 10, 15, 25, 40, 45]
DOMAIN_MAX_PARALLEL = [2, 3, 4, 5, 6, 7, 8]
DOMAIN_MAX_SESSIONS = [10, 15, 20, 25, 30]
DOMAIN_FRAMES_PER_RUN = 750  # Kept as a constant default scalar
DOMAIN_MASS_LOSS_START_CYCLE = [2, 3, 4, 5]
DOMAIN_MASS_LOSS_WEIGHT = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
DOMAIN_FLUID_LOSS_WEIGHT = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
DOMAIN_NOISE_STD = [0.005, 0.01, 0.015, 0.02]
DOMAIN_AR_STEPS = [5, 6, 7, 8]
DOMAIN_AR_START_CYCLE = [2, 3, 4, 5, 6]
DOMAIN_AR_INCREMENT_INTERVAL = [2, 3, 4, 5, 6]
DOMAIN_SKIP_INITIAL = [1, 3, 5, 7, 9, 11]

def run_simulations(iteration, run_name, runs_per_iteration, max_parallel, frames_per_run, data_dir, log_dir):
    """Launches simulations in parallel with a throttling limit."""
    print(f"[{run_name}] Launching {runs_per_iteration} simulations ({max_parallel} at a time)...")
    
    active_processes = []
    
    for r in range(1, runs_per_iteration + 1):
        log_path = os.path.join(log_dir, f"sim_c{iteration}_r{r}.log")
        log_file = open(log_path, "w")
        
        # Start simulation process
        p = subprocess.Popen(["./spawn_random.sh", str(frames_per_run)], 
                             stdout=log_file, stderr=subprocess.STDOUT)
        active_processes.append((p, log_file))
        
        # Throttle: Wait if we reached max_parallel
        while len([proc for proc, _ in active_processes if proc.poll() is None]) >= max_parallel:
            time.sleep(1)
            
    # Wait for all remaining processes
    print(f"[{run_name}] Waiting for final simulations to complete...")
    for proc, log_file in active_processes:
        proc.wait()
        log_file.close()
    
    print(f"[{run_name}] All simulations for cycle {iteration} complete.")

def purge_experiment_data(data_dir):
    """Forcefully purges the entire data generation directory to save disk space."""
    if os.path.exists(data_dir):
        print(f"Purging generation directory to save space: {data_dir}")
        try:
            shutil.rmtree(data_dir)
            # Recreate the empty root directory shell for the next iterations if needed
            os.makedirs(data_dir, exist_ok=True)
        except Exception as e:
            print(f"Error purging data directory {data_dir}: {e}")

def get_random_configuration():
    """Lazily selects one random hyperparameter dictionary from the domain without generating the whole space."""
    return {
        "iterations": random.choice(DOMAIN_ITERATIONS),
        "runs_per_iteration": random.choice(DOMAIN_RUNS_PER_ITERATION),
        "max_parallel": random.choice(DOMAIN_MAX_PARALLEL),
        "max_sessions": random.choice(DOMAIN_MAX_SESSIONS),
        "mass_loss_start_cycle": random.choice(DOMAIN_MASS_LOSS_START_CYCLE),
        "mass_loss_weight": random.choice(DOMAIN_MASS_LOSS_WEIGHT),
        "fluid_weight": random.choice(DOMAIN_FLUID_LOSS_WEIGHT),
        "noise_std": random.choice(DOMAIN_NOISE_STD),
        "ar_steps": random.choice(DOMAIN_AR_STEPS),
        "ar_start_cycle": random.choice(DOMAIN_AR_START_CYCLE),
        "ar_increment_interval": random.choice(DOMAIN_AR_INCREMENT_INTERVAL),
        "skip_initial": random.choice(DOMAIN_SKIP_INITIAL)
    }

def run_active_learning_loop(args, config, run_base_name, exp_idx):
    """Runs a single active learning execution track for a sampled configuration."""
    
    # Extract config values
    iterations = config['iterations']
    runs_per_iteration = config['runs_per_iteration']
    max_parallel = config['max_parallel']
    mass_loss_start_cycle = config['mass_loss_start_cycle']
    mass_loss_weight = config['mass_loss_weight']
    fluid_weight = config['fluid_weight']
    noise_std = config['noise_std']
    ar_steps = config['ar_steps']
    ar_start_cycle = config['ar_start_cycle']
    ar_increment_interval = config['ar_increment_interval']
    skip_initial = config['skip_initial']

    # Unique isolated directory workspace paths
    grid_run_name = f"{run_base_name}_exp{exp_idx}"
    data_dir = os.path.abspath(os.path.join("data", grid_run_name))
    log_dir = os.path.abspath(os.path.join("logs", grid_run_name))
    attempts_dir = os.path.abspath(os.path.join("attempts", grid_run_name))
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(attempts_dir, exist_ok=True)
    
    # Write configuration details directly into the workspace folder for tracking logs
    config_track_path = os.path.join(attempts_dir, "experiment_config.txt")
    with open(config_track_path, "w") as f:
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        for key, val in config.items():
            f.write(f"{key}: {val}\n")

    os.environ["SPH_DATA_ROOT"] = data_dir
    
    print("\n" + "="*70)
    print(f"STARTING RANDOM EXPERIMENT {exp_idx}/1000: {grid_run_name}")
    print(f"Parameters Tracking Map Written to: {config_track_path}")
    print("="*70 + "\n")

    # Main Core Iteration Track
    for i in range(args.start_cycle, iterations + 1):
        print("-" * 50)
        print(f" Cycle {i} of {iterations} (Experiment Workspace: {grid_run_name})")
        print("-" * 50)

        if i < 5:
            current_epochs = 5
        elif i < 15:
            current_epochs = 10
        else:
            current_epochs = 20

        # Data Generation Track
        if not args.skip_sim:
            run_simulations(i, grid_run_name, runs_per_iteration, max_parallel, 
                            args.frames_per_run, data_dir, log_dir)
        else:
            print(f"Skipping simulation phase for cycle {i} as requested.")

        # Training Sequence Execution
        print(f"Launching compressor model optimization. Epoch count: {current_epochs}")
        
        train_cmd = [
            sys.executable, "compressor.py",
            "--cycle", str(i),
            "--epochs", str(current_epochs),
            "--data_dir", data_dir,
            "--output_dir", attempts_dir,
            "--model_name", "best_model.pth",
            "--mass_loss_weight", str(mass_loss_weight),
            "--mass_loss_start_cycle", str(mass_loss_start_cycle),
            "--fluid_weight", str(fluid_weight),
            "--batch_size", "0",
            "--effective_batch_size", "8",
            "--skip_frames", "5",
            "--n_steps", str(ar_steps),
            "--ar_start_cycle", str(ar_start_cycle),
            "--ar_increment_interval", str(ar_increment_interval),
            "--noise_std", str(noise_std),
            "--skip_initial", str(skip_initial),
            "--bf16"
        ]
        
        if args.use_8bit_adam:
            train_cmd.append("--use_8bit_adam")
            
        try:
            subprocess.run(train_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"WARNING: Training execution step failed at cycle {i} for {grid_run_name}: {e}")

        # --- Aggressive Post-Cycle Purge ---
        # Wipes all newly generated SPH physics files now that training for this cycle is done
        purge_experiment_data(data_dir)
        print(f"Cycle {i} tracking process and data cleanup complete.")

    # --- Final Experiment Lifecycle Clean up ---
    # Removes the empty experiment data folder completely once the full experiment run completes
    if os.path.exists(data_dir):
        try:
            shutil.rmtree(data_dir)
        except Exception:
            pass

    print(f"Experiment {grid_run_name} lifecycle finished execution tasks successfully.")

def main():
    parser = argparse.ArgumentParser(description="Random Search Active Learning Parameter Space Optimization Script")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--frames_per_run", type=int, default=DOMAIN_FRAMES_PER_RUN)
    parser.add_argument("--use_8bit_adam", action="store_true", default=False)
    parser.add_argument("--no_build", action="store_true", help="Skip building fluid sim binaries")
    parser.add_argument("--start_cycle", type=int, default=1)
    parser.add_argument("--skip_sim", action="store_true")
    parser.add_argument("--total_search_budget", type=int, default=1000, help="Total unique configurations to pick out of total search grid")
    args = parser.parse_args()

    # Environment Link Setup
    ld_path = "/scratch/s23adhik/acpp/lib/x86_64-unknown-linux-gnu"
    os.environ["LD_LIBRARY_PATH"] = f"{ld_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    run_base_name = args.run_name if args.run_name else f"randsearch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Build Verification Pipeline Execution Step
    if not args.no_build:
        print("Checking fluid simulation engine infrastructure builds...")
        sph_dir = os.path.abspath("../sph")
        if os.path.exists(sph_dir):
            try:
                subprocess.run(["make", "clean"], cwd=sph_dir, check=True)
                subprocess.run(["make", "-j"], cwd=sph_dir, check=True)
            except subprocess.CalledProcessError as e:
                print(f"CRITICAL: Structural build setup error encountered: {e}")
                sys.exit(1)
        else:
            print(f"Warning: SPH source directory not found at path {sph_dir}. Skipping automated step.")

    print(f"Initializing Lazy Random Configuration Pool. Target Space Budget: {args.total_search_budget}")
    
    seen_hashes = set()
    unique_configurations = []
    
    while len(unique_configurations) < args.total_search_budget:
        cfg = get_random_configuration()
        cfg_fingerprint = tuple(sorted(cfg.items()))
        
        if cfg_fingerprint not in seen_hashes:
            seen_hashes.add(cfg_fingerprint)
            unique_configurations.append(cfg)

    print(f"Successfully tracked {len(unique_configurations)} distinct experiment configs in memory pool.")

    # Execution loop engine
    for idx, configuration in enumerate(unique_configurations, start=1):
        run_active_learning_loop(args, configuration, run_base_name, idx)

    print("\n[Complete Success] All 1000 sampled configurations evaluated.")

if __name__ == "__main__":
    main()
