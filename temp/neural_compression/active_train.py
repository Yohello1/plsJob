#!/usr/bin/env python3
#SBATCH --job-name=cuda_test
#SBATCH --output=logs/cuda_test_%j.log
#SBATCH --partition=gpu-gen
#SBATCH --nodelist=gpu-pt1-04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
"""
Unified Active Learning Loop for SPH Neural Compression.

Python port of active_train_parallel.sh and spawn_random.sh combined into a
single script: scenario generation, GPU-sharded parallel simulation, rolling
storage cleanup, and the per-cycle training invocation all live here.

Simulations are sharded round-robin across every GPU on the system. Each child
process is pinned with CUDA_VISIBLE_DEVICES, since draw2 selects its SYCL
device via `default_selector{}` and exposes no device-index flag. --max_parallel
is therefore a per-GPU budget, giving max_parallel * num_gpus concurrent sims.
"""

import os
import sys
import time
import random
import shutil
import datetime
import argparse
import subprocess
from pathlib import Path
from typing import IO, Dict, List, Optional, Tuple


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
DEFAULT_USE_BF16 = True

# --- Scenario Constants (400x400 Resolution) ---
MIN_X = 50
MAX_X = 350
MIN_Y = 50
MAX_Y = 370

Rect = Tuple[int, int, int, int]


def bash_rand(mod: int, base: int = 0, rng: Optional[random.Random] = None) -> int:
    """
    Mirror the shell script's `$((RANDOM % mod + base))` idiom.

    That expression yields an integer in [base, base + mod - 1], so this is the
    exact equivalent. (Bash's modulo is very slightly non-uniform since RANDOM
    is 15-bit; uniform sampling here is the same distribution for practical
    purposes and is what the original was approximating.)

    `rng` makes the draw reproducible: the precompute script seeds one RNG per
    simulation index so a dataset can be regenerated exactly.
    """
    return rng.randrange(base, base + mod) if rng is not None else random.randrange(base, base + mod)


def check_intersect(x1: int, y1: int, w1: int, h1: int,
                   x2: int, y2: int, w2: int, h2: int) -> bool:
    """Return True if two axis-aligned rectangles overlap."""
    return (x1 < x2 + w2) and (x1 + w1 > x2) and (y1 < y2 + h2) and (y1 + h1 > y2)


def build_scenario_ghosts(scenario: int, rng: Optional[random.Random] = None) -> List[Rect]:
    """Build the static geometry for a scenario (x, y, w, h) tuples."""
    ghosts: List[Rect] = []

    if scenario == 0:
        # Single Fluid + Floor
        ghosts.append((bash_rand(150, 100, rng), 340, bash_rand(150, 100, rng), 40))

    elif scenario == 1:
        # Multiple Drop Scenario
        ghosts.append((MIN_X, 360, MAX_X - MIN_X, 20))

    elif scenario == 2:
        # Floating Obstacle Course
        for _ in range(5):
            ghosts.append((bash_rand(200, 50, rng), bash_rand(200, 100, rng),
                           bash_rand(60, 30, rng), bash_rand(40, 20, rng)))

    elif scenario == 3:
        # Complex Multi-level
        for _ in range(4):
            ghosts.append((bash_rand(150, 50, rng), MAX_Y - 50,
                           bash_rand(100, 50, rng), bash_rand(40, 10, rng)))

    elif scenario == 4:
        # Stair Steps
        for i in range(7):
            ghosts.append((60 + i * 40, 150 + i * 30, 80, 20))

    elif scenario == 5:
        # Central Bowl
        ghosts.append((100, 300, 200, 30))  # Bottom
        ghosts.append((100, 200, 30, 100))   # Left
        ghosts.append((270, 200, 30, 100))  # Right

    elif scenario == 6:
        # Plinko / Peg Board (forces fluid to split)
        for r in range(4):
            for c in range(6):
                offset = (r % 2) * 25
                ghosts.append((70 + c * 50 + offset, 150 + r * 50, 15, 15))

    elif scenario == 7:
        # Hourglass / Funnel (high pressure jet)
        ghosts.append((50, 200, 120, 20))   # Left slope
        ghosts.append((230, 200, 120, 20))  # Right slope
        ghosts.append((50, 220, 20, 100))   # Left wall
        ghosts.append((330, 220, 20, 100))  # Right wall

    elif scenario == 8:
        # Random Pillars
        for _ in range(8):
            ghosts.append((bash_rand(250, 70, rng), bash_rand(200, 100, rng),
                           15, bash_rand(60, 20, rng)))

    elif scenario == 9:
        # Two Dam Breaks (colliding fluid)
        ghosts.append((MIN_X, 360, MAX_X - MIN_X, 20))  # Floor

    return ghosts


def build_scenario_fluids(scenario: int, ghost_boxes: List[Rect],
                          rng: Optional[random.Random] = None) -> List[Rect]:
    """Place fluid regions that do not intersect any ghost geometry."""
    fluid_boxes: List[Rect] = []

    if scenario == 9:
        # Dam break walls are predefined, no drops needed
        return [(50, 50, 60, 200), (290, 50, 60, 200)]

    num_drop_targets = 5 if scenario == 1 else bash_rand(3, 1, rng)

    for _ in range(num_drop_targets):
        for _attempt in range(20):
            if (rng.randrange(10) if rng is not None else random.randrange(10)) < 3:
                # 30% chance of spawning a blobby cluster of 3 blobs
                cx = bash_rand(MAX_X - 100 - MIN_X, MIN_X, rng)
                cy = bash_rand(200 - MIN_Y, MIN_Y, rng)
                temp_blobs: List[Rect] = []
                collision = False

                for _blob in range(3):
                    fw = bash_rand(40, 30, rng)
                    fh = bash_rand(40, 30, rng)
                    fx = cx + bash_rand(30, 0, rng)
                    fy = cy + bash_rand(30, 0, rng)
                    if any(check_intersect(fx, fy, fw, fh, *g) for g in ghost_boxes):
                        collision = True
                        break
                    temp_blobs.append((fx, fy, fw, fh))

                if not collision:
                    fluid_boxes.extend(temp_blobs)
                    break
            else:
                # Normal rectangle drop
                fw = bash_rand(80, 40, rng)
                fh = bash_rand(80, 40, rng)
                fx = bash_rand(MAX_X - fw - MIN_X, MIN_X, rng)
                fy = bash_rand(250 - fh - MIN_Y, MIN_Y, rng)
                if not any(check_intersect(fx, fy, fw, fh, *g) for g in ghost_boxes):
                    fluid_boxes.append((fx, fy, fw, fh))
                    break

    return fluid_boxes


def generate_simulation_args(rng: Optional[random.Random] = None) -> List[str]:
    """
    Generate a randomized scenario and flatten it into draw2 CLI arguments.
    Ported from spawn_random.sh, which seeded RANDOM from nanoseconds + PID.

    Passing `rng` makes the scenario reproducible, which is what lets
    precompute_dataset.py rebuild an identical dataset from a base seed.
    """
    scenario = rng.randrange(10) if rng is not None else random.randrange(10)
    ghost_boxes = build_scenario_ghosts(scenario, rng)
    fluid_boxes = build_scenario_fluids(scenario, ghost_boxes, rng)

    args: List[str] = []
    for fx, fy, fw, fh in fluid_boxes:
        args.extend(["--fluid", f"{fx} {fy} {fw} {fh}"])
    for gx, gy, gw, gh in ghost_boxes:
        args.extend(["--ghost", f"{gx} {gy} {gw} {gh}"])
    return args


def detect_gpus() -> List[int]:
    """
    Return the indices of every GPU visible to the system.

    draw2 is an AdaptiveCpp (hipSYCL) program whose queue is built from
    `default_selector{}`, so it will happily bind to GPU 0 on its own. Sharding
    therefore happens by restricting each child process to a single device
    rather than by passing a device index to the binary (it has no such flag).
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []

    gpus: List[int] = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            gpus.append(int(line))
    return gpus


def gpu_env(gpu: Optional[int]) -> Optional[Dict[str, str]]:
    """
    Build a child environment pinned to one GPU, or None to leave it untouched.

    CUDA_VISIBLE_DEVICES is the authoritative lever: it constrains the driver so
    the process only ever sees the assigned device, and it cannot silently fall
    back the way a SYCL selector string does. The device is then renumbered to
    0 inside the process, hence `ONEAPI_DEVICE_SELECTOR=cuda:0` rather than the
    physical index.
    """
    if gpu is None:
        return None
    return {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "ONEAPI_DEVICE_SELECTOR": "cuda:0",
        "SYCL_DEVICE_FILTER": "*:0",
    }


def run_simulations(iteration: int, runs_per_iteration: int, max_parallel: int,
                    frames_per_run: int, log_dir: str, sim_binary: Path,
                    gpus: List[int]) -> None:
    """
    Launch simulations sharded round-robin across every GPU.

    `max_parallel` is a per-GPU limit, so total concurrency is
    max_parallel * len(gpus).
    """
    shards = gpus or [None]
    if gpus:
        print(f"Launching {runs_per_iteration} simulations across "
              f"{len(gpus)} GPU(s) ({gpus}), {max_parallel} per GPU "
              f"= {max_parallel * len(gpus)} concurrent)...")
    else:
        print(f"Launching {runs_per_iteration} simulations ({max_parallel} at a time)...")

    active: List[Tuple[subprocess.Popen, IO, Optional[int]]] = []
    active_per_gpu: Dict[Optional[int], int] = {}

    def reap() -> None:
        """Close finished simulations and free their GPU slots."""
        for entry in list(active):
            proc, log_file, gpu = entry
            if proc.poll() is not None:
                log_file.close()
                active.remove(entry)
                active_per_gpu[gpu] -= 1

    for r in range(1, runs_per_iteration + 1):
        gpu = shards[(r - 1) % len(shards)]

        # Throttle per shard: wait until this GPU has a free slot
        while active_per_gpu.get(gpu, 0) >= max_parallel:
            time.sleep(0.5)
            reap()

        log_path = os.path.join(log_dir, f"sim_c{iteration}_r{r}.log")
        log_file = open(log_path, "w")
        log_file.write(f"# assigned_gpu: {gpu if gpu is not None else 'n/a'}\n")
        log_file.flush()

        cmd = [str(sim_binary), str(frames_per_run), "--headless"] + generate_simulation_args()
        child_env = gpu_env(gpu)
        if child_env is None:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        else:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                    env={**os.environ, **child_env})
        active.append((proc, log_file, gpu))
        active_per_gpu[gpu] = active_per_gpu.get(gpu, 0) + 1

    print("Waiting for final simulations to complete...")
    for proc, log_file, _gpu in active:
        proc.wait()
        log_file.close()

    print(f"All simulations for cycle {iteration} complete.")


def cleanup_storage(data_dir: str, max_sessions: int) -> None:
    """Keeps only the most recent N simulation folders."""
    print(f"Cleaning up old simulation data in {data_dir} (keeping top {max_sessions})...")

    subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))]

    # Newest first, matching `ls -dt` in the shell version
    subdirs.sort(key=os.path.getmtime, reverse=True)

    for old_dir in subdirs[max_sessions:]:
        try:
            shutil.rmtree(old_dir)
        except OSError as e:
            print(f"Error removing {old_dir}: {e}")


def build_fluid_sim(sph_dir: Path) -> None:
    """Rebuild the SPH simulation binary."""
    print("Checking fluid sim build...")
    if not sph_dir.exists():
        print(f"Warning: SPH directory not found at {sph_dir}. Skipping build.")
        return
    try:
        subprocess.run(["make", "clean"], cwd=sph_dir, check=True)
        subprocess.run(["make", "-j"], cwd=sph_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        sys.exit(1)


def compute_epochs(cycle: int) -> int:
    """Dynamic epoch ramp: fast adaptation early, fine-tuning later."""
    if cycle < 5:
        return 5
    if cycle < 15:
        return 10
    return 20


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel Active Learning Loop for SPH Neural Compression")
    parser.add_argument("run_name", nargs="?", default=None,
                        help="Run name (default: run_<timestamp>)")
    parser.add_argument("--run_name", dest="run_name_flag", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--runs_per_iteration", type=int, default=DEFAULT_RUNS_PER_ITERATION)
    parser.add_argument("--max_parallel", type=int, default=DEFAULT_MAX_PARALLEL,
                        help="Concurrent simulations PER GPU (total = this x GPU count)")
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
    parser.add_argument("--use_8bit_adam", dest="use_8bit_adam", action="store_true",
                        default=DEFAULT_USE_8BIT_ADAM,
                        help="Use BitsAndBytes 8-bit AdamW optimizer")
    parser.add_argument("--no_8bit_adam", dest="use_8bit_adam", action="store_false",
                        help="Disable BitsAndBytes 8-bit AdamW optimizer")
    parser.add_argument("--bf16", dest="bf16", action="store_true", default=DEFAULT_USE_BF16,
                        help="Train with bfloat16 precision")
    parser.add_argument("--no_bf16", dest="bf16", action="store_false",
                        help="Disable bfloat16 precision")
    parser.add_argument("--no_build", action="store_true", help="Skip building fluid sim")
    parser.add_argument("--start_cycle", type=int, default=1,
                        help="Cycle to start from (useful for resuming)")
    parser.add_argument("--skip_sim", action="store_true",
                        help="Skip the simulation data generation phase")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU indices to shard across "
                             "(default: auto-detect every GPU on the system)")
    args = parser.parse_args()

    run_name = args.run_name_flag or args.run_name

    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)

    # Seed randomness to mirror the shell script's nanoseconds + PID seed
    random.seed((time.time_ns() // 10000 + os.getpid()) % 32768)

    # --- Environment Setup ---
    ld_path = "/scratch/s23adhik/acpp/lib/x86_64-unknown-linux-gnu"
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if current_ld:
        os.environ["LD_LIBRARY_PATH"] = f"{ld_path}:{current_ld}"
    else:
        os.environ["LD_LIBRARY_PATH"] = ld_path

    # --- Unique Run Setup ---
    if not run_name:
        run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    data_dir = os.path.abspath(os.path.join("data", run_name))
    log_dir = os.path.abspath(os.path.join("logs", run_name))
    attempts_dir = os.path.abspath(os.path.join("attempts", run_name))

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(attempts_dir, exist_ok=True)

    # Exported for the simulation binary (src/logging.cpp)
    os.environ["SPH_DATA_ROOT"] = data_dir

    print(f"Setting up unique run: {run_name}")
    print(f"Data Dir: {data_dir}")

    sph_dir = base_dir.parent / "sph"
    sim_binary = sph_dir / "draw2"

    if not args.no_build:
        build_fluid_sim(sph_dir)

    if not sim_binary.exists():
        print(f"Error: simulation binary not found at {sim_binary}")
        sys.exit(1)

    # --- GPU Shard Discovery ---
    if args.gpus is not None:
        gpus = [int(tok) for tok in args.gpus.split(",") if tok.strip() != ""]
        gpu_source = "explicit --gpus"
    else:
        gpus = detect_gpus()
        gpu_source = "auto-detected"

    if gpus:
        print(f"Data generation sharded across {len(gpus)} GPU(s) [{gpus}] ({gpu_source}).")
    else:
        print("No GPUs detected; running data generation unpinned on the CPU path.")

    # Prefer the project virtualenv, matching `./env/bin/python` from the shell script
    venv_python = base_dir / "env" / "bin" / "python"
    python_bin = str(venv_python) if venv_python.exists() else sys.executable

    print(f"Starting Parallel Active Learning Loop for {run_name}...")

    # --- Main Active Learning Loop ---
    for i in range(args.start_cycle, args.iterations + 1):
        print("----------------------------------------")
        print(f" Cycle {i} of {args.iterations} (Run: {run_name})")
        print("----------------------------------------")

        # 0. Dynamic Epoch Calculation
        current_epochs = compute_epochs(i)

        # 1. Generate Data
        if not args.skip_sim:
            run_simulations(i, args.runs_per_iteration, args.max_parallel,
                            args.frames_per_run, log_dir, sim_binary, gpus)
        else:
            print(f"Skipping simulation phase for cycle {i} as requested.")

        # 2. Storage Cleanup
        cleanup_storage(data_dir, args.max_sessions)

        # 3. Train on remaining data
        print(f"Starting training session with {current_epochs} epochs...")

        # Run as a separate process so memory is fully released between cycles
        train_cmd = [
            python_bin, "compressor.py",
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
        ]

        if args.use_8bit_adam:
            train_cmd.append("--use_8bit_adam")
        if args.bf16:
            train_cmd.append("--bf16")

        try:
            subprocess.run(train_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Training failed at cycle {i}: {e}")

        print(f"Cycle {i} complete.")

    print(f"Active Training Loop Finished for {run_name}.")


if __name__ == "__main__":
    main()
