#!/usr/bin/env python3
#SBATCH --job-name=sph_precompute
#SBATCH --output=logs/precompute_%j.log
#SBATCH --partition=gpu-gen
#SBATCH --nodelist=gpu-pt1-04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
"""
Precompute the whole SPH simulation dataset up front.

The active learning loop used to generate simulations inline, which serialised
GPU time (simulate -> train -> simulate -> train) and kept a rolling window of
only the most recent 20 sessions on disk. This script instead materialises the
entire pool once, so the loops can pick a subset per cycle and never generate
during training.

Design notes:

* Determinism - each simulation index derives its scenario from a
  `random.Random(seed + index)`, so the same `--seed` rebuilds the same dataset
  and a lost run can be regenerated without touching the rest.
* Identification - the sim binary names its output directory
  `<timestamp>_<rand4>` and appends `_task<N>` when `SLURM_ARRAY_TASK_ID` is
  set. Setting that variable to the sim index gives a deterministic way to map
  an index back to the directory it produced.
* Crash safety - `manifest.json` is rewritten after every completed sim, and
  `--resume` skips whatever is already recorded, so a killed job continues
  where it stopped.

Examples:
    # 450 sims (30 cycles x 15 runs), the default active_train.py plan
    python precompute_dataset.py --iterations 30 --runs_per_iteration 15

    # 1200 sims, 750 frames each, 4 concurrent sims per GPU
    python precompute_dataset.py --n 1200 --max_parallel 4

    # Check the disk bill without running anything
    python precompute_dataset.py --n 450 --dry_run
"""

import os
import sys
import json
import glob
import time
import random
import shutil
import argparse
import datetime
import subprocess
from typing import Dict, List, Optional, Tuple

from manifest_utils import FRAME_BYTES, sim_bytes
from active_train import generate_simulation_args, detect_gpus, gpu_env

DEFAULT_SEED = 1234
DEFAULT_FRAMES_PER_RUN = 750
DEFAULT_ITERATIONS = 30
DEFAULT_RUNS_PER_ITERATION = 15
DEFAULT_MAX_PARALLEL = 3
DEFAULT_RETRIES = 1

ACPP_LIB_PATH = "/scratch/s23adhik/acpp/lib/x86_64-unknown-linux-gnu"


def human(nbytes: float) -> str:
    """Format a byte count the way df would."""
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PiB"


def free_space(path: str) -> int:
    """Free bytes on the filesystem holding `path`, walking up to an existing parent."""
    probe = os.path.abspath(path)
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return 0


def find_session_dir(data_dir: str, index: int) -> Optional[str]:
    """
    Locate the output directory the sim binary created for `index`.

    `SLURM_ARRAY_TASK_ID` is set to the index, so the directory carries a
    `_task<index>` suffix. The timestamp prefix is second-resolution, so two
    sims launched in the same second would otherwise be indistinguishable; the
    suffix plus a newest-first sort resolves it.
    """
    matches = [d for d in glob.glob(os.path.join(data_dir, f"*_task{index}"))
               if os.path.isfile(os.path.join(d, "sim_data.bin"))]
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def inspect_session(path: str) -> Dict:
    """Frame count and byte size of a finished sim_data.bin."""
    size = os.path.getsize(os.path.join(path, "sim_data.bin"))
    return {"frames": size // FRAME_BYTES, "bytes": size}


def save_manifest(manifest_path: str, manifest: Dict) -> None:
    """Rewrite the manifest atomically so a kill mid-write cannot corrupt it."""
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, manifest_path)


def load_existing(manifest_path: str) -> Dict:
    if not os.path.isfile(manifest_path):
        return {}
    try:
        with open(manifest_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: ignoring unreadable manifest {manifest_path} ({e})")
        return {}


def build_sph(sph_dir: str) -> None:
    if not os.path.isdir(sph_dir):
        print(f"Error: SPH directory not found at {sph_dir}")
        sys.exit(1)
    print("Building SPH binary...")
    try:
        subprocess.run(["make", "clean"], cwd=sph_dir, check=True)
        subprocess.run(["make", "-j"], cwd=sph_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        sys.exit(1)


class Scheduler:
    """Round-robin sim launcher with a per-GPU concurrency cap and retries."""

    def __init__(self, data_dir: str, log_dir: str, sim_binary: str,
                 frames_per_run: int, seed: int, retries: int,
                 gpus: List[int], max_parallel: int, manifest: Dict, manifest_path: str):
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.sim_binary = sim_binary
        self.frames_per_run = frames_per_run
        self.seed = seed
        self.retries = retries
        self.gpus = gpus or [None]
        self.max_parallel = max_parallel
        self.manifest = manifest
        self.manifest_path = manifest_path
        self.entries: Dict[int, Dict] = {e["index"]: e for e in manifest["entries"]}
        self.active: List[Tuple[int, subprocess.Popen, object, Optional[int], int]] = []
        self.per_gpu: Dict[Optional[int], int] = {}
        self.done = 0
        self.total = 0

    def _flush(self) -> None:
        self.manifest["entries"] = sorted(self.entries.values(), key=lambda e: e["index"])
        self.manifest["completed"] = self.done
        save_manifest(self.manifest_path, self.manifest)

    def _discard_stale(self, index: int) -> None:
        """
        Remove output dirs left by earlier failed attempts at this index.

        Without this, a retry that crashes again would leave several `_task<i>`
        directories behind, and find_session_dir (newest-wins) could report a
        truncated file from a previous attempt as this one's result.
        """
        for path in glob.glob(os.path.join(self.data_dir, f"*_task{index}")):
            try:
                shutil.rmtree(path)
            except OSError as e:
                print(f"  warning: could not remove stale {path}: {e}")

    def launch(self, index: int, attempt: int) -> None:
        if attempt > 1:
            self._discard_stale(index)
        gpu = self.gpus[index % len(self.gpus)]
        sim_args = generate_simulation_args(random.Random(self.seed + index))

        log_path = os.path.join(self.log_dir, f"sim_{index:04d}.log")
        log_file = open(log_path, "w")
        log_file.write(f"# index={index} attempt={attempt} gpu={gpu} "
                       f"seed={self.seed + index}\n{' '.join(sim_args)}\n")
        log_file.flush()

        env = {
            **os.environ,
            "SPH_DATA_ROOT": self.data_dir,
            # Makes the output directory carry _task<index> for manifest mapping.
            "SLURM_ARRAY_TASK_ID": str(index),
        }
        pin = gpu_env(gpu)
        if pin:
            env.update(pin)

        cmd = [self.sim_binary, str(self.frames_per_run), "--headless"] + sim_args
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        self.active.append((index, proc, log_file, gpu, attempt))
        self.per_gpu[gpu] = self.per_gpu.get(gpu, 0) + 1

    def reap(self, attempts: Dict[int, int]) -> List[int]:
        """Harvest finished sims; return indices that failed and may be retried."""
        retry: List[int] = []
        for item in list(self.active):
            index, proc, log_file, gpu, attempt = item
            if proc.poll() is None:
                continue

            self.active.remove(item)
            self.per_gpu[gpu] -= 1
            log_file.close()
            # A retried index is already counted, so only count first completions.
            self.done += 1 if index not in self.entries else 0

            path = find_session_dir(self.data_dir, index)
            if proc.returncode == 0 and path:
                info = inspect_session(path)
                self.entries[index] = {
                    "index": index, "path": path, "status": "ok",
                    "frames": info["frames"], "bytes": info["bytes"],
                    "gpu": gpu, "attempts": attempt,
                }
                short = "" if info["frames"] >= self.frames_per_run else "  [SHORT RUN]"
                print(f"  [{self.done}/{self.total}] sim {index:4d} ok  "
                      f"{info['frames']:4d} frames  gpu={gpu}{short}", flush=True)
            else:
                self.entries[index] = {
                    "index": index, "path": path or "", "status": "failed",
                    "frames": 0, "bytes": 0, "gpu": gpu,
                    "attempts": attempt, "returncode": proc.returncode,
                }
                print(f"  [{self.done}/{self.total}] sim {index:4d} FAILED "
                      f"rc={proc.returncode} gpu={gpu} "
                      f"(log: {os.path.join(self.log_dir, f'sim_{index:04d}.log')})",
                      flush=True)
                if attempts[index] <= self.retries:
                    retry.append(index)

            self._flush()
        return retry

    def run(self, queue: List[int]) -> None:
        """Run `queue` to completion, requeueing failures while attempts remain."""
        attempts: Dict[int, int] = {i: 1 for i in queue}
        pending = list(queue)
        gpu_count = len([g for g in self.gpus if g is not None]) or 1
        # Retries re-enter the queue, so count distinct indices or the progress
        # denominator overshoots the job.
        self.total = len(set(queue)) + self.done

        print(f"Generating {len(pending)} simulations "
              f"({self.max_parallel} concurrent per GPU x {gpu_count} GPU(s) "
              f"= {self.max_parallel * gpu_count} at a time)...", flush=True)

        while pending or self.active:
            while pending:
                index = pending[0]
                gpu = self.gpus[index % len(self.gpus)]
                # Throttle this GPU's shard; reaping may requeue failures at the
                # end of `pending`, so drain retries last rather than starving.
                while self.per_gpu.get(gpu, 0) >= self.max_parallel:
                    time.sleep(0.5)
                    for failed in self.reap(attempts):
                        if failed not in pending:
                            pending.append(failed)
                            attempts[failed] += 1
                pending.pop(0)
                self.launch(index, attempts[index])

            if not self.active:
                break
            time.sleep(0.5)
            for failed in self.reap(attempts):
                if failed not in pending:
                    pending.append(failed)
                    attempts[failed] += 1

        manifest = self.manifest
        manifest["failed"] = sorted(i for i, e in self.entries.items()
                                    if e["status"] != "ok")
        self._flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute the full SPH simulation dataset for offline training")
    parser.add_argument("--n", type=int, default=None,
                        help="Total simulations (default: --iterations * --runs_per_iteration)")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                        help="Active learning cycles the dataset should cover")
    parser.add_argument("--runs_per_iteration", type=int, default=DEFAULT_RUNS_PER_ITERATION)
    parser.add_argument("--frames_per_run", type=int, default=DEFAULT_FRAMES_PER_RUN)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Dataset directory (default: data/dataset_TIMESTAMP)")
    parser.add_argument("--max_parallel", type=int, default=DEFAULT_MAX_PARALLEL,
                        help="Concurrent simulations PER GPU (total = this x GPU count)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU indices (default: auto-detect all)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Base seed; simulation i uses seed+i, so the dataset is reproducible")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                        help="Extra attempts per failed simulation")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip simulations already recorded as ok (default)")
    parser.add_argument("--no_resume", dest="resume", action="store_false",
                        help="Regenerate everything, ignoring the existing manifest")
    parser.add_argument("--no_build", action="store_true", help="Skip building the SPH binary")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the disk/time plan and exit without running anything")
    args = parser.parse_args()

    total = args.n if args.n is not None else args.iterations * args.runs_per_iteration
    if total <= 0:
        sys.exit("Error: dataset size must be positive")

    per_sim = sim_bytes(args.frames_per_run)
    total_bytes = per_sim * total

    print("=" * 72)
    print(f"SPH dataset precompute: {total} simulations x {args.frames_per_run} frames")
    print(f"  Per simulation : {human(per_sim)}")
    print(f"  Total dataset  : {human(total_bytes)}  ({total_bytes / 1e9:.1f} GB)")
    print(f"  Seed           : {args.seed}  (sim i uses seed {args.seed}+i)")
    print("=" * 72)
    if args.dry_run:
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_name = args.output_dir or \
        f"data/dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_dir = os.path.abspath(dataset_name)
    log_dir = os.path.abspath(os.path.join("logs", os.path.basename(data_dir)))
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    needed = total_bytes + (2 * 1024 ** 3)  # headroom for logs and partial writes
    free = free_space(os.path.dirname(data_dir))
    print(f"Dataset dir : {data_dir}")
    print(f"Log dir     : {log_dir}")
    print(f"Free space  : {human(free)}  (need ~{human(needed)})")
    if free and free < needed:
        print("WARNING: not enough free space; generation will fail partway through.")

    os.environ["LD_LIBRARY_PATH"] = f"{ACPP_LIB_PATH}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    # The sim binary reads this to decide where to write its output directory.
    os.environ["SPH_DATA_ROOT"] = data_dir

    sph_dir = os.path.abspath(os.path.join(base_dir, "..", "sph"))
    sim_binary = os.path.join(sph_dir, "draw2")
    if not args.no_build:
        build_sph(sph_dir)
    if not os.path.isfile(sim_binary):
        print(f"Error: simulation binary not found at {sim_binary}")
        sys.exit(1)

    if args.gpus is not None:
        gpus = [int(tok) for tok in args.gpus.split(",") if tok.strip()]
        source = "explicit --gpus"
    else:
        gpus = detect_gpus()
        source = "auto-detected"
    if gpus:
        print(f"Sharding across {len(gpus)} GPU(s) {gpus} ({source}), "
              f"{args.max_parallel} concurrent each.")
    else:
        print("No GPUs detected; running on the CPU path.")
    print()

    manifest_path = os.path.join(data_dir, "manifest.json")
    previous = load_existing(manifest_path) if args.resume else {}

    done: Dict[int, Dict] = {}
    for entry in previous.get("entries", []):
        if entry.get("status") == "ok" and \
                os.path.isfile(os.path.join(entry.get("path", ""), "sim_data.bin")):
            done[entry["index"]] = entry
    if done:
        print(f"Resuming: {len(done)}/{total} simulations already complete.")

    manifest = {
        "version": 1,
        "created": previous.get("created") or
                   datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "frames_per_run": args.frames_per_run,
        "total_requested": total,
        "frame_bytes": FRAME_BYTES,
        "data_dir": data_dir,
        "completed": len(done),
        "entries": sorted(done.values(), key=lambda e: e["index"]),
    }
    save_manifest(manifest_path, manifest)

    pending = [i for i in range(total) if i not in done]
    if not pending:
        print("Dataset already complete.")
        report(manifest, manifest_path, args)
        return

    scheduler = Scheduler(data_dir, log_dir, sim_binary, args.frames_per_run,
                          args.seed, args.retries, gpus, args.max_parallel,
                          manifest, manifest_path)
    scheduler.done = len(done)
    started = time.time()
    scheduler.run(pending)
    elapsed = time.time() - started

    print()
    print(f"Wall time: {elapsed / 3600:.2f} h")
    report(manifest, manifest_path, args)


def report(manifest: Dict, manifest_path: str, args) -> None:
    ok = [e for e in manifest["entries"] if e["status"] == "ok"]
    bad = [e for e in manifest["entries"] if e["status"] != "ok"]
    size = sum(e["bytes"] for e in ok)
    print("=" * 72)
    print(f"Complete: {len(ok)}/{args.n or args.iterations * args.runs_per_iteration} simulations")
    print(f"Dataset size: {human(size)}  ({size / 1e9:.1f} GB)")
    if bad:
        print(f"Failed ({len(bad)}): {[e['index'] for e in bad]}")
        print("Re-run the same command with --resume to retry just these.")
    print(f"Manifest: {manifest_path}")
    print("=" * 72)
    print()
    print("Train against it with:")
    print(f"  python active_train.py --manifest {manifest_path} "
          f"--iterations {args.iterations}")


if __name__ == "__main__":
    main()
