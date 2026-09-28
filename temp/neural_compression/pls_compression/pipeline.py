"""Two-phase pipeline entry point: generate simulation data, then train.

This is the Python implementation of what used to live in
``active_train_parallel.sh``. It keeps the same environment-variable defaults,
the same phase argument, the same command-line options, the same data budget
guard, and produces the same argument vector for
:func:`pls_compression.orchestration.main`.

Values are carried as strings from the environment all the way into the
argument vector so the text is preserved exactly; ``orchestration`` owns the
type conversion and validation. Only the numbers the budget guard needs are
parsed locally.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .orchestration import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EFFECTIVE_BATCH_SIZE,
    DEFAULT_FRAMES_PER_RUN,
    DEFAULT_MODEL_FILENAME,
    PACKAGE_ROOT,
    PHASE_ALL,
    PHASE_GENERATE,
    PHASE_TRAIN,
    PHASES,
    main as orchestration_main,
)

GIB = 1073741824
MIB = 1048576
DEFAULT_SIM_WIDTH = 400
DEFAULT_SIM_HEIGHT = 400

DESCRIPTION = """\
Generate simulation data in parallel, then train. Run the phases separately so a
long generation step does not have to be repeated in order to train again.
"""

EPILOG = """\
environment variables (command-line options win):
  VARIANT density | density_velocity            DATA_DIR session root
  OUTPUT_DIR checkpoint root                    FRAMES_PER_RUN frames per session
  RUNS_PER_CYCLE sessions per cycle             MAX_PARALLEL concurrent simulators
  CYCLES generate/train cycles                  EPOCHS epochs per cycle
  BATCH_SIZE samples per forward pass           EFFECTIVE_BATCH accumulation target
  SKIP_FRAMES frames between context and target LEARNING_RATE AdamW rate
  VALIDATION_FRACTION held-out session fraction NUM_WORKERS dataloader workers
  MODEL_FILENAME checkpoint name                MIN_DELTA min gain worth a write
  SAVE_EVERY write on epochs divisible by this  KEEP_LAST_CHECKPOINTS newest N to keep
  DEVICE auto | cuda | cpu                      MAX_SESSIONS prune threshold
  SIMULATION_SEED simulator seed                SEED training seed
  WIDTH/HEIGHT model shape                     LATENT_DIM latent width
  SIM_WIDTH/SIM_HEIGHT simulator resolution    MAX_BATCHES limit batches per epoch
  SMOKE set for a smoke run                     SPH_ROOT simulator checkout
  NO_RESUME set to restart each cycle           PRUNE set to trim old sessions

SIM_WIDTH and SIM_HEIGHT describe the resolution the simulator produces, which is
fixed by the C++ build, and are deliberately separate from the model WIDTH and
HEIGHT. The data budget is computed from the simulator resolution, not the model
shape, and a mismatch between the two is reported because session validation
will reject the data.
"""


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "") not in (None, "", "0", "false", "False", "no")


def _value_option(
    parser: argparse.ArgumentParser,
    *names: str,
    env: str,
    default: str,
    help_text: str,
    dest: str | None = None,
) -> None:
    """Add a string-valued option whose default comes from ``env``.

    The environment is read here rather than at the call site so no option can
    accidentally ignore its variable. Values stay strings so the text reaching
    ``orchestration`` is unchanged; argparse there converts and validates.
    ``dest`` defaults to the lowercased variable name and must be given when the
    attribute this module reads differs from that.
    """
    resolved = dest or env.lower()
    parser.add_argument(*names, dest=resolved, default=_env_str(env, default), help=help_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="active_train_parallel",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "phase",
        nargs="?",
        default=PHASE_ALL,
        choices=PHASES,
        help="generate data, train, or both (default: all)",
    )
    _value_option(parser, "--data-dir", "--data_dir", env="DATA_DIR", default=str(PACKAGE_ROOT / "data"), help_text="session root")
    _value_option(parser, "--output-dir", "--output_dir", env="OUTPUT_DIR", default=str(PACKAGE_ROOT / "attempts"), help_text="checkpoint root")
    _value_option(parser, "--variant", "--model-variant", "--model_variant", env="VARIANT", default="density", help_text="density or density_velocity")
    _value_option(parser, "--width", env="WIDTH", default="400", help_text="model width, must match the data")
    _value_option(parser, "--height", env="HEIGHT", default="400", help_text="model height, must match the data")
    _value_option(parser, "--latent-dim", "--latent_dim", env="LATENT_DIM", default="1024", help_text="latent width")
    _value_option(parser, "--device", env="DEVICE", default="auto", help_text="auto, cuda, or cpu")
    _value_option(parser, "--epochs", env="EPOCHS", default="1", help_text="epochs per cycle")
    _value_option(parser, "--cycles", env="CYCLES", default="1", help_text="generate and train cycles")
    _value_option(parser, "--runs-per-cycle", "--runs_per_cycle", env="RUNS_PER_CYCLE", default="1", help_text="sessions per cycle; use 2 or more so the train/validation split is real")
    _value_option(parser, "--max-parallel", "--max_parallel", env="MAX_PARALLEL", default="1", help_text="concurrent simulator processes")
    _value_option(parser, "--max-sessions", "--max_sessions", env="MAX_SESSIONS", default="0", help_text="prune threshold, 0 disables")
    _value_option(parser, "--frames-per-run", "--frames_per_run", env="FRAMES_PER_RUN", default=str(DEFAULT_FRAMES_PER_RUN), help_text="frames per generated session")
    _value_option(parser, "--batch-size", "--batch_size", env="BATCH_SIZE", default=str(DEFAULT_BATCH_SIZE), help_text="samples per forward pass")
    _value_option(parser, "--effective-batch-size", "--effective_batch_size", env="EFFECTIVE_BATCH", default=str(DEFAULT_EFFECTIVE_BATCH_SIZE), help_text="accumulation target, at least --batch-size", dest="effective_batch_size")
    _value_option(parser, "--skip-frames", "--skip_frames", env="SKIP_FRAMES", default="10", help_text="frames between context and target")
    _value_option(parser, "--learning-rate", "--learning_rate", env="LEARNING_RATE", default="5e-4", help_text="AdamW learning rate")
    _value_option(parser, "--validation-fraction", "--validation_fraction", env="VALIDATION_FRACTION", default="0.1", help_text="held-out session fraction")
    _value_option(parser, "--num-workers", "--num_workers", env="NUM_WORKERS", default="0", help_text="dataloader workers")
    _value_option(parser, "--model-filename", "--model_filename", env="MODEL_FILENAME", default=DEFAULT_MODEL_FILENAME, help_text="checkpoint name")
    _value_option(parser, "--min-delta", "--min_delta", env="MIN_DELTA", default="0", help_text="minimum validation-loss gain worth a checkpoint write")
    _value_option(parser, "--save-every", "--save_every", env="SAVE_EVERY", default="1", help_text="only write on epochs divisible by this")
    _value_option(parser, "--keep-last-checkpoints", "--keep_last_checkpoints", env="KEEP_LAST_CHECKPOINTS", default="0", help_text="retain only the newest N cycle checkpoints, 0 keeps all")
    _value_option(parser, "--simulation-seed", "--simulation_seed", env="SIMULATION_SEED", default="0", help_text="simulator seed")
    _value_option(parser, "--seed", env="SEED", default="0", help_text="training seed")
    _value_option(parser, "--sph-root", "--sph_root", env="SPH_ROOT", default=str(PACKAGE_ROOT.parent / "sph"), help_text="simulator checkout")
    _value_option(parser, "--max-batches", "--max_batches", env="MAX_BATCHES", default="", help_text="limit batches per epoch")
    _value_option(parser, "--sim-width", "--sim_width", env="SIM_WIDTH", default=str(DEFAULT_SIM_WIDTH), help_text="resolution the simulator produces")
    _value_option(parser, "--sim-height", "--sim_height", env="SIM_HEIGHT", default=str(DEFAULT_SIM_HEIGHT), help_text="resolution the simulator produces")
    parser.add_argument("--smoke", action="store_true", default=_env_flag("SMOKE"), help="smoke run")
    parser.add_argument("--prune", action="store_true", default=_env_flag("PRUNE"), help="prune old sessions")
    parser.add_argument("--no-resume", dest="no_resume", action="store_true", default=_env_flag("NO_RESUME"), help="restart from random init each cycle")
    return parser


@dataclass(frozen=True)
class DataBudget:
    """Projected data size for the requested generation, plus the filesystem it lands on."""

    frame_bytes: int
    session_bytes: int
    total_bytes: int
    free_bytes: int | None
    probe: str
    frames_per_run: int
    runs_per_cycle: int
    cycles: int

    @property
    def exceeds_free_space(self) -> bool:
        return self.free_bytes is not None and self.total_bytes > self.free_bytes

    @property
    def session_mib(self) -> int:
        return self.session_bytes // MIB

    @property
    def free_gib(self) -> int:
        return (self.free_bytes or 0) // GIB

    @property
    def needed_gib(self) -> int:
        return self.total_bytes // GIB + 1


def nearest_existing_directory(path: str | Path) -> Path:
    """Walk up until a directory exists, because the data root usually does not yet.

    Asking ``shutil.disk_usage`` about a missing path raises, which would silently
    skip the budget check.
    """
    candidate = Path(path).expanduser()
    while not candidate.is_dir():
        parent = candidate.parent
        if parent == candidate:
            return candidate
        candidate = parent
    return candidate


def compute_data_budget(
    data_dir: str | Path,
    frames_per_run: int,
    runs_per_cycle: int,
    cycles: int,
    sim_width: int,
    sim_height: int,
) -> DataBudget:
    probe = nearest_existing_directory(data_dir)
    frame_bytes = 4 * sim_width * sim_height * 4
    session_bytes = frames_per_run * frame_bytes
    try:
        free_bytes: int | None = shutil.disk_usage(probe).free
    except OSError:
        free_bytes = None
    return DataBudget(
        frame_bytes=frame_bytes,
        session_bytes=session_bytes,
        total_bytes=runs_per_cycle * cycles * session_bytes,
        free_bytes=free_bytes,
        probe=str(probe),
        frames_per_run=frames_per_run,
        runs_per_cycle=runs_per_cycle,
        cycles=cycles,
    )


def resolution_warning(width: int, height: int, sim_width: int, sim_height: int) -> str | None:
    if width == sim_width and height == sim_height:
        return None
    return (
        f"warning: model is {width}x{height} but the simulator produces {sim_width}x{sim_height}; "
        "training will reject the data\n"
        "set WIDTH/HEIGHT to match, or SIM_WIDTH/SIM_HEIGHT if the C++ build was compiled differently"
    )


def describe_budget(budget: DataBudget) -> str:
    return (
        f"data budget: {budget.cycles} cycles x {budget.runs_per_cycle} runs x {budget.frames_per_run} frames "
        f"= ~{budget.session_mib} MiB ({budget.free_gib} GiB free on {budget.probe})"
    )


def build_orchestration_argv(args: argparse.Namespace) -> list[str]:
    """Build the argument vector for ``orchestration.main`` in a fixed order."""
    argv = [
        "--phase", args.phase,
        "--data-dir", args.data_dir,
        "--output-dir", args.output_dir,
        "--variant", args.variant,
        "--width", args.width,
        "--height", args.height,
        "--latent-dim", args.latent_dim,
        "--device", args.device,
        "--epochs", args.epochs,
        "--cycles", args.cycles,
        "--runs-per-cycle", args.runs_per_cycle,
        "--max-parallel", args.max_parallel,
        "--max-sessions", args.max_sessions,
        "--frames-per-run", args.frames_per_run,
        "--batch-size", args.batch_size,
        "--effective-batch-size", args.effective_batch_size,
        "--skip-frames", args.skip_frames,
        "--learning-rate", args.learning_rate,
        "--validation-fraction", args.validation_fraction,
        "--num-workers", args.num_workers,
        "--model-filename", args.model_filename,
        "--min-delta", args.min_delta,
        "--save-every", args.save_every,
        "--keep-last-checkpoints", args.keep_last_checkpoints,
        "--simulation-seed", args.simulation_seed,
        "--seed", args.seed,
        "--sph-root", args.sph_root,
    ]
    if args.max_batches:
        argv += ["--max-batches", args.max_batches]
    if args.smoke:
        argv.append("--smoke")
    if args.prune:
        argv.append("--prune")
    if args.no_resume:
        argv.append("--no-resume")
    if args.phase == PHASE_TRAIN:
        argv.append("--skip-sim")
    return argv


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Only the numbers the guard and the warning need are parsed here; every
    # other value is passed through to orchestration for conversion.
    try:
        frames_per_run = int(args.frames_per_run)
        runs_per_cycle = int(args.runs_per_cycle)
        cycles = int(args.cycles)
        width = int(args.width)
        height = int(args.height)
        sim_width = int(args.sim_width)
        sim_height = int(args.sim_height)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.phase != PHASE_TRAIN:
        message = resolution_warning(width, height, sim_width, sim_height)
        if message is not None:
            print(message, file=sys.stderr)
    budget = compute_data_budget(
        args.data_dir, frames_per_run, runs_per_cycle, cycles, sim_width, sim_height
    )
    if budget.free_bytes is None:
        print(f"warning: could not determine free space for {budget.probe}; skipping the budget check", file=sys.stderr)
    elif budget.exceeds_free_space:
        print(
            f"refusing to generate: {budget.cycles} cycles x {budget.runs_per_cycle} runs x "
            f"{budget.frames_per_run} frames needs ~{budget.needed_gib} GiB but only ~{budget.free_gib} GiB "
            f"is free on {budget.probe}",
            file=sys.stderr,
        )
        print("lower FRAMES_PER_RUN, RUNS_PER_CYCLE, or CYCLES", file=sys.stderr)
        return 1
    else:
        print(describe_budget(budget))
    resume_state = "off" if args.no_resume else "on"
    print(
        f"phase={args.phase} variant={args.variant} data={args.data_dir} output={args.output_dir} "
        f"batch={args.batch_size} effective={args.effective_batch_size} skip_frames={args.skip_frames} "
        f"resume={resume_state}"
    )
    try:
        return orchestration_main(build_orchestration_argv(args))
    except (FileNotFoundError, OSError, ValueError, RuntimeError, TypeError, IndexError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
