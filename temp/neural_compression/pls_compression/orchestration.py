from __future__ import annotations

import argparse
import copy
import math
import os
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from .dataset import discover_sessions
from .schema import (
    DEFAULT_MODEL_VARIANT,
    HEIGHT,
    WIDTH,
    ModelConfig,
    canonical_model_variant,
    model_variant_label,
)
from .training import TrainingConfig, TrainingResult, train_model


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DEFAULT_SPH_ROOT = PACKAGE_ROOT.parent / "sph"
SPH_ROOT = DEFAULT_SPH_ROOT
DEFAULT_FRAMES_PER_RUN = 100
DEFAULT_BATCH_SIZE = 8
DEFAULT_EFFECTIVE_BATCH_SIZE = 32
DEFAULT_MODEL_FILENAME = "best_model.pth"
PHASE_GENERATE = "generate"
PHASE_TRAIN = "train"
PHASE_ALL = "all"
PHASES = (PHASE_ALL, PHASE_GENERATE, PHASE_TRAIN)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result != value and not (isinstance(value, str) and str(result) == value.strip()):
        raise ValueError(f"{name} must be a positive integer")
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


@dataclass
class ActiveLearningConfig:
    data_dir: str | Path
    output_dir: str | Path
    cycles: int = 1
    epochs: int = 1
    runs_per_cycle: int = 0
    max_parallel: int = 1
    max_sessions: int = 0
    simulation_command: str | Sequence[str] | None = None
    simulation_args: Sequence[str] = ()
    model_variant: str = DEFAULT_MODEL_VARIANT
    training_config: TrainingConfig | None = None
    seed: int = 0
    prune: bool = False
    frames_per_run: int = DEFAULT_FRAMES_PER_RUN
    simulation_seed: int = 0
    sph_root: str | Path | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    effective_batch_size: int | None = DEFAULT_EFFECTIVE_BATCH_SIZE
    skip_frames: int = 10
    n_steps: int = 1
    skip_initial: int = 1
    learning_rate: float = 5e-4
    validation_fraction: float = 0.1
    gradient_clip_norm: float | None = None
    noise_std: float = 0.0
    num_workers: int = 0
    model_filename: str = DEFAULT_MODEL_FILENAME
    resume: bool = True
    phase: str = PHASE_ALL

    def __post_init__(self) -> None:
        try:
            self.cycles = int(self.cycles)
            self.epochs = int(self.epochs)
            self.runs_per_cycle = int(self.runs_per_cycle)
            self.max_parallel = int(self.max_parallel)
            self.max_sessions = int(self.max_sessions)
            self.frames_per_run = int(self.frames_per_run)
        except (TypeError, ValueError) as exc:
            raise ValueError("active-learning limits must be integers") from exc
        if self.cycles <= 0 or self.epochs <= 0:
            raise ValueError("cycles and epochs must be positive")
        if self.runs_per_cycle < 0 or self.max_parallel <= 0 or self.max_sessions < 0:
            raise ValueError("invalid active-learning limits")
        if self.frames_per_run <= 0:
            raise ValueError("frames_per_run must be positive")
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {self.phase!r}")
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.model_variant = canonical_model_variant(self.model_variant)
        self.seed = int(self.seed)
        self.simulation_seed = int(self.simulation_seed)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.simulation_seed < 0:
            raise ValueError("simulation_seed must be non-negative")
        self.batch_size = _positive_int(self.batch_size, "batch_size")
        if self.effective_batch_size is not None:
            self.effective_batch_size = _positive_int(self.effective_batch_size, "effective_batch_size")
            if self.effective_batch_size < self.batch_size:
                raise ValueError("effective_batch_size must be at least batch_size")
        self.skip_frames = _positive_int(self.skip_frames, "skip_frames")
        self.n_steps = _positive_int(self.n_steps, "n_steps")
        self.skip_initial = _positive_int(self.skip_initial, "skip_initial")
        if int(self.num_workers) != self.num_workers or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        self.num_workers = int(self.num_workers)
        self.learning_rate = float(self.learning_rate)
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        self.validation_fraction = float(self.validation_fraction)
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        self.noise_std = float(self.noise_std)
        if not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be non-negative and finite")
        if self.gradient_clip_norm is not None:
            self.gradient_clip_norm = float(self.gradient_clip_norm)
            if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
                raise ValueError("gradient_clip_norm must be positive and finite")
        self.model_filename = str(self.model_filename)
        if not self.model_filename:
            raise ValueError("model_filename must not be empty")
        self.resume = bool(self.resume)
        if self.simulation_args is None:
            self.simulation_args = ()
        elif isinstance(self.simulation_args, str):
            self.simulation_args = tuple(shlex.split(self.simulation_args))
        else:
            self.simulation_args = tuple(str(value) for value in self.simulation_args)

    @property
    def runs_simulation(self) -> bool:
        return self.phase in (PHASE_ALL, PHASE_GENERATE)

    @property
    def trains(self) -> bool:
        return self.phase in (PHASE_ALL, PHASE_TRAIN)

    @property
    def simulation_variant(self) -> str:
        return self.model_variant

    @property
    def simulation_target(self) -> str:
        return simulation_build_target(self.model_variant)


def resolve_sph_root(value: str | Path | None = None) -> Path:
    configured = value or os.environ.get("SPH_ROOT") or os.environ.get("SPH_DIR") or os.environ.get("SPH_SIM_ROOT") or DEFAULT_SPH_ROOT
    return Path(configured).expanduser().resolve()


def simulation_build_target(model_variant: str = DEFAULT_MODEL_VARIANT) -> str:
    variant = canonical_model_variant(model_variant)
    return "draw2-density-only" if variant == DEFAULT_MODEL_VARIANT else "draw2-density-velocity"


def simulation_binary(model_variant: str = DEFAULT_MODEL_VARIANT, sph_root: str | Path | None = None) -> Path:
    return resolve_sph_root(sph_root) / simulation_build_target(model_variant)


def default_simulation_command(model_variant: str = DEFAULT_MODEL_VARIANT) -> list[str]:
    variant = canonical_model_variant(model_variant)
    script = PACKAGE_ROOT / "spawn_random.sh"
    if not script.is_file():
        raise FileNotFoundError(f"simulation script not found: {script}")
    return [str(script), "--variant", model_variant_label(variant), "--build-target", simulation_build_target(variant)]


def _simulation_command(config: ActiveLearningConfig) -> list[str]:
    if config.simulation_command is None:
        return default_simulation_command(config.model_variant)
    if isinstance(config.simulation_command, (str, Path)):
        return [str(config.simulation_command)]
    return [str(value) for value in config.simulation_command]


def _uses_spawn_script(command: Sequence[str]) -> bool:
    if not command:
        return False
    try:
        return Path(command[0]).name == "spawn_random.sh"
    except (TypeError, ValueError):
        return False


def _option_value(values: Sequence[str], option: str) -> str | None:
    for index, value in enumerate(values[:-1]):
        if value == option:
            return values[index + 1]
        if value.startswith(f"{option}="):
            return value.split("=", 1)[1]
    return None


def _has_legacy_frame_count(values: Sequence[str]) -> bool:
    index = 1
    while index < len(values):
        value = values[index]
        if value in {"--fluid", "-f", "--ghost", "-g"}:
            index += 5
            continue
        if value in {"--variant", "--model-variant", "--model_variant", "--mode", "--simulation-mode", "--simulation_mode", "--build-target", "--target", "--frames", "--seed", "--scenario-seed"}:
            index += 2
            continue
        if value == "--headless":
            index += 1
            continue
        if value == "--":
            index += 1
            continue
        if value.isdigit() and int(value) > 0:
            return True
        index += 1
    return False


def simulation_arguments(config: ActiveLearningConfig, cycle: int, run_index: int) -> list[str]:
    command = _simulation_command(config)
    values = list(config.simulation_args)
    if _uses_spawn_script(command):
        seed = config.simulation_seed + cycle * 1000003 + (run_index - 1) * 1009
        combined = command + values
        if _option_value(combined, "--variant") != model_variant_label(config.model_variant):
            values.extend(["--variant", model_variant_label(config.model_variant)])
        if _option_value(combined, "--build-target") != simulation_build_target(config.model_variant):
            values.extend(["--build-target", simulation_build_target(config.model_variant)])
        if "--frames" not in combined and _option_value(combined, "--frames") is None and not _has_legacy_frame_count(combined):
            values.extend(["--frames", str(config.frames_per_run)])
        if "--scenario-seed" not in combined and _option_value(combined, "--seed") is None:
            values.extend(["--seed", str(seed)])
    return values


def run_simulations(config: ActiveLearningConfig, cycle: int = 1) -> None:
    command = _simulation_command(config)
    if config.runs_per_cycle == 0 and config.simulation_command is None:
        return
    config.data_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["SPH_DATA_ROOT"] = str(config.data_dir)
    environment["SPH_MODEL_VARIANT"] = model_variant_label(config.model_variant)
    environment["SPH_MODEL_VARIANT_CANONICAL"] = config.model_variant
    environment["SPH_VARIANT"] = model_variant_label(config.model_variant)
    environment["SPH_BUILD_TARGET"] = simulation_build_target(config.model_variant)
    if config.sph_root is not None or (not environment.get("SPH_BINARY") and not environment.get("SPH_DRAW2")):
        environment["SPH_BINARY"] = str(simulation_binary(config.model_variant, config.sph_root))
    environment["SPH_SCENARIO_SEED"] = str(config.simulation_seed + cycle * 1000003)
    if config.sph_root is not None:
        environment["SPH_ROOT"] = str(resolve_sph_root(config.sph_root))

    def launch(index: int) -> None:
        values = command + simulation_arguments(config, cycle, index)
        launch_environment = environment.copy()
        launch_environment["SPH_SCENARIO_SEED"] = str(config.simulation_seed + cycle * 1000003 + (index - 1) * 1009)
        launch_environment["SPH_FRAMES_PER_RUN"] = str(config.frames_per_run)
        launch_environment["SPH_CYCLE"] = str(cycle)
        launch_environment["SPH_RUN_INDEX"] = str(index)
        subprocess.run(values, cwd=PACKAGE_ROOT, env=launch_environment, check=True)

    if config.runs_per_cycle == 0:
        launch(1)
        return
    workers = min(config.max_parallel, config.runs_per_cycle)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(launch, range(1, config.runs_per_cycle + 1)))


def prune_session_directories(
    data_dir: str | Path,
    max_sessions: int,
    dry_run: bool = False,
) -> list[Path]:
    if max_sessions < 0:
        raise ValueError("max_sessions must be non-negative")
    root = Path(data_dir).expanduser().resolve()
    if max_sessions == 0 or not root.is_dir():
        return []
    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir() and (path / "sim_data.bin").is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    removed: list[Path] = []
    for path in candidates[max_sessions:]:
        if path == root or root not in path.parents:
            continue
        removed.append(path)
        if not dry_run:
            if path.is_symlink():
                path.unlink()
            else:
                shutil.rmtree(path)
    return removed


def cleanup_storage(data_dir: str | Path, max_sessions: int, dry_run: bool = False) -> list[Path]:
    return prune_session_directories(data_dir, max_sessions, dry_run)


def _cycle_training_config(config: ActiveLearningConfig, cycle: int) -> TrainingConfig:
    shared: dict[str, Any] = {
        "data_dir": config.data_dir,
        "output_dir": config.output_dir / f"cycle_{cycle}",
        "epochs": config.epochs,
        "device": "auto",
        "batch_size": config.batch_size,
        "effective_batch_size": config.effective_batch_size,
        "skip_frames": config.skip_frames,
        "n_steps": config.n_steps,
        "skip_initial": config.skip_initial,
        "learning_rate": config.learning_rate,
        "validation_fraction": config.validation_fraction,
        "gradient_clip_norm": config.gradient_clip_norm,
        "noise_std": config.noise_std,
        "num_workers": config.num_workers,
        "model_filename": config.model_filename,
        "seed": config.seed,
        "model_config": ModelConfig(model_variant=config.model_variant),
    }
    if config.training_config is None:
        return TrainingConfig(**shared)
    training_config = copy.deepcopy(config.training_config)
    if isinstance(training_config, Mapping):
        training_config = TrainingConfig(**dict(training_config))
    training_config.data_dir = config.data_dir
    training_config.output_dir = shared["output_dir"]
    for name in (
        "batch_size",
        "effective_batch_size",
        "skip_frames",
        "n_steps",
        "skip_initial",
        "learning_rate",
        "validation_fraction",
        "gradient_clip_norm",
        "noise_std",
        "num_workers",
        "model_filename",
    ):
        setattr(training_config, name, shared[name])
    training_config.epochs = config.epochs
    training_config.seed = config.seed
    if training_config.model_config is None:
        training_config.model_config = ModelConfig(model_variant=config.model_variant)
    elif training_config.model_config.model_variant != config.model_variant:
        training_config.model_config = training_config.model_config.replace(model_variant=config.model_variant)
    return training_config


def run_active_learning(config: ActiveLearningConfig) -> list[TrainingResult]:
    if config.cycles <= 0:
        raise ValueError("cycles must be positive")
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[TrainingResult] = []
    previous_checkpoint: Path | None = None
    for cycle in range(1, config.cycles + 1):
        if config.runs_simulation:
            run_simulations(config, cycle)
        if config.prune:
            prune_session_directories(config.data_dir, config.max_sessions)
        if not config.trains:
            continue
        training_config = _cycle_training_config(config, cycle)
        resume_path = None
        if config.resume and previous_checkpoint is not None and Path(previous_checkpoint).is_file():
            resume_path = previous_checkpoint
        result = train_model(config=training_config, resume_path=resume_path)
        results.append(result)
        previous_checkpoint = result.checkpoint
    return results


def _environment_variant() -> str:
    value = os.environ.get("SPH_MODEL_VARIANT") or os.environ.get("SPH_VARIANT") or os.environ.get("MODEL_VARIANT") or os.environ.get("VARIANT")
    if value is None:
        return DEFAULT_MODEL_VARIANT
    return canonical_model_variant(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the bounded neural-compression active-learning loop")
    parser.add_argument("--phase", choices=PHASES, default=PHASE_ALL, help="generate data, train, or both")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="attempts")
    parser.add_argument("--cycles", "--iterations", dest="cycles", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--runs-per-cycle", "--runs_per_cycle", "--runs-per-iteration", "--runs_per_iteration", dest="runs_per_cycle", type=int, default=0)
    parser.add_argument("--max-parallel", "--max_parallel", dest="max_parallel", type=int, default=1)
    parser.add_argument("--max-sessions", "--max_sessions", dest="max_sessions", type=int, default=0)
    parser.add_argument("--simulation-command", default=None)
    parser.add_argument("--model-variant", "--model_variant", "--simulation-variant", "--simulation_variant", "--simulation-mode", "--simulation_mode", "--variant", "--mode", choices=("density", "density_only", "density-only", "density_velocity", "density-velocity", "density+velocity"), default=_environment_variant())
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--latent-dim", type=int, default=1024)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--bottleneck-channels", type=int, default=None)
    parser.add_argument("--context-channels", type=int, default=None)
    parser.add_argument("--projection-dim", type=int, default=None)
    parser.add_argument("--num-downsamples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--frames-per-run", "--frames_per_run", dest="frames_per_run", type=int, default=DEFAULT_FRAMES_PER_RUN)
    parser.add_argument("--simulation-seed", "--simulation_seed", dest="simulation_seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sph-root", "--sph_root", dest="sph_root", default=None)
    parser.add_argument("--skip-sim", action="store_true")
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--effective-batch-size", "--effective_batch_size", dest="effective_batch_size", type=int, default=DEFAULT_EFFECTIVE_BATCH_SIZE)
    parser.add_argument("--skip-frames", "--skip_frames", dest="skip_frames", type=int, default=10)
    parser.add_argument("--n-steps", "--n_steps", dest="n_steps", type=int, default=1)
    parser.add_argument("--skip-initial", "--skip_initial", dest="skip_initial", type=int, default=1)
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=5e-4)
    parser.add_argument("--validation-fraction", "--validation_fraction", dest="validation_fraction", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", "--gradient_clip_norm", dest="gradient_clip_norm", type=float, default=None)
    parser.add_argument("--noise-std", "--noise_std", dest="noise_std", type=float, default=0.0)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--model-filename", "--model_filename", dest="model_filename", default=DEFAULT_MODEL_FILENAME)
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True, help="restart from random init each cycle")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.skip_sim and args.phase == PHASE_GENERATE:
        raise ValueError("--skip-sim cannot be combined with --phase generate; nothing would run")
    phase = PHASE_TRAIN if args.skip_sim else args.phase
    model_config_values: dict[str, Any] = {
        "width": args.width,
        "height": args.height,
        "model_variant": args.model_variant,
        "latent_dim": args.latent_dim,
    }
    for name in ("base_channels", "bottleneck_channels", "context_channels", "projection_dim", "num_downsamples"):
        value = getattr(args, name, None)
        if value is not None:
            model_config_values[name] = value
    model_config = ModelConfig(**model_config_values)
    config = ActiveLearningConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        cycles=args.cycles,
        epochs=args.epochs,
        runs_per_cycle=0 if args.skip_sim else args.runs_per_cycle,
        max_parallel=args.max_parallel,
        max_sessions=args.max_sessions,
        simulation_command=shlex.split(args.simulation_command) if args.simulation_command else None,
        model_variant=args.model_variant,
        seed=args.seed,
        frames_per_run=args.frames_per_run,
        simulation_seed=args.simulation_seed,
        sph_root=args.sph_root,
        prune=args.prune,
        phase=phase,
        batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size,
        skip_frames=args.skip_frames,
        n_steps=args.n_steps,
        skip_initial=args.skip_initial,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        gradient_clip_norm=args.gradient_clip_norm,
        noise_std=args.noise_std,
        num_workers=args.num_workers,
        model_filename=args.model_filename,
        resume=args.resume,
        training_config=TrainingConfig(
            data_dir=Path(args.data_dir).expanduser().resolve(),
            output_dir=Path(args.output_dir).expanduser().resolve(),
            epochs=args.epochs,
            device=args.device,
            smoke=args.smoke,
            max_batches=args.max_batches,
            model_config=model_config,
            seed=args.seed,
        ),
    )
    results = run_active_learning(config)
    if not config.trains:
        sessions = discover_sessions(config.data_dir)
        frame_bytes = model_config.schema.frame_bytes
        print(f"generated {len(sessions)} session(s) in {config.data_dir}")
        for path in sessions:
            frames = (path / "sim_data.bin").stat().st_size // frame_bytes
            print(f"  {path}  frames={frames}")
        return 0
    for result in results:
        print(result.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
