from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Sequence

from .dataset import discover_sessions
from .evaluation import evaluate_checkpoint, plot_evaluation
from .schema import DEFAULT_MODEL_VARIANT, HEIGHT, WIDTH, ModelConfig, canonical_model_variant
from .training import TrainingConfig, train_model


def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", default=None)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="attempts")
    parser.add_argument("--session-dirs", "--session_dirs", dest="session_dirs", default=None)
    parser.add_argument("--model-variant", "--model_variant", "--variant", dest="model_variant", choices=("density", "density_only", "density-only", "density_velocity", "density-velocity", "density+velocity"), default=DEFAULT_MODEL_VARIANT)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--latent-dim", "--latent_dim", dest="latent_dim", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--bottleneck-channels", type=int, default=None)
    parser.add_argument("--context-channels", type=int, default=None)
    parser.add_argument("--projection-dim", type=int, default=None)
    parser.add_argument("--num-downsamples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument("--effective-batch-size", "--effective_batch_size", dest="effective_batch_size", type=int, default=None)
    parser.add_argument("--skip-frames", "--skip_frames", dest="skip_frames", type=int, default=10)
    parser.add_argument("--n-steps", "--n_steps", dest="n_steps", type=int, default=1)
    parser.add_argument("--skip-initial", "--skip_initial", dest="skip_initial", type=int, default=1)
    parser.add_argument("--max-batches", "--max_batches", dest="max_batches", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=5e-4)
    parser.add_argument("--validation-fraction", "--validation_fraction", dest="validation_fraction", type=float, default=0.1)
    parser.add_argument("--model-filename", "--model_name", dest="model_filename", default="best_model.pth")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--amp", dest="use_amp", action="store_true")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--use-8bit-adam", "--use_8bit_adam", dest="use_8bit_adam", action="store_true")


def _add_check_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True)
    parser.add_argument("--index", "--idx", dest="index", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-frames", "--skip_frames", dest="skip_frames", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", "--out", dest="output", default="check_model.png")
    parser.add_argument("--model-variant", "--model_variant", "--variant", dest="model_variant", choices=("density", "density_only", "density-only", "density_velocity", "density-velocity", "density+velocity"), default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pls-compression")
    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train")
    _add_train_arguments(train_parser)
    check_parser = subparsers.add_parser("check")
    _add_check_arguments(check_parser)
    info_parser = subparsers.add_parser("info")
    info_parser.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True)
    return parser


def _model_config_from_args(args: argparse.Namespace) -> ModelConfig:
    values: dict[str, object] = {
        "width": args.width if args.width is not None else WIDTH,
        "height": args.height if args.height is not None else HEIGHT,
        "model_variant": args.model_variant,
        "latent_dim": args.latent_dim if args.latent_dim is not None else 1024,
    }
    for name in ("base_channels", "bottleneck_channels", "context_channels", "projection_dim", "num_downsamples"):
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    return ModelConfig(**values)


def _run_train(args: argparse.Namespace) -> int:
    session_dirs = None
    if args.session_dirs:
        if "," in args.session_dirs:
            session_dirs = [value.strip() for value in args.session_dirs.split(",") if value.strip()]
        else:
            session_dirs = [value.strip() for value in shlex.split(args.session_dirs) if value.strip()]
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_config=_model_config_from_args(args),
        epochs=args.epochs,
        batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size,
        skip_frames=args.skip_frames,
        n_steps=args.n_steps,
        skip_initial=args.skip_initial,
        device=args.device,
        max_batches=args.max_batches,
        smoke=args.smoke,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        use_amp=args.use_amp,
        bf16=args.bf16,
        fused=args.fused,
        use_8bit_adam=args.use_8bit_adam,
        model_filename=args.model_filename,
    )
    result = train_model(sessions=session_dirs, config=config)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _run_check(args: argparse.Namespace) -> int:
    index = getattr(args, "index", 0)
    steps = getattr(args, "steps", 2)
    skip_frames = getattr(args, "skip_frames", 1)
    model_variant = getattr(args, "model_variant", None)
    if index < 0:
        raise ValueError("index must be non-negative")
    result = evaluate_checkpoint(
        args.model,
        args.data_dir,
        index=index,
        steps=steps,
        skip=skip_frames,
        device=getattr(args, "device", "auto"),
    )
    if model_variant is not None and result.variant != canonical_model_variant(model_variant):
        raise ValueError(f"checkpoint variant {result.variant} does not match requested {model_variant}")
    if getattr(args, "plot", False):
        result.plot = plot_evaluation(result, getattr(args, "output", "check_model.png"))
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _run_info(args: argparse.Namespace) -> int:
    sessions = discover_sessions(args.data_dir)
    print(json.dumps({"sessions": [str(path) for path in sessions]}, indent=2))
    return 0


def _has_variant_flag(values: Sequence[str]) -> bool:
    return any(value in {"--model-variant", "--model_variant", "--variant"} or value.startswith("--model-variant=") or value.startswith("--model_variant=") or value.startswith("--variant=") for value in values)


def main(argv: Sequence[str] | None = None, default_variant: str | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values:
        values = ["train"]
    elif values[0].startswith("-"):
        values = ["train", *values]
    if default_variant is not None and values[0] == "train" and not _has_variant_flag(values[1:]):
        values = ["train", "--model-variant", default_variant, *values[1:]]
    parser = build_parser()
    args = parser.parse_args(values)
    if args.command is None:
        parser.print_help()
        return 2
    try:
        if args.command == "train":
            return _run_train(args)
        if args.command == "check":
            return _run_check(args)
        return _run_info(args)
    except (FileNotFoundError, OSError, ValueError, RuntimeError, TypeError, IndexError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def cli_main(argv: Sequence[str] | None = None) -> int:
    return main(argv)
