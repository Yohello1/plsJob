#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pls_compression.evaluation import evaluate_checkpoint, plot_evaluation


def visualize_reconstruction(
    model_path: str | Path,
    data_dir: str | Path,
    output_png: str | Path = "reconstruction.png",
    frame_idx: int | None = None,
    skip: int = 1,
    steps: int = 1,
    device: str = "auto",
    plot: bool = False,
):
    index = 0 if frame_idx is None else frame_idx
    result = evaluate_checkpoint(
        model_path,
        data_dir,
        index=index,
        steps=steps,
        skip=skip,
        device=device,
    )
    if plot:
        result.plot = plot_evaluation(result, output_png)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct a saved SPH compression checkpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", "--data-dir", "--data_dir", dest="data", required=True)
    parser.add_argument("--out", "--output", dest="out", default="reconstruction.png")
    parser.add_argument("--idx", "--index", dest="idx", type=int, default=None)
    parser.add_argument("--skip", "--skip-frames", "--skip_frames", dest="skip", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        visualize_reconstruction(
            args.model,
            args.data,
            output_png=args.out,
            frame_idx=args.idx,
            skip=args.skip,
            steps=args.steps,
            device=args.device,
            plot=args.plot,
        )
    except (FileNotFoundError, OSError, ValueError, RuntimeError, TypeError, IndexError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
