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


def run_iterative_test(
    run_name: str,
    model_path: str | Path,
    data_path: str | Path,
    steps: int = 2,
    skip: int = 10,
    index: int = 0,
    device: str = "auto",
    output_png: str | Path | None = None,
    plot: bool = False,
):
    result = evaluate_checkpoint(
        model_path,
        data_path,
        index=index,
        steps=steps,
        skip=skip,
        device=device,
    )
    if plot:
        if output_png is None:
            safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in run_name)
            output_png = Path.cwd() / f"stability_{safe_name}.png"
        result.plot = plot_evaluation(result, output_png)
    payload = result.to_dict()
    payload["run_name"] = run_name
    print(json.dumps(payload, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a two-step autoregressive checkpoint evaluation")
    parser.add_argument("--run", default="SPH_Reconstruction")
    parser.add_argument("--model", "--model-path", "--model_path", dest="model", required=True)
    parser.add_argument("--data", "--data-path", "--data_path", dest="data", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--skip", "--skip-frames", "--skip_frames", dest="skip", type=int, default=10)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", "--output", dest="out", default=None)
    parser.add_argument("--plot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_iterative_test(
            args.run,
            args.model,
            args.data,
            steps=args.steps,
            skip=args.skip,
            index=args.index,
            device=args.device,
            output_png=args.out,
            plot=args.plot,
        )
    except (FileNotFoundError, OSError, ValueError, RuntimeError, TypeError, IndexError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
