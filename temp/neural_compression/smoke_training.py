#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pls_compression.dataset import SPHDataset, compute_global_stats, discover_sessions
from pls_compression.losses import hybrid_loss
from pls_compression.models import build_model
from pls_compression.schema import NormalizationMetadata, canonical_model_variant
from pls_compression.training import select_device

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_SPH_ROOT = PACKAGE_ROOT.parent / "sph"


def run_smoke(
    variant: str,
    data_root: str | Path | None = None,
    device: str = "auto",
    sph_root: str | Path = DEFAULT_SPH_ROOT,
    frames: int = 2,
) -> dict[str, object]:
    model_variant = canonical_model_variant(variant)
    target = "draw2-density-only" if model_variant == "density" else "draw2-density-velocity"
    simulator = Path(sph_root).expanduser().resolve() / target
    if not simulator.is_file() or not os.access(simulator, os.X_OK):
        raise FileNotFoundError(f"simulator binary is not built: {simulator}")
    selected_device = select_device(device)
    temporary_directory = None
    if data_root is None:
        temporary_directory = tempfile.TemporaryDirectory(prefix="pls-compression-smoke-")
        root = Path(temporary_directory.name)
    else:
        root = Path(data_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
    try:
        environment = os.environ.copy()
        environment["SPH_DATA_ROOT"] = str(root)
        subprocess.run(
            [
                str(simulator),
                "--headless",
                "--frames",
                str(frames),
                "--fluid",
                "120",
                "120",
                "24",
                "24",
            ],
            cwd=Path(sph_root).expanduser().resolve(),
            env=environment,
            check=True,
            timeout=300,
        )
        sessions = discover_sessions(root)
        if not sessions:
            raise FileNotFoundError("native smoke simulation produced no sessions")
        model = build_model(model_variant).to(selected_device)
        density_max, velocity_max = compute_global_stats(sessions, model.config.schema)
        normalization = NormalizationMetadata.from_maxima(density_max, velocity_max)
        dataset = SPHDataset(
            sessions,
            skip=1,
            n_steps=1,
            augment=False,
            schema=model.config.schema,
            normalization=normalization,
        )
        try:
            if len(dataset) == 0:
                raise ValueError("native smoke simulation produced no complete training sample")
            batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
            values = [value.to(selected_device, dtype=torch.float32) for value in batch]
            model.train()
            prediction = model(values[0], values[1], values[2][:, 0], values[3][:, 0], values[4])
            target_value = values[2][:, 0]
            if model.output_channels == 3:
                target_value = torch.cat((target_value, values[3][:, 0]), dim=1)
            loss = hybrid_loss(prediction, target_value)
            if not torch.isfinite(loss):
                raise RuntimeError("production smoke loss is not finite")
            loss.backward()
            gradient_norm = math.sqrt(
                sum(float(parameter.grad.detach().float().square().sum().cpu()) for parameter in model.parameters() if parameter.grad is not None)
            )
            torch.optim.AdamW(model.parameters(), lr=1e-5).step()
            return {
                "variant": model_variant,
                "device": str(selected_device),
                "shape": list(prediction.shape),
                "loss": float(loss.detach().cpu()),
                "gradient_norm": gradient_norm,
                "normalization": normalization.to_dict(),
                "batches": 1,
                "completed_epochs": 0,
            }
        finally:
            dataset.close()
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one real 400x400 model training batch")
    parser.add_argument("--variant", choices=("density", "density_velocity"), default="density")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--sph-root", default=str(DEFAULT_SPH_ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frames", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frames < 2:
        print("error: --frames must be at least 2", file=sys.stderr)
        return 2
    try:
        result = run_smoke(args.variant, args.data_root, args.device, args.sph_root, args.frames)
    except (FileNotFoundError, OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
