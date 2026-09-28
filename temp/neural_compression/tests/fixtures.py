from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_session(
    root: Path,
    name: str = "session",
    width: int = 8,
    height: int = 8,
    frames: int = 5,
    variant: str = "density",
    metadata: bool = True,
    partial: bool = True,
) -> Path:
    session = root / name
    session.mkdir(parents=True, exist_ok=True)
    values = np.arange(frames * 4 * height * width, dtype=np.float32).reshape(frames, 4, height, width)
    values[:, 0] = np.maximum(values[:, 0] / max(1.0, float(values[:, 0].max())), 0.01)
    values[:, 1] = values[:, 1] / 100.0
    values[:, 2] = -values[:, 2] / 100.0
    values[:, 3] = (np.indices((height, width))[0] + np.indices((height, width))[1]) % 2
    values.tofile(session / "sim_data.bin")
    if partial:
        with (session / "sim_data.bin").open("ab") as handle:
            handle.write(b"partial")
    if metadata:
        (session / "metadata.json").write_text(
            json.dumps(
                {
                    "width": width,
                    "height": height,
                    "fields": ["density", "velocity_x", "velocity_y", "obstacle_mask"],
                    "dtype": "float32",
                    "model_variant": variant,
                }
            ),
            encoding="utf-8",
        )
    return session
