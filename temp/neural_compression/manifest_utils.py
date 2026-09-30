"""Shared helpers for the precomputed simulation dataset.

The active-learning loop used to spawn simulations on demand and keep only a
rolling window of the most recent ones. With the dataset precomputed up front
by ``precompute_dataset.py``, the sims live in one immutable pool described by
a ``manifest.json``. Both loops and ``compressor.py`` read that manifest to pick
a per-cycle subset, so they all agree on which sessions belong to a cycle.

This module deliberately imports nothing from torch so the loop scripts can use
it without paying the CUDA import cost.
"""

import json
import os
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

FRAME_BYTES = 4 * 400 * 400 * 4  # density, v_x, v_y, mask as float32

SUBSET_MODES = ("window", "expanding", "all")

DatasetSelection = namedtuple("DatasetSelection", ["train", "val"])


def sim_bytes(frames: int) -> int:
    """Bytes a simulation of `frames` frames occupies on disk."""
    return frames * FRAME_BYTES


def load_manifest(manifest_path: str) -> Dict:
    """Read a manifest.json written by precompute_dataset.py."""
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    if "entries" not in manifest:
        raise ValueError(f"Malformed manifest (no 'entries'): {manifest_path}")
    return manifest


def manifest_sessions(manifest: Dict, require_complete: bool = True) -> List[Dict]:
    """Completed entries, ordered by generation index."""
    entries = [e for e in manifest["entries"] if e.get("status") == "ok"]
    if require_complete:
        entries = [e for e in entries if e.get("frames", 0) > 0]
    entries.sort(key=lambda e: e["index"])
    return entries


def select_sessions(manifest_path: str, cycle: int, runs_per_iteration: int,
                    subset_mode: str = "window", subset_size: int = 0,
                    n_val: int = 0) -> DatasetSelection:
    """Pick the train/val sessions one active-learning cycle should use.

    This reproduces the data state the online loop would have had at `cycle`:
    only simulations indexed below `cycle * runs_per_iteration` had been
    generated, and the rolling cleanup kept the most recent `subset_size` of
    them. The difference now is that nothing is generated or deleted to make
    that true, the pool is simply read.

    The validation holdout is reserved from the *end of the full pool* so it
    stays the same sessions at every cycle, and is excluded from the training
    window at all times.

    subset_mode:
        window     - newest `subset_size` training sessions available at this cycle
        expanding  - every session available at this cycle (ignores subset_size)
        all        - the whole pool from cycle 1 on
    """
    if subset_mode not in SUBSET_MODES:
        raise ValueError(f"subset_mode must be one of {SUBSET_MODES}, got {subset_mode!r}")

    sessions = manifest_sessions(load_manifest(manifest_path))
    if not sessions:
        return DatasetSelection([], [])

    train_pool, val = split_holdout(sessions, n_val)

    if subset_mode == "all":
        return DatasetSelection(train_pool, val)

    # Only sessions that the online loop would have produced by this cycle.
    available = [s for s in train_pool
                 if s["index"] < cycle * max(1, runs_per_iteration)]
    if not available:
        # Small dataset, or a cycle that predates any training data: fall back
        # to whatever exists rather than training on nothing.
        available = train_pool

    if subset_mode == "expanding" or subset_size <= 0:
        return DatasetSelection(available, val)

    return DatasetSelection(available[-subset_size:], val)


def split_holdout(sessions: List[Dict], n_val: int) -> Tuple[List[Dict], List[Dict]]:
    """
    Split the whole pool into a training pool and a fixed validation holdout.

    The holdout is taken from the *end* of the full manifest, not from the end
    of a per-cycle window. That matters: a sliding window would drag new
    sessions into the holdout every cycle, so val_loss would be measured
    against different data each time. Holding these back once keeps val_loss
    comparable across cycles, which is what best-checkpoint tracking and
    ReduceLROnPlateau both depend on.

    Train and val pools are disjoint by construction, so nothing validated in
    one cycle is trained on in the next.
    """
    sessions = list(sessions)
    if n_val <= 0 or len(sessions) < 2:
        return sessions, []
    n_val = min(n_val, len(sessions) - 1)
    return sessions[:-n_val], sessions[-n_val:]


def session_paths(sessions: List[Dict]) -> List[str]:
    return [s["path"] for s in sessions]


def describe_selection(train: List[Dict], val: List[Dict], subset_mode: str,
                       subset_size: int, runs_per_iteration: int) -> str:
    """One-line human summary of a selection, for training logs."""
    if not train:
        return "Dataset Subset: EMPTY (no completed sessions)"
    idx = [s["index"] for s in train]
    total_gb = sum(s.get("bytes", 0) for s in train) / (1024 ** 3)
    val_note = ""
    if val:
        val_idx = [s["index"] for s in val]
        val_note = (f" | fixed val holdout: {len(val)} sessions "
                    f"(index {min(val_idx)}..{max(val_idx)})")
    return (f"Dataset Subset: {len(train)} sessions "
            f"(mode={subset_mode}, size={subset_size}, runs/cycle={runs_per_iteration}, "
            f"index {min(idx)}..{max(idx)}, {total_gb:.1f} GiB){val_note}")


def find_manifest(dataset_dir: str) -> Optional[str]:
    """Locate a manifest inside a dataset directory, if one was written."""
    candidate = os.path.join(dataset_dir, "manifest.json")
    return candidate if os.path.isfile(candidate) else None
