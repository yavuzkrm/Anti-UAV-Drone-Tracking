"""Shared helpers for the scripts: config loading and running a tracker over a sequence."""

import time
from pathlib import Path

import numpy as np
import yaml

from . import build_tracker
from .dataset import list_sequences, load_sequence
from .detections import load_detections

ROOT = Path(__file__).resolve().parents[1]  # repo root, so scripts run from any directory


def load_yaml(name):
    with open(ROOT / "configs" / name) as f:
        return yaml.safe_load(f)


def resolve(path):
    """Paths in the configs are relative to the repo root."""
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def cache_path(cfg, split, seq_name):
    return resolve(cfg["cache_dir"]) / cfg["modality"] / split / f"{seq_name}.npz"


def load_split(cfg, split):
    """Ground truth and cached detections of every sequence in a split: [(Sequence, detections)]."""
    data = []
    for seq_dir in list_sequences(resolve(cfg["data_root"]), split):
        path = cache_path(cfg, split, seq_dir.name)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run: python scripts/detect.py {split}")
        data.append((load_sequence(seq_dir, cfg["modality"]), load_detections(path)))
    return data


def run_tracker(kind, params, detections, stride=1):
    """
    Run a fresh tracker over one sequence of cached detections.

    stride - the detector runs only on every stride-th frame. The trackers get None
             on the frames in between ("detector did not run") and have to bridge them.

    Returns predictions (T, 4) with NaN rows for "no box", and tracker time in ms/frame.
    """
    tracker = build_tracker(kind, **params)
    pred = np.full((len(detections), 4), np.nan)

    start = time.perf_counter()
    for t, dets in enumerate(detections):
        box = tracker.update(dets if t % stride == 0 else None)
        if box is not None:
            pred[t] = box
    ms_per_frame = (time.perf_counter() - start) * 1000 / max(len(detections), 1)

    return pred, ms_per_frame
