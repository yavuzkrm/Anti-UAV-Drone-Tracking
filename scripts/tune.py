"""
Grid search of tracker parameters on the validation split.

Every combination below is run on all val sequences and scored by mean state
accuracy. The best one per method is written to configs/trackers.yaml, which
evaluate.py then uses on the test split. Test data is never used for tuning.

The search is repeated for each detector stride (detector runs on every k-th frame),
since bridging longer gaps needs different settings (e.g. a longer max_missed).

Example:
    python scripts/tune.py              # strides 1 2 3 5
    python scripts/tune.py 1
"""

import csv
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking.metrics import evaluate_sequence
from tracking.runner import ROOT, load_split, load_yaml, run_tracker

# Detections are cached down to score 0.05, so no threshold below that makes sense
GRIDS = {
    "detector_only": {
        "det_thresh": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    },
    "kalman": {
        "det_thresh": [0.05, 0.1, 0.2, 0.3, 0.5],
        "max_missed": [0, 2, 5, 10, 20],
        "q_scale": [1, 10, 100, 1000],
        "gate": [9.21, 100.0, float("inf")],
    },
    "bytetrack": {
        "track_high_thresh": [0.1, 0.2, 0.3, 0.5],
        "track_low_thresh": [0.05],
        "new_track_thresh": [0.1, 0.2, 0.3, 0.5],
        "match_thresh": [0.8, 0.9, 0.95],
        "coast": [0, 2, 5, 10, 20],
    },
}
STRIDES = [1, 2, 3, 5]

_data = None  # val split, loaded once per worker process


def _init_worker(cfg):
    global _data
    _data = load_split(cfg, "val")


def _score(job):
    kind, params, stride = job
    accs = [evaluate_sequence(run_tracker(kind, params, dets, stride)[0],
                              seq.gt_boxes, seq.exist)["state_accuracy"]
            for seq, dets in _data]
    return float(np.mean(accs))


def combinations(grid):
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


if __name__ == "__main__":
    cfg = load_yaml("config.yaml")
    strides = [int(s) for s in sys.argv[1:]] or STRIDES

    out_dir = ROOT / "results" / "val"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_file = ROOT / "configs" / "trackers.yaml"
    best = yaml.safe_load(best_file.read_text()) if best_file.exists() else {}

    with ProcessPoolExecutor(initializer=_init_worker, initargs=(cfg,)) as pool:
        for stride in strides:
            best[f"stride_{stride}"] = {}
            for kind, grid in GRIDS.items():
                combos = combinations(grid)
                scores = list(pool.map(_score, [(kind, c, stride) for c in combos], chunksize=4))
                i = int(np.argmax(scores))
                best[f"stride_{stride}"][kind] = combos[i]
                print(f"stride {stride} | {kind:13s} | val SA {scores[i]:.4f} | {combos[i]}")

                with open(out_dir / f"tuning_{kind}_stride{stride}.csv", "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=[*grid, "val_state_accuracy"])
                    writer.writeheader()
                    for c, s in sorted(zip(combos, scores), key=lambda x: -x[1]):
                        writer.writerow({**c, "val_state_accuracy": round(s, 4)})

            # Save after every stride, so an interrupted run keeps what it finished
            with open(best_file, "w") as f:
                f.write("# Written by scripts/tune.py - best parameters on the val split per detector stride\n")
                yaml.safe_dump(best, f, sort_keys=False)
