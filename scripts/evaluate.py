"""
Evaluate all trackers on the test split with the parameters tuned on val.

Writes:
    results/test/summary.md        - every table used in the README
    results/test/per_sequence.csv  - main metrics per sequence and method
    results/plots/*.png            - success / precision plots, accuracy vs detector stride

Example:
    python scripts/evaluate.py
"""

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking.metrics import DIST_THRESHOLDS, IOU_THRESHOLDS, box_iou, center_error, evaluate_sequence
from tracking.runner import ROOT, cache_path, load_split, load_yaml, run_tracker

SPLIT = "test"
OUT_DIR = ROOT / "results" / SPLIT
PLOT_DIR = ROOT / "results" / "plots"

# Display name -> tracker kind. Colors: first three categorical slots of the
# reference palette (validated colorblind-safe for any pair of the three).
METHODS = {
    "Detector only": "detector_only",
    "Kalman (ours)": "kalman",
    "ByteTrack": "bytetrack",
}
COLORS = {"Kalman (ours)": "#2a78d6", "ByteTrack": "#eb6834", "Detector only": "#1baf7a"}

# Ablations: the tuned stride-1 parameters with exactly one thing switched off
ABLATIONS = {
    "Kalman, constant velocity instead of acceleration": ("kalman", {"accel": False}),
    "Kalman, no coasting (max_missed = 0)": ("kalman", {"max_missed": 0}),
    "ByteTrack, no low-score 2nd round (BYTE off)": ("bytetrack", "no_byte"),
    "ByteTrack, no coasting (coast = 0)": ("bytetrack", {"coast": 0}),
}

METRIC_COLUMNS = [
    ("state_accuracy", "State acc. ↑"),
    ("success_auc", "Success AUC ↑"),
    ("precision_20", "Precision@20px ↑"),
    ("false_alarm_rate", "False alarm ↓"),
]


def run_method(data, kind, params, stride=1):
    """Run one method on every sequence. Returns per-sequence metrics, predictions and tracker ms/frame."""
    results, preds, ms = [], [], []
    for seq, dets in data:
        pred, t = run_tracker(kind, params, dets, stride)
        results.append(evaluate_sequence(pred, seq.gt_boxes, seq.exist))
        preds.append(pred)
        ms.append(t)
    return results, preds, float(np.mean(ms))


def mean_metric(results, key):
    """Mean over sequences (Anti-UAV protocol). False alarm only exists where the drone is absent."""
    return float(np.nanmean([r[key] for r in results]))


def fmt(x):
    return "-" if np.isnan(x) else f"{x:.3f}"


def metrics_table(rows):
    """rows: [(name, results, extra_cells)]"""
    lines = ["| Method | " + " | ".join(label for _, label in METRIC_COLUMNS) + " |",
             "|---|" + "---:|" * len(METRIC_COLUMNS)]
    for name, results in rows:
        lines.append(f"| {name} | " + " | ".join(fmt(mean_metric(results, k)) for k, _ in METRIC_COLUMNS) + " |")
    return "\n".join(lines)


def detector_miss_table(data, preds_by_method, baseline_preds):
    """
    Frames where the drone is visible but the detector-only baseline reports nothing.
    These are the frames a tracker can win back by predicting through the gap.
    """
    lines = ["| Method | Frames | Precision@20px | IoU > 0.5 |", "|---|---:|---:|---:|"]
    for name, preds in preds_by_method.items():
        hits20, hits_iou, total = 0, 0, 0
        for (seq, _), pred, base in zip(data, preds, baseline_preds):
            mask = seq.exist & np.isnan(base[:, 0])
            total += mask.sum()
            hits20 += (center_error(pred[mask], seq.gt_boxes[mask]) <= 20).sum()
            hits_iou += (box_iou(pred[mask], seq.gt_boxes[mask]) > 0.5).sum()
        lines.append(f"| {name} | {total} | {hits20 / total:.3f} | {hits_iou / total:.3f} |")
    return "\n".join(lines)


def plot_curves(results_by_method):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    for name, results in results_by_method.items():
        success = np.mean([r["success_curve"] for r in results], axis=0)
        precision = np.mean([r["precision_curve"] for r in results], axis=0)
        ax1.plot(IOU_THRESHOLDS, success, color=COLORS[name], linewidth=2,
                 label=f"{name} [AUC {success.mean():.3f}]")
        ax2.plot(DIST_THRESHOLDS, precision, color=COLORS[name], linewidth=2,
                 label=f"{name} [{precision[20]:.3f}]")

    ax1.set(title="Success plot", xlabel="IoU threshold", ylabel="Success rate", ylim=(0, 1.02))
    ax2.set(title="Precision plot", xlabel="Center error threshold (px)", ylabel="Precision", ylim=(0, 1.02))
    ax2.axvline(20, color="#c3c2b7", linewidth=1, linestyle=":")
    for ax in (ax1, ax2):
        ax.grid(True, color="#e8e7e3", linewidth=0.8)
        ax.legend(frameon=False, loc="lower left" if ax is ax1 else "lower right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "success_precision.png", dpi=150)
    plt.close(fig)


def plot_stride(stride_scores):
    """stride_scores: {method: [(stride, state_accuracy)]}"""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for name, points in stride_scores.items():
        k, sa = zip(*points)
        ax.plot(k, sa, color=COLORS[name], linewidth=2, marker="o", markersize=7, label=name)
    ax.set(title="Accuracy when the detector runs on every k-th frame",
           xlabel="Detector stride k", ylabel="State accuracy (test)", xticks=k)
    ax.grid(True, color="#e8e7e3", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "stride.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    cfg = load_yaml("config.yaml")
    tuned = load_yaml("trackers.yaml")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_split(cfg, SPLIT)
    detector_ms = float(np.mean([np.load(cache_path(cfg, SPLIT, seq.name))["detector_ms"] for seq, _ in data]))
    params = tuned["stride_1"]
    report = [f"# Results on the {SPLIT} split ({len(data)} sequences, {sum(len(s) for s, _ in data)} frames)\n",
              "All numbers are means over sequences. Parameters were tuned on the val split "
              "(`configs/trackers.yaml`).\n"]

    # 1. Main comparison
    main_results, main_preds, speed_rows = {}, {}, []
    for name, kind in METHODS.items():
        results, preds, ms = run_method(data, kind, params[kind])
        main_results[name], main_preds[name] = results, preds
        speed_rows.append(f"| {name} | {ms:.3f} |")
        print(f"{name:15s} SA {mean_metric(results, 'state_accuracy'):.4f}")
    report += ["## Main comparison\n", metrics_table(main_results.items()), ""]
    report += [f"Detector (YOLO, GPU): {detector_ms:.1f} ms/frame. Tracker overhead on top of it:\n",
               "| Method | Tracker ms/frame |", "|---|---:|", *speed_rows, ""]

    # 2. Frames the detector misses
    report += ["## Frames where the detector misses the drone\n",
               "Drone visible, but no detection above the baseline's threshold.\n",
               detector_miss_table(data, main_preds, main_preds["Detector only"]), ""]

    # 3. Ablations
    ablation_rows = [(n, main_results[n]) for n in ["Kalman (ours)", "ByteTrack"]]
    for name, (kind, change) in ABLATIONS.items():
        p = dict(params[kind])
        if change == "no_byte":
            p["track_low_thresh"] = p["track_high_thresh"]
        else:
            p.update(change)
        ablation_rows.append((name, run_method(data, kind, p)[0]))
    report += ["## Ablations\n", "Tuned parameters with one component switched off.\n",
               metrics_table(ablation_rows), ""]

    # 4. Detector stride
    strides = sorted(int(key.split("_")[1]) for key in tuned)
    stride_scores = {name: [] for name in METHODS}
    lines = ["| Detector stride | Detector ms/frame | " + " | ".join(METHODS) + " |",
             "|---:|---:|" + "---:|" * len(METHODS)]
    for k in strides:
        cells = []
        for name, kind in METHODS.items():
            results = main_results[name] if k == 1 else run_method(data, kind, tuned[f"stride_{k}"][kind], k)[0]
            sa = mean_metric(results, "state_accuracy")
            stride_scores[name].append((k, sa))
            cells.append(f"{sa:.3f}")
        lines.append(f"| {k} | {detector_ms / k:.1f} | " + " | ".join(cells) + " |")
    report += ["## Running the detector on every k-th frame\n",
               "State accuracy when YOLO only runs on every k-th frame and the tracker bridges the rest "
               "(parameters tuned on val separately for each k).\n", *lines, ""]

    # Per-sequence table
    with open(OUT_DIR / "per_sequence.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "frames", "absent_frames", "method", *[k for k, _ in METRIC_COLUMNS]])
        for i, (seq, _) in enumerate(data):
            for name in METHODS:
                r = main_results[name][i]
                writer.writerow([seq.name, len(seq), int((~seq.exist).sum()), name,
                                 *[round(float(r[k]), 4) for k, _ in METRIC_COLUMNS]])

    (OUT_DIR / "summary.md").write_text("\n".join(report), encoding="utf-8")
    plot_curves(main_results)
    plot_stride(stride_scores)
    print("\n".join(report))
