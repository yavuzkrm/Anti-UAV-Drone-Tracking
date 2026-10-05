"""
Tracking metrics against ground truth.

Predictions and ground truth are (T, 4) arrays of [x1, y1, x2, y2] per frame,
with NaN rows for "no box" (tracker reported nothing / drone not visible).

    State accuracy (SA) - official Anti-UAV metric. Per frame:
                            drone visible -> IoU with the ground truth (0 if no box reported)
                            drone absent  -> 1 if no box reported, else 0
                          then averaged over all frames. Rewards both following the drone
                          and correctly saying "it is not there".
    Success AUC         - fraction of visible frames with IoU > t, averaged over
                          t = 0, 0.05, ..., 1 (area under the success plot, OTB style)
    Precision@20px      - fraction of visible frames whose predicted center is within
                          20 px of the ground-truth center
    False alarm rate    - fraction of absent frames where a box was still reported
"""

import numpy as np

IOU_THRESHOLDS = np.linspace(0, 1, 21)
DIST_THRESHOLDS = np.arange(0, 51)


def box_iou(a, b):
    """Row-wise IoU of two (T, 4) arrays. NaN rows give IoU 0."""
    x1 = np.maximum(a[:, 0], b[:, 0])
    y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 2], b[:, 2])
    y2 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = inter / (area_a + area_b - inter)
    return np.nan_to_num(iou, nan=0.0)


def center_error(a, b):
    """Row-wise distance between box centers. NaN rows give infinity."""
    ca = (a[:, :2] + a[:, 2:]) / 2
    cb = (b[:, :2] + b[:, 2:]) / 2
    return np.nan_to_num(np.linalg.norm(ca - cb, axis=1), nan=np.inf)


def evaluate_sequence(pred, gt, exist):
    has_pred = ~np.isnan(pred[:, 0])
    iou = box_iou(pred, gt)
    dist = center_error(pred, gt)

    vis_iou, vis_dist = iou[exist], dist[exist]
    success_curve = np.array([(vis_iou > t).mean() for t in IOU_THRESHOLDS])
    precision_curve = np.array([(vis_dist <= d).mean() for d in DIST_THRESHOLDS])

    return {
        "state_accuracy": np.where(exist, iou, ~has_pred).mean(),
        "success_auc": success_curve.mean(),
        "precision_20": (vis_dist <= 20).mean(),
        "mean_iou": vis_iou.mean(),
        "false_alarm_rate": has_pred[~exist].mean() if (~exist).any() else np.nan,
        "success_curve": success_curve,
        "precision_curve": precision_curve,
    }
