"""
Unit tests that need no dataset or GPU.

    python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking import build_tracker
from tracking.detections import Detections
from tracking.kalman import KalmanFilter
from tracking.metrics import box_iou, evaluate_sequence

NAN_BOX = [np.nan] * 4


def moving_target(frames=40, vx=5.0, vy=2.0, size=20.0):
    """Boxes of a target moving with constant velocity."""
    t = np.arange(frames)
    cx, cy = 100 + vx * t, 100 + vy * t
    return np.stack([cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2], axis=1)


def test_kalman_learns_velocity_and_predicts_through_gap():
    kf = KalmanFilter(100, 100)
    for t in range(1, 21):
        kf.predict()
        kf.update([100 + 5 * t, 100 + 2 * t])
    np.testing.assert_allclose(kf.velocity, [5, 2], atol=0.1)

    for _ in range(10):  # no measurements: prediction only
        kf.predict()
    np.testing.assert_allclose(kf.position, [250, 160], atol=1.0)


def test_kalman_tracker_bridges_missing_detections():
    boxes = moving_target()
    tracker = build_tracker("kalman", det_thresh=0.5, max_missed=10)
    for t, box in enumerate(boxes):
        dets = Detections([box], [0.9]) if not 25 <= t < 30 else Detections(np.zeros((0, 4)), [])
        out = tracker.update(dets)
        if 25 <= t < 30:
            assert tracker.status == "predicted"
            assert box_iou(out[None], box[None])[0] > 0.7


def test_kalman_tracker_drops_track_after_max_missed():
    tracker = build_tracker("kalman", det_thresh=0.5, max_missed=2)
    tracker.update(Detections([[0, 0, 10, 10]], [0.9]))
    empty = Detections(np.zeros((0, 4)), [])
    outputs = [tracker.update(empty) for _ in range(3)]
    assert outputs[1] is not None and outputs[2] is None


def test_detector_only_holds_last_box_when_detector_skipped():
    tracker = build_tracker("detector_only", det_thresh=0.5)
    first = tracker.update(Detections([[0, 0, 10, 10], [5, 5, 20, 20]], [0.6, 0.8]))
    np.testing.assert_array_equal(first, [5, 5, 20, 20])  # highest score wins
    np.testing.assert_array_equal(tracker.update(None), first)


def test_bytetrack_follows_constant_motion():
    boxes = moving_target()
    tracker = build_tracker("bytetrack", track_high_thresh=0.5, new_track_thresh=0.6)
    outputs = [tracker.update(Detections([box], [0.9])) for box in boxes]
    tail = np.array(outputs[5:])
    assert np.all(box_iou(tail, boxes[5:]) > 0.7)


def test_state_accuracy_rewards_correct_absence():
    gt = np.array([[0, 0, 10, 10], NAN_BOX, NAN_BOX])
    exist = np.array([True, False, False])
    # Perfect box on the visible frame, nothing reported on the absent frames -> 1.0
    pred = np.array([[0, 0, 10, 10], NAN_BOX, NAN_BOX])
    assert evaluate_sequence(pred, gt, exist)["state_accuracy"] == 1.0
    # A box reported where the drone is absent counts as an error
    pred[1] = [0, 0, 10, 10]
    r = evaluate_sequence(pred, gt, exist)
    assert np.isclose(r["state_accuracy"], 2 / 3)
    assert r["false_alarm_rate"] == 0.5


def test_box_iou():
    a = np.array([[0, 0, 10, 10], [0, 0, 10, 10], NAN_BOX])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [0, 0, 10, 10]])
    np.testing.assert_allclose(box_iou(a, b), [1.0, 50 / 150, 0.0])
