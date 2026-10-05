"""
Single-target tracker built on the Kalman Filter in kalman.py.

Every frame:
    1. predict where the drone should be
    2. keep detections above det_thresh that are close enough to the prediction (gating)
    3. found one  -> update the filter with it
       found none -> keep the prediction ("coasting") and count a miss
    4. too many misses in a row -> drop the track, start again from the next detection

The filter only tracks the center point; box width/height are smoothed separately.
"""

import numpy as np

from .kalman import KalmanFilter

# Chi-square value for 2 degrees of freedom at 99%: a detection farther than this
# (in Mahalanobis distance) from the prediction is treated as a different object.
GATE_CHI2_99 = 9.21


class KalmanTracker:

    def __init__(self, det_thresh=0.5, max_missed=10, q_scale=1.0, r_std=2.0, accel=True,
                 gate=GATE_CHI2_99, size_alpha=0.3):
        """
        det_thresh - minimum detection score that is used
        max_missed - frames to keep predicting without a detection before dropping the track
        q_scale    - process noise scale, see KalmanFilter
        r_std      - detector center error in pixels, see KalmanFilter
        accel      - constant acceleration (True) or constant velocity (False) model
        gate       - Mahalanobis gate (squared), see GATE_CHI2_99
        size_alpha - weight of the new detection when smoothing box width/height
        """
        self.det_thresh = det_thresh
        self.max_missed = max_missed
        self.q_scale = q_scale
        self.r_std = r_std
        self.accel = accel
        self.gate = gate
        self.size_alpha = size_alpha

        self.kf = None
        self.size = None   # smoothed [w, h]
        self.missed = 0
        self.status = "lost"  # "detected", "predicted" or "lost", useful for visualization

    def update(self, detections):
        """
        Process one frame. Returns the drone box [x1, y1, x2, y2] or None.
        detections=None means the detector did not run on this frame: only predict,
        and do not count it as a miss.
        """
        if detections is None:
            if self.kf is None:
                return None
            self.kf.predict()
            self.status = "predicted"
            return self._box()

        dets = detections.above(self.det_thresh)

        if self.kf is None:
            return self._start(dets)

        self.kf.predict()

        if len(dets) > 0:
            centers = dets.xywh[:, :2]
            dist = self.kf.mahalanobis(centers)
            if dist.min() < self.gate:
                i = int(dist.argmin())  # closest detection inside the gate
                self.kf.update(centers[i])
                self.size = self.size_alpha * dets.xywh[i, 2:] + (1 - self.size_alpha) * self.size
                self.missed = 0
                self.status = "detected"
                return self._box()

        # No usable detection: coast on the prediction
        self.missed += 1
        if self.missed > self.max_missed:
            self.kf = None
            return self._start(dets)  # re-acquire right away if something was detected

        self.status = "predicted"
        return self._box()

    def _start(self, dets):
        """Start a new track from the highest-scoring detection."""
        if len(dets) == 0:
            self.status = "lost"
            return None
        i = int(dets.conf.argmax())
        cx, cy, w, h = dets.xywh[i]
        self.kf = KalmanFilter(cx, cy, q_scale=self.q_scale, r_std=self.r_std, accel=self.accel)
        self.size = np.array([w, h])
        self.missed = 0
        self.status = "detected"
        return self._box()

    def _box(self):
        cx, cy = self.kf.position
        w, h = self.size
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
