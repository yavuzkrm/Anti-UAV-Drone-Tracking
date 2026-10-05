"""
Cached detector output.

The detector runs once per sequence and its boxes are saved to disk. Every tracker
then reads the same file, so the comparison measures tracking only - the detections
are identical for all methods. Boxes are kept down to a low score (0.05) because
ByteTrack's second association stage uses low-score detections.
"""

from pathlib import Path

import numpy as np


class Detections:
    """
    Detections of one frame.
    Exposes conf / cls / xywh and boolean indexing, which is the interface
    ultralytics' BYTETracker expects (same as ultralytics Boxes).
    """

    def __init__(self, xyxy, conf):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.zeros_like(self.conf)  # single class: drone

    @property
    def xywh(self):
        """Center x, center y, width, height"""
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return Detections(self.xyxy[mask], self.conf[mask])

    def above(self, threshold):
        return self[self.conf >= threshold]


def save_detections(path, per_frame, detector_ms):
    """per_frame: list of (xyxy (N, 4), conf (N,)) per frame"""
    frame_ids = np.concatenate([np.full(len(conf), t) for t, (_, conf) in enumerate(per_frame)])
    np.savez_compressed(
        path,
        num_frames=len(per_frame),
        frame_ids=frame_ids.astype(np.int32),
        xyxy=np.concatenate([xyxy for xyxy, _ in per_frame]).astype(np.float32).reshape(-1, 4),
        conf=np.concatenate([conf for _, conf in per_frame]).astype(np.float32),
        detector_ms=detector_ms,
    )


def load_detections(path):
    """Return a list with one Detections object per frame."""
    data = np.load(Path(path))
    frame_ids, xyxy, conf = data["frame_ids"], data["xyxy"], data["conf"]
    return [Detections(xyxy[frame_ids == t], conf[frame_ids == t])
            for t in range(int(data["num_frames"]))]
