"""
Anti-UAV dataset reader.

Layout (one folder per sequence):
    <data_root>/{train,val,test}_videos/<sequence>/infrared.mp4
    <data_root>/{train,val,test}_videos/<sequence>/infrared.json

The JSON holds one entry per frame:
    exist   - 1 if the drone is visible in the frame, 0 if not
    gt_rect - [x, y, w, h] (top-left corner, pixels); [] or [0, 0, 0, 0] when absent
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Whole recordings (first 15 chars of the sequence name) that the detector never saw
# during training. Same list as VAL_RECORDINGS in the detection repo's make_splits.py.
# Tracker parameters are tuned on these, final numbers are reported on test_videos.
VAL_RECORDINGS = {"20190925_141417", "20190925_194211"}


@dataclass
class Sequence:
    name: str
    video_path: Path
    exist: np.ndarray    # (T,) bool
    gt_boxes: np.ndarray  # (T, 4) x1, y1, x2, y2 in pixels; NaN where the drone is absent

    def __len__(self):
        return len(self.exist)

    def frames(self):
        """Yield video frames (BGR) one by one."""
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()


def load_sequence(seq_dir, modality="infrared"):
    seq_dir = Path(seq_dir)
    with open(seq_dir / f"{modality}.json") as f:
        ann = json.load(f)

    exist = np.array(ann["exist"], dtype=bool)
    gt_boxes = np.full((len(exist), 4), np.nan)
    for t, rect in enumerate(ann["gt_rect"]):
        # Some files wrap the box in an extra list, absent frames use [] or [0, 0, 0, 0]
        if len(rect) == 1:
            rect = rect[0]
        if exist[t] and len(rect) == 4 and rect[2] > 0 and rect[3] > 0:
            x, y, w, h = rect
            gt_boxes[t] = [x, y, x + w, y + h]
        else:
            exist[t] = False

    return Sequence(seq_dir.name, seq_dir / f"{modality}.mp4", exist, gt_boxes)


def list_sequences(data_root, split):
    """
    Sequence folders of a split.
        val  - every sequence of the held-out VAL_RECORDINGS (from train_videos and val_videos)
        test - every sequence in test_videos
    """
    data_root = Path(data_root)
    if split == "test":
        dirs = (data_root / "test_videos").iterdir()
    elif split == "val":
        dirs = [d for folder in ["train_videos", "val_videos"]
                for d in (data_root / folder).iterdir() if d.name[:15] in VAL_RECORDINGS]
    else:
        raise ValueError(f"Unknown split: {split} (use 'val' or 'test')")

    return sorted((d for d in dirs if d.is_dir()), key=lambda d: d.name)
