"""
Run the YOLO detector once over every sequence of a split and cache the boxes.

All trackers read this cache, so they work on exactly the same detections.

Example:
    python scripts/detect.py           # val and test
    python scripts/detect.py test
"""

import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking.dataset import list_sequences, load_sequence
from tracking.detections import save_detections
from tracking.runner import cache_path, load_yaml, resolve


def detect_split(model, cfg, split):
    for seq_dir in list_sequences(resolve(cfg["data_root"]), split):
        out = cache_path(cfg, split, seq_dir.name)
        if out.exists():
            print(f"  {seq_dir.name}: cached, skipping")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)

        seq = load_sequence(seq_dir, cfg["modality"])
        per_frame, ms = [], []
        results = model.predict(source=str(seq.video_path), conf=cfg["cache_conf"],
                                stream=True, verbose=False)
        for r in results:
            boxes = r.boxes.cpu().numpy()
            per_frame.append((boxes.xyxy, boxes.conf))
            ms.append(sum(r.speed.values()))  # preprocess + inference + postprocess

        if len(per_frame) != len(seq):
            print(f"  ⚠️  {seq.name}: {len(per_frame)} video frames vs {len(seq)} annotations")
        save_detections(out, per_frame, detector_ms=float(np.mean(ms)))
        print(f"  {seq.name}: {len(per_frame)} frames, {np.mean(ms):.1f} ms/frame")


if __name__ == "__main__":
    cfg = load_yaml("config.yaml")
    splits = sys.argv[1:] or ["val", "test"]

    model_path = resolve(cfg["model"])
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    model = YOLO(str(model_path))

    for split in splits:
        print(f"Detecting on {split}...")
        detect_split(model, cfg, split)
