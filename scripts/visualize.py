"""
Render a test sequence with the ground truth and the trackers' boxes.

    white   - ground truth
    blue    - Kalman (ours)
    orange  - ByteTrack
Dashed boxes are predictions made without a matching detection (coasting).

Example:
    python scripts/visualize.py 20190925_111757_1_7
    python scripts/visualize.py 20190925_111757_1_7 --start 300 --frames 150 --gif results/demo.gif
    python scripts/visualize.py 20190925_111757_1_7 --stride 3     # detector on every 3rd frame
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking import build_tracker
from tracking.dataset import list_sequences, load_sequence
from tracking.detections import load_detections
from tracking.runner import ROOT, cache_path, load_yaml, resolve

# BGR, same hues as the plots
STYLES = {
    "kalman": ("Kalman", (214, 120, 42)),
    "bytetrack": ("ByteTrack", (52, 104, 235)),
}
GT_COLOR = (255, 255, 255)
GIF_SCALE = 0.75
GIF_FRAME_STEP = 2  # every 2nd frame at 10 fps: same speed, half the file size


def draw_box(frame, box, color, dashed=False):
    x1, y1, x2, y2 = np.round(box).astype(int)
    if not dashed:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        return
    for (ax, ay), (bx, by) in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                               ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
        n = max(int(np.hypot(bx - ax, by - ay) // 6), 1)
        for i in range(0, n, 2):
            p = (int(ax + (bx - ax) * i / n), int(ay + (by - ay) * i / n))
            q = (int(ax + (bx - ax) * (i + 1) / n), int(ay + (by - ay) * (i + 1) / n))
            cv2.line(frame, p, q, color, 2)


def add_legend_banner(frame, t, stride, statuses, height=56):
    """Legend on a dark strip above the frame, so it never covers the video's own overlay text."""
    banner = np.full((height, frame.shape[1], 3), 30, dtype=np.uint8)
    title = f"frame {t}" + (f"  |  detector runs every {stride} frames" if stride > 1 else "")
    cv2.putText(banner, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)

    items = [("ground truth", GT_COLOR)] + [(f"{STYLES[k][0]}: {statuses[k]}", STYLES[k][1]) for k in STYLES]
    x = 10
    for text, color in items:
        cv2.rectangle(banner, (x, 34), (x + 14, 48), color, -1)
        cv2.putText(banner, text, (x + 20, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        x += 20 + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] + 22
    return np.vstack([banner, frame])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence")
    parser.add_argument("--split", default="test")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None, help="number of frames to render")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--out", default=None, help="mp4 path (default results/videos/<sequence>.mp4)")
    parser.add_argument("--gif", default=None, help="also write a GIF (scaled by GIF_SCALE) here")
    args = parser.parse_args()

    cfg = load_yaml("config.yaml")
    params = load_yaml("trackers.yaml")[f"stride_{args.stride}"]
    seq_dir = next(d for d in list_sequences(resolve(cfg["data_root"]), args.split) if d.name == args.sequence)
    seq = load_sequence(seq_dir, cfg["modality"])
    detections = load_detections(cache_path(cfg, args.split, seq.name))
    trackers = {kind: build_tracker(kind, **params[kind]) for kind in STYLES}

    end = len(seq) if args.frames is None else min(len(seq), args.start + args.frames)
    out_path = Path(args.out) if args.out else ROOT / "results" / "videos" / f"{seq.name}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer, gif_frames = None, []

    # Trackers must see every frame from 0, rendering starts at --start
    for t, frame in enumerate(seq.frames()):
        if t >= end:
            break
        dets = detections[t] if t % args.stride == 0 else None
        boxes = {kind: tracker.update(dets) for kind, tracker in trackers.items()}
        if t < args.start:
            continue

        if seq.exist[t]:
            draw_box(frame, seq.gt_boxes[t], GT_COLOR)
        for kind, box in boxes.items():
            if box is not None:
                draw_box(frame, box, STYLES[kind][1], dashed=trackers[kind].status == "predicted")
        frame = add_legend_banner(frame, t, args.stride, {k: tr.status for k, tr in trackers.items()})

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        writer.write(frame)
        if args.gif and (t - args.start) % GIF_FRAME_STEP == 0:
            small = cv2.resize(frame, None, fx=GIF_SCALE, fy=GIF_SCALE, interpolation=cv2.INTER_AREA)
            gif_frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))

    writer.release()
    print(f"✅ Video: {out_path}")
    if args.gif:
        gif_frames[0].save(args.gif, save_all=True, append_images=gif_frames[1:], duration=50 * GIF_FRAME_STEP, loop=0,
                           optimize=True)
        print(f"✅ GIF: {args.gif}")
