# Results on the test split (15 sequences, 14348 frames)

All numbers are means over sequences. Parameters were tuned on the val split (`configs/trackers.yaml`).

## Main comparison

| Method | State acc. ↑ | Success AUC ↑ | Precision@20px ↑ | False alarm ↓ |
|---|---:|---:|---:|---:|
| Detector only | 0.563 | 0.545 | 0.712 | 0.345 |
| Kalman (ours) | 0.563 | 0.554 | 0.727 | 0.501 |
| ByteTrack | 0.516 | 0.495 | 0.667 | 0.284 |

Detector (YOLO, GPU): 13.3 ms/frame. Tracker overhead on top of it:

| Method | Tracker ms/frame |
|---|---:|
| Detector only | 0.005 |
| Kalman (ours) | 0.044 |
| ByteTrack | 0.162 |

## Frames where the detector misses the drone

Drone visible, but no detection above the baseline's threshold.

| Method | Frames | Precision@20px | IoU > 0.5 |
|---|---:|---:|---:|
| Detector only | 1494 | 0.000 | 0.000 |
| Kalman (ours) | 1494 | 0.114 | 0.078 |
| ByteTrack | 1494 | 0.067 | 0.055 |

## Ablations

Tuned parameters with one component switched off.

| Method | State acc. ↑ | Success AUC ↑ | Precision@20px ↑ | False alarm ↓ |
|---|---:|---:|---:|---:|
| Kalman (ours) | 0.563 | 0.554 | 0.727 | 0.501 |
| ByteTrack | 0.516 | 0.495 | 0.667 | 0.284 |
| Kalman, constant velocity instead of acceleration | 0.563 | 0.555 | 0.728 | 0.501 |
| Kalman, no coasting (max_missed = 0) | 0.563 | 0.545 | 0.714 | 0.345 |
| ByteTrack, no low-score 2nd round (BYTE off) | 0.515 | 0.493 | 0.665 | 0.284 |
| ByteTrack, no coasting (coast = 0) | 0.508 | 0.485 | 0.653 | 0.133 |

## Running the detector on every k-th frame

State accuracy when YOLO only runs on every k-th frame and the tracker bridges the rest (parameters tuned on val separately for each k).

| Detector stride | Detector ms/frame | Detector only | Kalman (ours) | ByteTrack |
|---:|---:|---:|---:|---:|
| 1 | 13.3 | 0.563 | 0.563 | 0.516 |
| 2 | 6.7 | 0.521 | 0.534 | 0.444 |
| 3 | 4.4 | 0.486 | 0.495 | 0.368 |
| 5 | 2.7 | 0.431 | 0.427 | 0.318 |
