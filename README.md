# Anti-UAV Drone Tracking

Real-time drone detection and tracking system using YOLO and Kalman Filter for the Anti-UAV dataset.

## Overview

This project implements a robust drone tracking pipeline that combines:
- **YOLO Detection**: State-of-the-art object detection (87.5% mAP50 on Anti-UAV dataset)
- **Kalman Filter**: 6D motion model for smooth tracking and prediction through detection gaps

## Features

- ✅ Real-time drone detection with YOLOv8
- ✅ Kalman Filter tracking with constant acceleration model
- ✅ Position, velocity, and acceleration estimation
- ✅ IoU (Intersection over Union) evaluation metrics
- ✅ Processing speed: ~22 FPS (real-time capable on GPU)
- ✅ Comprehensive metrics logging (JSON output)

## Project Structure

```
Anti-UAV-Drone-Tracking/
├── src/
│   ├── kalmanfilter.py      # 6D Kalman Filter implementation
│   └── track.py              # Main tracking script with IoU evaluation
├── models/
│   └── best.pt               # Trained YOLO model (download required)
├── datasets/
│   └── videos/               # Input video files
├── output/
│   ├── videos/               # Tracked video output
│   └── metrics/              # JSON metrics files
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

### Prerequisites
- Python 3.8+
- GPU (recommended): NVIDIA GPU with CUDA support
- CUDA 11.8+ (for GPU acceleration)
- cuDNN 8.0+ (for GPU acceleration)

### Setup Steps

1. **Clone repository:**
```bash
git clone https://github.com/yavuzkrm/Anti-UAV-Drone-Tracking
cd Anti-UAV-Drone-Tracking
```

2. **Install dependencies:**

**For GPU (recommended for real-time performance):**
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**For CPU only:**
```bash
pip install -r requirements.txt
```

3. **Download trained YOLO model:**
```bash
# Place best.pt in models/ directory
# Trained model with 87.5% mAP50 on Anti-UAV dataset
# File size: ~100MB
```

4. **Prepare input videos:**
```bash
# Place video files in datasets/videos/
```

## Usage

### Basic Tracking
```bash
cd src
python track.py
```

This will:
1. Load the trained YOLO model
2. Process input video and detect drones
3. Track with Kalman Filter
4. Output tracked video to `output/videos/kalman_tracked.mp4`
5. Save metrics to `output/metrics/metrics.json`

### Custom Video Path
Edit `track.py` and modify:
```python
if __name__ == "__main__":
    kalman_filter_tracking(input_video='path/to/your/video.mp4')
```

## Kalman Filter Details

### State Vector (6D)
```
x = [x_pos, y_pos, vx, vy, ax, ay]
```

### Motion Model
- Position: `x_new = x + vx + 0.5*ax`
- Velocity: `vx_new = vx + ax`
- Acceleration: `ax_new = ax` (constant acceleration model)

### Measurement
Only position is measured from YOLO:
```
z = [x_measured, y_measured]
```

### Noise Covariance
- **Process Noise (Q)**: `diag([0.01, 0.01, 0.1, 0.1, 1.0, 1.0])`
- **Measurement Noise (R)**: Resolution-dependent
  - 640×512 (IR): `diag([0.0001, 0.0001])`
  - 1920×1080 (Visible): `diag([0.00001, 0.00001])`

## Output Metrics

JSON file contains:
- `total_frames`: Total frames processed
- `frames_with_detection`: Frames with drone detected
- `frames_without_detection`: Frames with only prediction
- `frames_tracked`: Frames with active tracker
- `processing_fps`: Actual processing speed
- `average_position_x/y`: Mean normalized position
- `average_velocity_x/y`: Mean velocity estimation
- `average_acceleration_x/y`: Mean acceleration estimation
- `average_iou`: Mean IoU (YOLO detection vs Kalman prediction)

### Example Output
```json
{
    "algorithm": "Kalman Filter",
    "total_frames": 1000,
    "frames_with_detection": 1000,
    "frames_without_detection": 0,
    "processing_fps": 21.94,
    "average_iou": 1.0,
    "iou_frames": 1000
}
```

## Performance

### GPU (NVIDIA RTX 3060 or similar)
| Metric | Value |
|--------|-------|
| Detection mAP50 | 87.5% |
| Processing Speed | ~22 FPS |
| Real-time Capable | ✅ Yes |
| Average IoU | 1.0 |
| Memory Usage | ~4GB VRAM |

### CPU (Intel i7/Ryzen 7)
| Metric | Value |
|--------|-------|
| Detection mAP50 | 87.5% |
| Processing Speed | ~2-3 FPS |
| Real-time Capable | ❌ No |
| Memory Usage | ~8GB RAM |

**Recommendation**: Use GPU for real-time tracking. Processing speed is 6-8x faster on GPU.

## Future Work

- [ ] ByteTrack implementation for comparison
- [ ] Multi-drone tracking support
- [ ] Extended Kalman Filter (EKF) variant
- [ ] Real-time visualization during processing
- [ ] Web interface for results

## Dataset

Uses the **Anti-UAV 200** dataset:
- 200 video sequences
- Infrared (640×512) and visible (1920×1080) streams
- Frame-by-frame ground truth annotations

## References

- YOLO: [YOLOv8 Documentation](https://docs.ultralytics.com/)
- Kalman Filter: Welch & Bishop, 2006 - "An Introduction to the Kalman Filter"
- Anti-UAV Dataset: [Anti-UAV Challenge](https://anti-uav.github.io/)


## Authors

- Yavuz K. - Computer Engineering Student
- Claude Haiku 4.5 - Code assistance

---

**Last Updated**: September 22, 2026

For questions or issues, open a GitHub issue.
