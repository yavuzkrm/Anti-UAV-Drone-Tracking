import cv2
from ultralytics import YOLO
import numpy as np
from kalmanfilter import KalmanFilter
from datetime import datetime
import json
import os


# Configuration
CONF_THRESHOLD = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (0, 255, 0)  # Green (BGR)
FONT_THICKNESS = 1
BBOX_COLOR = (0, 255, 0)  # Green (BGR)
BBOX_THICKNESS = 2


def ensure_output_dirs():
    """Create output directories if they don't exist"""
    os.makedirs("output/videos", exist_ok=True)
    os.makedirs("output/metrics", exist_ok=True)


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    box1, box2 = (x1, y1, x2, y2) in pixel coordinates
    Returns: IoU value (0-1)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # No intersection
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def get_bbox_from_detection(detection, width, height):
    """
    Extract bounding box from YOLO detection and normalize coordinates
    Returns: x_norm, y_norm, w, h (all normalized to 0-1)
    """
    try:
        bbox = detection.boxes.xywh[0].cpu().numpy()
        x_pixel, y_pixel, w, h = bbox

        x_norm = x_pixel / width
        y_norm = y_pixel / height

        return x_norm, y_norm, w, h
    except Exception as e:
        print(f"Error extracting bbox: {e}")
        return None




def draw_drone_info(frame, position, velocity, acceleration, bbox_w, bbox_h, width, height, iou=None):
    """Draw drone position, velocity, acceleration, and IoU on frame"""
    x_center = position[0] * width
    y_center = position[1] * height

    # Convert to pixel coordinates
    x1 = int(x_center - bbox_w/2)
    y1 = int(y_center - bbox_h/2)
    x2 = int(x_center + bbox_w/2)
    y2 = int(y_center + bbox_h/2)

    # Draw predicted bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)

    # Draw crosshair at center
    cv2.circle(frame, (int(x_center), int(y_center)), 3, BBOX_COLOR, -1)

    # Display velocity and acceleration
    y_offset = 30
    cv2.putText(frame, f"Vx: {velocity[0]:.4f}", (10, y_offset),
                FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
    cv2.putText(frame, f"Vy: {velocity[1]:.4f}", (10, y_offset + 25),
                FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
    cv2.putText(frame, f"Ax: {acceleration[0]:.4f}", (10, y_offset + 50),
                FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
    cv2.putText(frame, f"Ay: {acceleration[1]:.4f}", (10, y_offset + 75),
                FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)

    # Display position
    cv2.putText(frame, f"X: {position[0]:.3f} Y: {position[1]:.3f}",
                (10, y_offset + 100), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)

    # Display IoU if available
    if iou is not None:
        cv2.putText(frame, f"IoU: {iou:.3f}", (10, y_offset + 125),
                    FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)


def kalman_filter_tracking(input_video='input/videos/test_video.mp4'):
    """
    Main tracking function using Kalman Filter with IoU evaluation
    """
    start_time = datetime.now()

    # Ensure output directories exist
    ensure_output_dirs()

    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO("models/best.pt").to("cuda")

    # Open video
    print(f"Opening video: {input_video}")
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height} @ {fps} FPS, Total frames: {total_frames}")

    # Setup video writer
    output_path = 'output/videos/kalman_tracked.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize tracking variables
    tracker = None
    frame_count = 0
    frames_with_detection = 0
    frames_without_detection = 0
    bbox_w, bbox_h = 0, 0

    # Metrics accumulation
    total_x_position = 0
    total_y_position = 0
    total_vx = 0
    total_vy = 0
    total_ax = 0
    total_ay = 0
    total_iou = 0
    iou_count = 0
    tracked_frames = 0

    print("Starting tracking...")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # YOLO detection
        results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
        detection = results[0]

        # Detection available
        if len(detection.boxes) > 0:
            bbox_data = get_bbox_from_detection(detection, width, height)

            if bbox_data is not None:
                x_norm, y_norm, w, h = bbox_data
                bbox_w, bbox_h = w, h

                # Initialize tracker on first detection
                if tracker is None:
                    print(f"Tracker initialized at frame {frame_count}")
                    tracker = KalmanFilter(x_norm, y_norm, video_width=width)

                # Predict and update
                tracker.predict()
                new_z = np.array([x_norm, y_norm])
                tracker.update(new_z)

                frames_with_detection += 1

        # No detection but tracker exists
        elif tracker is not None:
            tracker.predict()
            frames_without_detection += 1

        # Skip frame if tracker not initialized yet
        else:
            out.write(frame)
            frame_count += 1
            continue

        # Draw tracking info on frame
        if tracker is not None:
            position = tracker.get_position()
            velocity = tracker.get_velocity()
            acceleration = tracker.get_acceleration()

            # Calculate IoU: compare YOLO detection with Kalman prediction
            iou = None
            if len(detection.boxes) > 0 and bbox_data is not None:
                # YOLO measured position
                yolo_x_norm, yolo_y_norm, yolo_w, yolo_h = bbox_data
                yolo_x_center = yolo_x_norm * width
                yolo_y_center = yolo_y_norm * height
                yolo_x1 = int(yolo_x_center - yolo_w/2)
                yolo_y1 = int(yolo_y_center - yolo_h/2)
                yolo_box = (yolo_x1, yolo_y1, int(yolo_x1 + yolo_w), int(yolo_y1 + yolo_h))

                # Kalman predicted position
                pred_x_center = position[0] * width
                pred_y_center = position[1] * height
                pred_x1 = int(pred_x_center - bbox_w/2)
                pred_y1 = int(pred_y_center - bbox_h/2)
                pred_box = (pred_x1, pred_y1, int(pred_x1 + bbox_w), int(pred_y1 + bbox_h))

                iou = calculate_iou(yolo_box, pred_box)
                total_iou += iou
                iou_count += 1

            # Draw on frame
            draw_drone_info(frame, position, velocity, acceleration,
                          bbox_w, bbox_h, width, height, iou)

            # Accumulate metrics
            total_x_position += position[0]
            total_y_position += position[1]
            total_vx += velocity[0]
            total_vy += velocity[1]
            total_ax += acceleration[0]
            total_ay += acceleration[1]
            tracked_frames += 1

        # Write frame to output video
        out.write(frame)
        frame_count += 1

        # Progress indicator
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    # Cleanup
    cap.release()
    out.release()

    # Calculate metrics
    finish_time = datetime.now()
    processing_time = (finish_time - start_time).total_seconds()

    metrics = {
        "algorithm": "Kalman Filter",
        "total_frames": frame_count,
        "frames_with_detection": frames_with_detection,
        "frames_without_detection": frames_without_detection,
        "frames_tracked": tracked_frames,
        "processing_time_seconds": round(processing_time, 2),
        "fps": round(fps, 2),
        "processing_fps": round(frame_count / processing_time, 2) if processing_time > 0 else 0,
    }

    # Average metrics (only when tracker existed)
    if tracked_frames > 0:
        metrics.update({
            "average_position_x": round(total_x_position / tracked_frames, 4),
            "average_position_y": round(total_y_position / tracked_frames, 4),
            "average_velocity_x": round(total_vx / tracked_frames, 4),
            "average_velocity_y": round(total_vy / tracked_frames, 4),
            "average_acceleration_x": round(total_ax / tracked_frames, 4),
            "average_acceleration_y": round(total_ay / tracked_frames, 4),
        })

    # IoU metric
    if iou_count > 0:
        metrics["average_iou"] = round(total_iou / iou_count, 4)
        metrics["iou_frames"] = iou_count

    # Save metrics
    metrics_path = 'output/metrics/metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    # Print summary
    print("\n" + "="*50)
    print("Tracking Completed!")
    print("="*50)
    print(f"Output Video: {output_path}")
    print(f"Metrics File: {metrics_path}")
    print(f"Total Frames: {frame_count}")
    print(f"Frames with Detection: {frames_with_detection}")
    print(f"Frames without Detection (predicted): {frames_without_detection}")
    print(f"Processing Time: {processing_time:.2f}s")
    print(f"Processing Speed: {frame_count / processing_time:.2f} FPS")
    if iou_count > 0:
        print(f"Average IoU: {metrics.get('average_iou', 'N/A')}")
        print(f"IoU Frames: {iou_count}")
    print("="*50 + "\n")


if __name__ == "__main__":
    kalman_filter_tracking()
