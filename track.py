import cv2
from ultralytics import YOLO
import numpy as np
from kalmanfilter import KalmanFilter
from datetime import datetime
import json

def kalman_filter_tracking():

    start_time = datetime.now()
    model = YOLO("models/best.pt")

    cap = cv2.VideoCapture('input/videos/test_video.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        'output/videos/kalman_tracked.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    track = None
    frame_count = 0
    frame_with_detection = 0
    frame_without_detection = 0
    success = True
    bbox_w, bbox_h = 0, 0  # Önceki frame'den w, h saklama
    total_x_position = 0
    total_y_position = 0

    while success:
        success, frame = cap.read()
        if not success:
            break

        result = model.predict(source=frame, conf=0.5)
        detection = result[0]

        if len(detection.boxes) > 0:
            bbox = detection.boxes.xywh[0].cpu().numpy()
            x_pixel, y_pixel, w, h = bbox
            x_norm = x_pixel / width
            y_norm = y_pixel / height
            bbox_w, bbox_h = w, h  # Yeni detection ise bbox güncelle
            total_x_position += x_norm
            total_y_position += y_norm
            
            if frame_count == 0:
                track = KalmanFilter(x_norm, y_norm, video_width=width)
            
            track.predict()
            new_z = np.array([x_norm, y_norm])
            track.update(new_z)
            
        elif track is None:
            frame_count += 1
            frame_without_detection += 1
            out.write(frame)
            continue
        else:
            track.predict()

        if track is not None:
            position = track.get_position()
            x_center = position[0] * width
            y_center = position[1] * height
            x1 = int(x_center - bbox_w/2)
            y1 = int(y_center - bbox_h/2)
            x2 = int(x_center + bbox_w/2)
            y2 = int(y_center + bbox_h/2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        out.write(frame)
        frame_count += 1

    frame_with_detection = frame_count - frame_without_detection
    cap.release()
    out.release()
    finish_time = datetime.now()
    processing_time_seconds = round((finish_time - start_time).total_seconds(), 2)
    # float() fonksiyonu ile standart Python sayısına çeviriyoruz
    average_position = [
        round(float(total_x_position / frame_count), 2), 
        round(float(total_y_position / frame_count), 2)
    ]
    data = {
            "algorithm": "Kalman Filter",
            "total_frames": frame_count,
            "frames_with_detection": frame_with_detection,
            "frames_without_detection": frame_without_detection,
            "processing_time_seconds": processing_time_seconds,
            "fps": fps,
            "average_position": average_position
        }
    with open("output/metrics/metrics.json", "w") as file:
        json.dump(data, file, indent=6)

    print("Tracking tamamlandı.")
    print("Output: output/videos/kalman_tracked.mp4")
    print("Output: output/metrics/metrics.json")

if __name__ == "__main__":
    kalman_filter_tracking()