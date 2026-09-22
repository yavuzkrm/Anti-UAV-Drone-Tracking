import cv2
from ultralytics import YOLO
import numpy as np
from kalmanfilter import KalmanFilter

def kalman_filter_tracking():

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
    success = True
    bbox_w, bbox_h = 0, 0  # Önceki frame'den w, h saklama

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
            
            if frame_count == 0:
                track = KalmanFilter(x_norm, y_norm, video_width=width)
            
            track.predict()
            new_z = np.array([x_norm, y_norm])
            track.update(new_z)
            
        elif track is None:
            frame_count += 1
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

    cap.release()
    out.release()
    print(f"Tracking tamamlandı. Output: output/videos/kalman_tracked.mp4")

if __name__ == "__main__":
    kalman_filter_tracking()