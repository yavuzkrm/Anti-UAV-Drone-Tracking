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

    track = None
    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()

        detection = model.predict(
            source=frame,
            conf=0.5
        )
        x_pixel, y_pixel, right, bottom = detection.boxes.xywh[0]
        x_norm = x_pixel / width
        y_norm = y_pixel / height

        if frame_count == 0:
            track = KalmanFilter(x_norm, y_norm, video_width=width)
        else:
            track.predict()
            new_z = np.array([x_norm, y_norm])
            track.update(new_z)

        frame_count += 1