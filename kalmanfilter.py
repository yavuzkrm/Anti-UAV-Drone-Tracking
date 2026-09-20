import numpy as np


class KalmanFilter:
    def __init__(self, initial_x, initial_y, initial_vx=0, initial_vy=0, ax=0, ay=0, video_width=640):
        self.x = np.array([initial_x, initial_y, initial_vx, initial_vy, ax, ay], dtype=float)
        self.z = np.array([initial_x, initial_y], dtype=float)
        self.video_width = video_width
        
        # Matrisleri oluştur
        self.createMatrix()

    def createMatrix(self):
        # F: Durum Geçiş Matrisi (State Transition Matrix)
        self.F = np.array([
            [1.0, 0.0, 1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ])

        # H: Ölçüm Matrisi (Measurement Matrix)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        ])

        # Q: Süreç Gürültüsü Kovaryans Matrisi (Process Noise Covariance)
        self.Q = np.diag([0.01, 0.01, 0.1, 0.1, 1.0, 1.0])

        # R: Ölçüm Gürültüsü Kovaryans Matrisi (video resolution'a göre)
        if self.video_width == 640:  # infrared
            self.R = np.diag([0.0001, 0.0001])
        else:  # 1920×1080, visible
            self.R = np.diag([0.00001, 0.00001])

        # Identity Matris
        self.I = np.eye(6, dtype=float)

        # P: State Covariance Matrisi (başlangıçta initialize)
        self.P = np.diag([0.0001, 0.0001, 0.04, 0.04, 0.01, 0.01])

    def predict(self):
        """Tahmin aşaması: bir sonraki frame'de drone nerede olacağını tahmin et"""
        self.x_predicted = self.F @ self.x
        self.P_predicted = self.F @ self.P @ np.transpose(self.F) + self.Q

    def update(self, z):
        """Güncelleme aşaması: yeni ölçüm (YOLO bbox) ile tahminimizi düzelt"""
        self.z = z  # Yeni ölçüm [x_yolo, y_yolo]
        
        # Innovation (ölçüm - tahmin)
        y = self.z - self.H @ self.x_predicted
        
        # Innovation covariance
        S = self.H @ self.P_predicted @ self.H.T + self.R
        
        # Kalman gain
        K = self.P_predicted @ self.H.T @ np.linalg.inv(S) # np.transpose(self.H) yerine self.H.T yazılabilir! np.linalg.inv(S) ise tersini alma işlemi yapar
        
        # Güncellenmiş state
        self.x_updated = self.x_predicted + K @ y
        
        # Güncellenmiş covariance
        self.P_updated = (self.I - K @ self.H) @ self.P_predicted
        
        # Bir sonraki iteration için state'i güncelle
        self.x = self.x_updated
        self.P = self.P_updated
        
        return self.x_updated

    def get_position(self):
        """Güncellenmiş drone konumunu döndür"""
        return self.x_updated[:2]  # [x, y]