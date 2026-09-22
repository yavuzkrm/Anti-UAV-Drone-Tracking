import numpy as np


class KalmanFilter:
    """
    6D Kalman Filter for drone tracking
    State: [x, y, vx, vy, ax, ay]
    Measurement: [x, y] (from YOLO detection)
    """

    def __init__(self, initial_x, initial_y, initial_vx=0, initial_vy=0, ax=0, ay=0, video_width=640):
        """Initialize Kalman Filter with initial state and video width"""
        self.x = np.array([initial_x, initial_y, initial_vx, initial_vy, ax, ay], dtype=float)
        self.video_width = video_width

        self._create_matrices()

    def _create_matrices(self):
        """Create state transition, measurement, and covariance matrices"""

        # State Transition Matrix (F) - kinematic model
        # x_new = x + vx + 0.5*ax
        # y_new = y + vy + 0.5*ay
        # vx_new = vx + ax
        # vy_new = vy + ay
        # ax_new = ax (constant acceleration model)
        # ay_new = ay
        self.F = np.array([
            [1.0, 0.0, 1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ])

        # Measurement Matrix (H) - we only measure x, y
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        ])

        # Process Noise Covariance (Q)
        # Position and velocity have low uncertainty
        # Acceleration has high uncertainty (can't measure directly)
        self.Q = np.diag([0.01, 0.01, 0.1, 0.1, 1.0, 1.0])

        # Measurement Noise Covariance (R) - depends on video resolution
        if self.video_width == 640:  # Infrared (640x512)
            self.R = np.diag([0.0001, 0.0001])
        else:  # Visible light (1920x1080)
            self.R = np.diag([0.00001, 0.00001])

        # Identity Matrix
        self.I = np.eye(6, dtype=float)

        # State Covariance Matrix (P) - initial uncertainty
        # High uncertainty for velocity and acceleration (unknown initially)
        self.P = np.diag([0.0001, 0.0001, 0.04, 0.04, 0.01, 0.01])

    def predict(self):
        """Prediction step: estimate next state without measurement"""
        self.x_predicted = self.F @ self.x
        self.P_predicted = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """Update step: correct prediction using YOLO measurement"""
        self.z = z  # [x_measured, y_measured]

        # Innovation (residual): difference between measurement and prediction
        innovation = self.z - self.H @ self.x_predicted

        # Innovation covariance
        S = self.H @ self.P_predicted @ self.H.T + self.R

        # Kalman gain: how much to trust measurement vs prediction
        K = self.P_predicted @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x_updated = self.x_predicted + K @ innovation

        # Update state covariance
        self.P_updated = (self.I - K @ self.H) @ self.P_predicted

        # Prepare for next iteration
        self.x = self.x_updated
        self.P = self.P_updated

    def get_position(self):
        """Return current estimated position [x, y]"""
        return self.x_updated[:2]

    def get_velocity(self):
        """Return current estimated velocity [vx, vy]"""
        return self.x_updated[2:4]

    def get_acceleration(self):
        """Return current estimated acceleration [ax, ay]"""
        return self.x_updated[4:6]

    def get_full_state(self):
        """Return full state vector"""
        return self.x_updated
