"""
6D constant-acceleration Kalman Filter for the drone's center point.

State:       x = [cx, cy, vx, vy, ax, ay]   (pixels, px/frame, px/frame^2)
Measurement: z = [cx, cy]                   (center of the detected box)
"""

import numpy as np


class KalmanFilter:

    def __init__(self, cx, cy, q_scale=1.0, r_std=2.0, accel=True):
        """
        q_scale - multiplies the process noise Q. Larger = trust the motion model less
                  (follows sharp turns faster, but the box is noisier).
        r_std   - expected detector center error in pixels (sets R).
        accel   - False turns the model into constant velocity: acceleration is
                  fixed at 0 (same idea as ByteTrack's filter).
        """
        self.x = np.array([cx, cy, 0.0, 0.0, 0.0, 0.0])

        # State transition (F): one frame forward with constant acceleration
        #   cx_new = cx + vx + 0.5*ax,   vx_new = vx + ax,   ax_new = ax
        self.F = np.array([
            [1.0, 0.0, 1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])

        # Measurement matrix (H): the detector only gives the position
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ])

        # Process noise (Q): how wrong the motion model can be in one frame
        self.Q = q_scale * np.diag([1.0, 1.0, 0.5, 0.5, 0.1, 0.1])

        # Measurement noise (R): detector center error, r_std pixels standard deviation
        self.R = np.diag([r_std ** 2, r_std ** 2])

        # Initial uncertainty (P): position comes from the first detection,
        # velocity and acceleration are unknown
        self.P = np.diag([r_std ** 2, r_std ** 2, 100.0, 100.0, 10.0, 10.0])

        if not accel:
            # Remove every acceleration term: it stays 0 and never affects position/velocity
            self.F[:4, 4:] = 0.0
            self.Q[4:, 4:] = 0.0
            self.P[4:, 4:] = 0.0

        self.I = np.eye(6)

    def predict(self):
        """Move the state one frame forward with the motion model."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """Correct the predicted state with a measured center z = [cx, cy]."""
        innovation = np.asarray(z) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain: measurement vs prediction trust

        self.x = self.x + K @ innovation
        self.P = (self.I - K @ self.H) @ self.P

    def mahalanobis(self, z):
        """
        Squared distance between measurement(s) z (N, 2) and the predicted position,
        scaled by the current uncertainty S. Used to reject detections that are too
        far from where the drone should be (gating).
        """
        S = self.H @ self.P @ self.H.T + self.R
        d = np.atleast_2d(z) - self.H @ self.x
        return np.einsum("ni,ij,nj->n", d, np.linalg.inv(S), d)

    @property
    def position(self):
        return self.x[:2]

    @property
    def velocity(self):
        return self.x[2:4]
