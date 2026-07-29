"""
tracking.py - Extended Kalman Filter on SE(2) with Zero Velocity Update (ZUPT) and NIS.
"""

import numpy as np
from scipy.linalg import block_diag
from ..datatypes import TrackingState

class SE2ManifoldEKF:
    def __init__(self, config: dict):
        cfg = config["tracking"]
        self.Q = np.diag([cfg["q_var_pos"], cfg["q_var_pos"], cfg["q_var_yaw"],
                          0.001, 0.001, 0.001])
        self.R = np.diag([cfg["r_var_pos"], cfg["r_var_pos"], cfg["r_var_yaw"]])
        self.zupt_thresh = cfg["zupt_variance_threshold"]
        self.zupt_window = cfg["zupt_window_size"]

        self.x = np.zeros(6)          # [x, y, psi, vx, vy, omega]
        self.P = np.eye(6) * 0.1
        self.history_pos = []
        self.trajectory_log = []
        self.is_initialized = False
        self.is_zupt_active = False
        self.nis = 0.0
        # Compass (bearing) state
        self.bearing_compass_mu = 0.0
        self.bearing_compass_kappa = 1.0

    def predict(self, dt: float):
        if not self.is_initialized:
            return

        if self.is_zupt_active:
            self.x[3:] = 0.0
            # Reduce velocity covariance
            self.P[3:6, 3:6] *= 0.1
            return

        psi = self.x[2]
        vx, vy, omega = self.x[3], self.x[4], self.x[5]

        dx = (vx * np.cos(psi) - vy * np.sin(psi)) * dt
        dy = (vx * np.sin(psi) + vy * np.cos(psi)) * dt
        dpsi = omega * dt

        self.x[0] += dx
        self.x[1] += dy
        self.x[2] = np.arctan2(np.sin(self.x[2] + dpsi), np.cos(self.x[2] + dpsi))

        F = np.eye(6)
        F[0, 2] = (-vx * np.sin(psi) - vy * np.cos(psi)) * dt
        F[0, 3] = np.cos(psi) * dt
        F[0, 4] = -np.sin(psi) * dt
        F[1, 2] = (vx * np.cos(psi) - vy * np.sin(psi)) * dt
        F[1, 3] = np.sin(psi) * dt
        F[1, 4] = np.cos(psi) * dt
        F[2, 5] = dt

        self.P = F @ self.P @ F.T + self.Q * dt

    def update(self, z_meas: np.ndarray) -> float:
        if not self.is_initialized:
            self.x[0] = z_meas[0]
            self.x[1] = z_meas[1]
            self.x[2] = z_meas[2]
            self.is_initialized = True
            self.trajectory_log.append(self.x[:3].copy())
            return 0.0

        y = np.zeros(3)
        y[0] = z_meas[0] - self.x[0]
        y[1] = z_meas[1] - self.x[1]
        y[2] = np.arctan2(np.sin(z_meas[2] - self.x[2]), np.cos(z_meas[2] - self.x[2]))

        # ZUPT check
        self.history_pos.append(z_meas[:2].copy())
        if len(self.history_pos) > self.zupt_window:
            self.history_pos.pop(0)

        if len(self.history_pos) == self.zupt_window:
            pos_var = np.sum(np.var(self.history_pos, axis=0))
            if pos_var < self.zupt_thresh:
                self.is_zupt_active = True
                self.x[3:] = 0.0
                mean_pos = np.mean(self.history_pos, axis=0)
                self.x[0] = mean_pos[0]
                self.x[1] = mean_pos[1]
                self.trajectory_log.append(self.x[:3].copy())
                self.nis = 0.0
                return 0.0

        self.is_zupt_active = False

        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))
        self.P = (np.eye(6) - K @ H) @ self.P

        self.trajectory_log.append(self.x[:3].copy())
        self.nis = float(y.T @ np.linalg.inv(S) @ y)
        return self.nis

    def get_state(self) -> TrackingState:
        return TrackingState(
            pose_se2=self.x[:3].copy(),
            velocity_se2=self.x[3:].copy(),
            covariance=self.P.copy(),
            is_zupt_active=self.is_zupt_active,
            surprise_triggered=False,
            bearing_compass_kappa=self.bearing_compass_kappa,
            trajectory_history=list(self.trajectory_log),
            nis=self.nis
        )