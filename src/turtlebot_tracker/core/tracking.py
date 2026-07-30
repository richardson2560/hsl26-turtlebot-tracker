"""
tracking.py - EKF on SE(2) with Non-Locking ZUPT Pseudo-Measurements and FSM Lifecycle.
"""

import numpy as np

from turtlebot_tracker.datatypes import LifecycleState, TrackingState


def wrap_to_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class SE2ManifoldEKF:
    """Extended Kalman Filter on SE(2) manifold with ZUPT and FSM management."""

    def __init__(self, config: dict):
        cfg = config.get("tracking", {})
        fsm_cfg = config.get("fsm", {})

        self.Q = np.diag(np.array([
            cfg.get("q_var_pos", 0.02), cfg.get("q_var_pos", 0.02), cfg.get("q_var_yaw", 0.03),
            0.001, 0.001, 0.001
        ], dtype=np.float64))

        self.R = np.diag(np.array([
            cfg.get("r_var_pos", 0.02), cfg.get("r_var_pos", 0.02), cfg.get("r_var_yaw", 0.04)
        ], dtype=np.float64))

        self.zupt_thresh = cfg.get("zupt_variance_threshold", 0.008)
        self.zupt_window = cfg.get("zupt_window_size", 5)
        self.zupt_v_thresh = cfg.get("zupt_v_est_threshold", 0.03)

        self.max_coasting_time = fsm_cfg.get("max_coasting_time_sec", 3.0)
        self.klost_trigger = fsm_cfg.get("klost_trigger", 2)

        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 0.05

        self.lifecycle_state = LifecycleState.SEARCHING_MAP
        self.is_initialized = False
        self.is_zupt_active = False
        self.coasting_time = 0.0
        self.consecutive_rejected = 0
        self.nis = 0.0

        self.history_pos = []
        self.trajectory_log = []
        self.z_log = []

        self.bearing_compass_mu = 0.0
        self.bearing_compass_kappa = cfg.get("von_mises_default_kappa", 3.0)

        self.z_ground = 0.0
        self.robot_half_height = config.get("tracking", {}).get("z_robot_half_height", 0.24)

    def predict(self, dt: float) -> None:
        if not self.is_initialized or self.lifecycle_state == LifecycleState.SEARCHING_MAP:
            return

        psi = self.x[2]
        vx, vy, omega = self.x[3], self.x[4], self.x[5]

        dx = (vx * np.cos(psi) - vy * np.sin(psi)) * dt
        dy = (vx * np.sin(psi) + vy * np.cos(psi)) * dt
        dpsi = omega * dt

        self.x[0] += dx
        self.x[1] += dy
        self.x[2] = wrap_to_pi(self.x[2] + dpsi)

        F = np.eye(6, dtype=np.float64)
        F[0, 2] = -(vx * np.sin(psi) + vy * np.cos(psi)) * dt
        F[0, 3] = np.cos(psi) * dt
        F[0, 4] = -np.sin(psi) * dt
        F[1, 2] = (vx * np.cos(psi) - vy * np.sin(psi)) * dt
        F[1, 3] = np.sin(psi) * dt
        F[1, 4] = np.cos(psi) * dt
        F[2, 5] = dt

        self.P = F @ self.P @ F.T + self.Q * dt

    def update(self, z_meas: np.ndarray, z_ground: float = 0.0) -> float:
        self.z_ground = z_ground

        if not self.is_initialized:
            self.x[0:2] = z_meas[0:2]
            self.x[2] = wrap_to_pi(z_meas[2])
            self.x[3:6] = 0.0
            self.is_initialized = True
            self.lifecycle_state = LifecycleState.ACTIVE_TRACKING
            self.trajectory_log.append(self.x[:3].copy())
            self.z_log.append(self.z_ground + self.robot_half_height)
            return 0.0

        y = np.zeros(3, dtype=np.float64)
        y[0] = z_meas[0] - self.x[0]
        y[1] = z_meas[1] - self.x[1]
        y[2] = wrap_to_pi(z_meas[2] - self.x[2])

        self.history_pos.append(z_meas[:2].copy())
        if len(self.history_pos) > self.zupt_window:
            self.history_pos.pop(0)

        v_est_norm = np.linalg.norm(self.x[3:5])
        if v_est_norm < self.zupt_v_thresh and len(self.history_pos) == self.zupt_window:
            pos_var = np.sum(np.var(self.history_pos, axis=0))
            if pos_var < self.zupt_thresh:
                self.is_zupt_active = True
                self._apply_zupt_pseudo_measurement()
                self.trajectory_log.append(self.x[:3].copy())
                self.z_log.append(self.z_ground + self.robot_half_height)
                self.nis = 0.0
                return 0.0

        self.is_zupt_active = False

        H = np.zeros((3, 6), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        S = H @ self.P @ H.T + self.R + np.eye(3) * 1e-6
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(3))

        self.x += K @ y
        self.x[2] = wrap_to_pi(self.x[2])
        self.P = (np.eye(6) - K @ H) @ self.P

        self.trajectory_log.append(self.x[:3].copy())
        self.z_log.append(self.z_ground + self.robot_half_height)
        self.nis = float(y.T @ np.linalg.solve(S, y))
        return self.nis

    def _apply_zupt_pseudo_measurement(self) -> None:
        H_v = np.zeros((3, 6), dtype=np.float64)
        H_v[0, 3] = 1.0
        H_v[1, 4] = 1.0
        H_v[2, 5] = 1.0

        R_v = np.eye(3, dtype=np.float64) * 0.001
        S_v = H_v @ self.P @ H_v.T + R_v + np.eye(3) * 1e-6
        K_v = self.P @ H_v.T @ np.linalg.solve(S_v, np.eye(3))

        y_v = -self.x[3:6]
        self.x += K_v @ y_v
        self.P = (np.eye(6) - K_v @ H_v) @ self.P

    def update_lifecycle(self, detection_accepted: bool, dt: float) -> None:
        if self.lifecycle_state == LifecycleState.SEARCHING_MAP:
            if detection_accepted:
                self.lifecycle_state = LifecycleState.ACTIVE_TRACKING
                self.consecutive_rejected = 0
                self.coasting_time = 0.0
            return

        if self.lifecycle_state == LifecycleState.ACTIVE_TRACKING:
            if detection_accepted:
                self.consecutive_rejected = 0
                self.coasting_time = 0.0
            else:
                self.consecutive_rejected += 1
                if self.consecutive_rejected >= self.klost_trigger:
                    self.lifecycle_state = LifecycleState.COASTING_LOST

        elif self.lifecycle_state == LifecycleState.COASTING_LOST:
            if detection_accepted:
                self.lifecycle_state = LifecycleState.ACTIVE_TRACKING
                self.consecutive_rejected = 0
                self.coasting_time = 0.0
            else:
                self.coasting_time += dt
                if self.coasting_time > self.max_coasting_time:
                    self.reset_to_searching()

    def reset_to_searching(self) -> None:
        self.lifecycle_state = LifecycleState.SEARCHING_MAP
        self.is_initialized = False
        self.is_zupt_active = False
        self.coasting_time = 0.0
        self.consecutive_rejected = 0
        self.x.fill(0.0)
        self.P = np.eye(6, dtype=np.float64) * 0.05
        self.history_pos.clear()

    def get_state(self) -> TrackingState:
        z_abs = self.z_ground + self.robot_half_height
        return TrackingState(
            pose_se2=self.x[:3].copy(),
            velocity_se2=self.x[3:].copy(),
            covariance=self.P.copy(),
            z=z_abs,
            lifecycle_state=self.lifecycle_state,
            is_zupt_active=self.is_zupt_active,
            surprise_triggered=(self.lifecycle_state == LifecycleState.COASTING_LOST),
            coasting_time=self.coasting_time,
            bearing_compass_kappa=self.bearing_compass_kappa,
            bearing_compass_mu=self.bearing_compass_mu,
            trajectory_history=list(self.trajectory_log),
            z_log=list(self.z_log),
            nis=self.nis
        )