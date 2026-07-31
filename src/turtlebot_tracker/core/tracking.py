"""
tracking.py - EKF on SE(2) with aggressive snap-on-target and coasting.
"""

import time
import numpy as np
from turtlebot_tracker.datatypes import LifecycleState, TrackingState

def wrap_to_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))

class SE2ManifoldEKF:
    def __init__(self, config: dict):
        cfg = config.get("tracking", {})

        self.R_target = np.diag([
            cfg.get("r_var_pos", 0.01),
            cfg.get("r_var_pos", 0.01),
            cfg.get("r_var_yaw", 0.02)
        ])
        self.R_low_conf = self.R_target * 5.0

        self.Q = np.diag([
            cfg.get("q_var_pos", 0.02),
            cfg.get("q_var_pos", 0.02),
            cfg.get("q_var_yaw", 0.03),
            0.005, 0.005, 0.005
        ])

        # State: [x, y, psi, vx, vy, omega]
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 0.1

        # Health / Reliability (0-100)
        self.health = 0
        self.init_threshold = 20          # Lower threshold for faster lock
        self.is_initialized = False
        self.lifecycle_state = LifecycleState.SEARCHING_MAP

        # Coasting & ZUPT
        self.is_zupt_active = False
        self._last_target_time = 0.0
        self._timeout_duration = 2.0      # seconds before full reset

        # Persistent trajectory (breadcrumbs) – only cleared on full reset
        self.trajectory_log = []
        self.z_log = []
        self.z_ground = 0.0
        self.robot_half_height = cfg.get("z_robot_half_height", 0.24)

        # For velocity initialization
        self._prev_measurement = None
        self._frame_count = 0

    def predict(self, dt: float):
        """Prediction step (always called when initialized)."""
        if not self.is_initialized:
            return

        psi = self.x[2]
        vx, vy, omega = self.x[3], self.x[4], self.x[5]

        self.x[0] += (vx * np.cos(psi) - vy * np.sin(psi)) * dt
        self.x[1] += (vx * np.sin(psi) + vy * np.cos(psi)) * dt
        self.x[2] = wrap_to_pi(self.x[2] + omega * dt)

        F = np.eye(6)
        F[0, 2] = -(vx * np.sin(psi) + vy * np.cos(psi)) * dt
        F[0, 3] = np.cos(psi) * dt
        F[0, 4] = -np.sin(psi) * dt
        F[1, 2] = (vx * np.cos(psi) - vy * np.sin(psi)) * dt
        F[1, 3] = np.sin(psi) * dt
        F[1, 4] = np.cos(psi) * dt
        F[2, 5] = dt

        self.P = F @ self.P @ F.T + self.Q * dt

    def update(self, z_meas, is_target=False, score_val=0.0, dt=0.1, timestamp=None):
        """
        Update the EKF with a measurement (or None for coasting).
        Returns True if the state was updated, False otherwise.
        """
        self._frame_count += 1
        now = timestamp if timestamp is not None else time.time()

        # ---- No measurement: coasting ----
        if z_meas is None:
            if self.is_initialized:
                self.health = max(0, self.health - 5)
                self.lifecycle_state = LifecycleState.COASTING_LOST
                if now - self._last_target_time > self._timeout_duration:
                    self.reset()
            return False

        # ---- Measurement available ----
        # KISS rule: if it's a TARGET, we trust it and snap (even if far)
        if is_target:
            # Snap: directly set state to measurement
            self.x[:3] = z_meas
            self.x[3:] = 0.0
            self.health = min(100, self.health + 25)
            self._last_target_time = now
            self.lifecycle_state = LifecycleState.ACTIVE_TRACKING
            self.is_initialized = True
            # Log trajectory
            self.trajectory_log.append(self.x[:3].copy())
            self.z_log.append(self.z_ground + self.robot_half_height)
            return True

        # ---- Non-target (Best-Rejected) measurement ----
        # Only apply if it's close enough to the prediction (Mahalanobis gate)
        if self.is_initialized:
            y = z_meas - self.x[:3]
            y[2] = wrap_to_pi(y[2])
            H = np.zeros((3, 6))
            H[:3, :3] = np.eye(3)
            S = H @ self.P @ H.T + self.R_low_conf
            try:
                inv_S = np.linalg.inv(S)
                mahal_dist = y.T @ inv_S @ y
                if mahal_dist > 9.21:   # 99% chi2 for df=3
                    self.health = max(0, self.health - 5)
                    return False
            except np.linalg.LinAlgError:
                pass

        # Update health
        self.health = min(100, self.health + 5)

        # Standard Kalman update (only for Best-Rejected or if not initialized)
        if not self.is_initialized:
            if self.health > self.init_threshold:
                self._initialize_state(z_meas)
                return True
            return False

        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        y = z_meas - self.x[:3]
        y[2] = wrap_to_pi(y[2])

        # Use low-confidence R for Best-Rejected
        S = H @ self.P @ H.T + self.R_low_conf
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        self.x[2] = wrap_to_pi(self.x[2])
        self.P = (np.eye(6) - K @ H) @ self.P

        # Log trajectory
        self.trajectory_log.append(self.x[:3].copy())
        self.z_log.append(self.z_ground + self.robot_half_height)

        # ZUPT (only if stationary and recently seen)
        self._apply_zupt()
        return True

    def _initialize_state(self, z_meas):
        self.x[:3] = z_meas
        self.x[3:] = 0.0
        self.is_initialized = True
        self.lifecycle_state = LifecycleState.ACTIVE_TRACKING
        self._last_target_time = time.time()
        self.trajectory_log.append(self.x[:3].copy())
        self.z_log.append(self.z_ground + self.robot_half_height)

    def _apply_zupt(self):
        vel_norm = np.linalg.norm(self.x[3:5])
        if vel_norm < 0.03 and (time.time() - self._last_target_time) < 1.0:
            H_v = np.zeros((3, 6))
            H_v[:, 3:] = np.eye(3)
            R_v = np.eye(3) * 0.001
            S_v = H_v @ self.P @ H_v.T + R_v
            K_v = self.P @ H_v.T @ np.linalg.inv(S_v)
            self.x += K_v @ (0.0 - self.x[3:])
            self.P = (np.eye(6) - K_v @ H_v) @ self.P
            self.is_zupt_active = True
        else:
            self.is_zupt_active = False

    def reset(self):
        """Full reset: clears state, trajectory, and returns to SEARCHING."""
        self.is_initialized = False
        self.lifecycle_state = LifecycleState.SEARCHING_MAP
        self.health = 0
        self.x.fill(0)
        self.P = np.eye(6) * 0.1
        self.trajectory_log = []
        self.z_log = []
        self._prev_measurement = None
        self._last_target_time = 0.0

    def get_state(self) -> TrackingState:
        return TrackingState(
            pose_se2=self.x[:3].copy(),
            velocity_se2=self.x[3:].copy(),
            covariance=self.P.copy(),
            z=self.z_ground + self.robot_half_height,
            lifecycle_state=self.lifecycle_state,
            is_zupt_active=self.is_zupt_active,
            surprise_triggered=(self.lifecycle_state == LifecycleState.COASTING_LOST),
            coasting_time=0.0,
            bearing_compass_kappa=0.0,
            bearing_compass_mu=0.0,
            trajectory_history=list(self.trajectory_log),
            z_log=list(self.z_log),
            nis=0.0,
            reliability_score=self.health   # Expose health as reliability_score
        )