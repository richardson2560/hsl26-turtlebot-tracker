"""
registration.py - Direct GMM MAP Registrator with VMF, Range-Aware Inflation,
Multi-start, and fixed-volume Wilks GLRT.
Also includes GPIS-W registrator.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import chi2
from scipy.interpolate import interp1d

from turtlebot_tracker.datatypes import ClusterCandidate, FrameData, TrackingState
from turtlebot_tracker.core.implicit_surface import load_model, ImplicitSurfaceModel


def wrap_to_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def evaluate_vmf_likelihood(view_vec, mu_dir, kappa):
    if kappa < 0.2:
        return 1.0 / (4.0 * np.pi)
    dot = np.clip(np.dot(view_vec, mu_dir), -1.0, 1.0)
    c_kappa = kappa / (4.0 * np.pi * np.sinh(np.clip(kappa, 0, 50)) + 1e-7)
    return max(float(c_kappa * np.exp(kappa * dot)), 1e-6)

# ============================================================================
#  GPIS-W REGISTRATOR (OPTIMIZADO CON SMART MULTI-START SCREENING)
# ============================================================================

class GPISRegistrator:
    """Registrator using Hermite-GPIS-W implicit surface model with Smart Screening."""

    def __init__(self, config: dict, model_path: str = "config/implicit_model.json"):
        cfg = config.get("registration", {})
        self.score_threshold = cfg.get("score_threshold", 2.0)
        self.max_iter = cfg.get("max_gn_iterations", 4)  # 4 iteraciones suficientes gracias al buen arranque
        self.n_starts = cfg.get("n_starts", 8)
        self.sigma_r = cfg.get("sigma_lidar", 0.012)
        self.model = load_model(model_path)
        print(f"[GPIS] Model loaded: {self.model.M} primitives, centroid={self.model.centroid}")

    def register_and_track(self, frame_data, candidates, ekf_tracker):
        passed = [c for c in candidates if c.passed_filters]
        if not passed:
            ekf_tracker.update_lifecycle(False, dt=0.1)
            state = ekf_tracker.get_state()
            state.surprise_triggered = True
            return state, None

        pred_pose = ekf_tracker.get_state().pose_se2
        is_initialized = ekf_tracker.is_initialized

        init_pose = None
        if is_initialized:
            psi = pred_pose[2]
            R0 = np.array([[np.cos(psi), -np.sin(psi), 0.],
                           [np.sin(psi),  np.cos(psi), 0.],
                           [0.,           0.,          1.]], dtype=np.float64)
            t0 = np.array([pred_pose[0], pred_pose[1], self.model.centroid[2]], dtype=np.float64)
            init_pose = (R0, t0)

        best_cand = None
        best_score = np.inf
        best_R = np.eye(3)
        best_t = np.zeros(3)

        for cand in passed:
            score, R, t, ll = self._fit_gpis_se2(cand.points, init_pose)
            if score < best_score:
                best_score = score
                best_cand = cand
                best_R = R
                best_t = t

        if best_cand is None or best_score > self.score_threshold:
            ekf_tracker.update_lifecycle(False, dt=0.1)
            state = ekf_tracker.get_state()
            state.surprise_triggered = True
            return state, None

        yaw = wrap_to_pi(np.arctan2(best_R[1, 0], best_R[0, 0]))
        z_meas = np.array([best_t[0], best_t[1], yaw])
        ekf_tracker.update_lifecycle(True, dt=0.1)
        ekf_tracker.update(z_meas)

        state = ekf_tracker.get_state()
        state.surprise_triggered = False
        return state, best_cand

    def _fit_gpis_se2(self, points, init_pose=None):
        if init_pose is not None:
            return self._run_gauss_newton(points, init_pose[0], init_pose[1], max_iter=self.max_iter)

        cand_center = np.mean(points, axis=0)
        angles = np.linspace(0, 2 * np.pi, self.n_starts, endpoint=False)

        # 1. SMART SCREENING: Evaluar residuo inicial E0 de los 8 ángulos en 0.2ms
        init_scores = []
        for yaw0 in angles:
            R0 = np.array([[np.cos(yaw0), -np.sin(yaw0), 0.],
                           [np.sin(yaw0),  np.cos(yaw0), 0.],
                           [0.,            0.,          1.]], dtype=np.float64)
            t0 = np.array([cand_center[0], cand_center[1], self.model.centroid[2]], dtype=np.float64)
            pts_trans = (R0.T @ (points - t0).T).T
            f_vals, _, _ = self.model.evaluate(pts_trans, compute_var=False)
            score_init = float(np.mean(f_vals**2 / (self.sigma_r ** 2)))
            init_scores.append((score_init, R0, t0))

        # Ordenar semillas por residuo inicial y tomar solo los 2 mejores ángulos
        init_scores.sort(key=lambda x: x[0])
        top_seeds = init_scores[:2]

        best_score = np.inf
        best_R = np.eye(3)
        best_t = np.zeros(3)
        best_ll = -999.0

        # 2. Ejecutar Gauss-Newton SOLO en los 2 mejores ángulos
        for _, R0, t0 in top_seeds:
            score, R, t, ll = self._run_gauss_newton(points, R0, t0, max_iter=3)  # 3 iteraciones son suficientes
            if score < best_score:
                best_score = score
                best_R = R
                best_t = t
                best_ll = ll

        return best_score, best_R, best_t, best_ll

    def _run_gauss_newton(self, points, R0, t0, max_iter=4):
        model = self.model
        R = R0.copy()
        yaw = np.arctan2(R[1, 0], R[0, 0])
        x, y = t0[0], t0[1]
        sigma_r2 = self.sigma_r ** 2

        for _ in range(max_iter):
            c, s = np.cos(yaw), np.sin(yaw)
            R_curr = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], dtype=np.float64)
            t_curr = np.array([x, y, model.centroid[2]], dtype=np.float64)

            # Transformación a coordenadas locales del modelo
            pts_trans = (R_curr.T @ (points - t_curr).T).T

            f_vals, grad_f, _ = model.evaluate(pts_trans, compute_var=False)

            abs_f = np.abs(f_vals)
            w_huber = np.where(abs_f < 0.03, 1.0, 0.03 / (abs_f + 1e-6))
            w = w_huber / sigma_r2

            # Jacobiano exacto d(f)/d(x, y, yaw)
            J = np.zeros((len(points), 3), dtype=np.float64)
            for j in range(len(points)):
                p_loc = pts_trans[j]
                g = grad_f[j]

                J[j, 0] = -(g[0] * c + g[1] * s)          # df/dx
                J[j, 1] = -(-g[0] * s + g[1] * c)          # df/dy
                J[j, 2] = g[0] * p_loc[1] - g[1] * p_loc[0]  # df/dyaw

            H_gn = (J.T * w) @ J + 1e-4 * np.eye(3)
            b_gn = (J.T * w) @ f_vals
            delta = np.linalg.solve(H_gn, b_gn)

            x -= delta[0]
            y -= delta[1]
            yaw -= delta[2]

            if np.linalg.norm(delta) < 1e-5:
                break

        R_final = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                            [np.sin(yaw),  np.cos(yaw), 0.],
                            [0.,           0.,          1.]], dtype=np.float64)
        t_final = np.array([x, y, model.centroid[2]], dtype=np.float64)

        pts_trans_final = (R_final.T @ (points - t_final).T).T
        f_final, _, _ = model.evaluate(pts_trans_final, compute_var=False)

        score = float(np.mean(f_final**2 / sigma_r2))
        ll = float(-0.5 * np.log(2 * np.pi * sigma_r2) - np.mean(f_final**2) / (2 * sigma_r2))

        return score, R_final, t_final, ll