"""
src/turtlebot_tracker/core/registration.py - Direct GMM EM registration with surprise guard and von Mises compass.
Now initialized with EKF prior for temporal consistency.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from scipy.special import i0
from turtlebot_tracker.datatypes import FrameData, ClusterCandidate, TrackingState
from turtlebot_tracker.core.hierarchical_gmm import HierarchicalGMM

class DirectGMMRegistrator:
    def __init__(self, config: dict, canonical_tree_path: str = "config/canonical_tree.json"):
        cfg = config["registration"]
        self.surprise_threshold = cfg["surprise_threshold"]
        self.outlier_weight = cfg["outlier_weight"]
        self.max_em_iterations = cfg["max_em_iterations"]
        self.kappa_default = cfg["von_mises_default_kappa"]
        self.kappa_min = cfg["von_mises_kappa_min"]
        self.kappa_max = cfg["von_mises_kappa_max"]
        self.v_max = cfg["robot_v_max"]

        tree_path = Path(canonical_tree_path)
        if tree_path.exists():
            self.hg = HierarchicalGMM.load(tree_path)
            self.canonical_gmm = self.hg.original
        else:
            self.canonical_gmm = [
                {'mu': [0.0, 0.0, -0.18], 'cov': np.eye(3)*0.01, 'weight': 0.35},
                {'mu': [0.0, 0.0,  0.00], 'cov': np.eye(3)*0.01, 'weight': 0.35},
                {'mu': [0.0, 0.0,  0.20], 'cov': np.eye(3)*0.01, 'weight': 0.30}
            ]
            self.hg = HierarchicalGMM(self.canonical_gmm)

    def register_and_track(
        self, frame_data: FrameData, candidates: List[ClusterCandidate], ekf_tracker
    ) -> Tuple[TrackingState, Optional[ClusterCandidate]]:

        passed_candidates = [c for c in candidates if c.passed_filters]
        if not passed_candidates:
            ekf_tracker.predict(0.1)
            state = ekf_tracker.get_state()
            state.surprise_triggered = True
            kappa = state.bearing_compass_kappa
            inv_kappa = 1.0/kappa + (self.v_max * 0.1 / 1.0)**2
            state.bearing_compass_kappa = np.clip(1.0/inv_kappa, self.kappa_min, self.kappa_max)
            return state, None

        N = len(passed_candidates[0].points)
        K0 = len(self.canonical_gmm)
        if N < 80:
            M = min(2, K0)
        elif N < 150:
            M = min(3, K0)
        else:
            M = K0
        gmm = self.hg.get_level(M)

        # --- FIX: Get EKF prior for initialization ---
        pred_pose = ekf_tracker.get_state().pose_se2
        is_ekf_initialized = len(ekf_tracker.trajectory_log) > 0
        init_pose = None
        if is_ekf_initialized:
            psi = pred_pose[2]
            init_R = np.array([
                [np.cos(psi), -np.sin(psi), 0.0],
                [np.sin(psi),  np.cos(psi), 0.0],
                [0.0,          0.0,         1.0]
            ])
            init_t = np.array([pred_pose[0], pred_pose[1], 0.0])
            init_pose = (init_R, init_t)

        best_cand = None
        best_score = -np.inf
        best_R = np.eye(3)
        best_t = np.zeros(3)

        for cand in passed_candidates:
            R_est, t_est, ll = self._fit_em_se2(cand.points, gmm, init_pose=init_pose)
            # Gating penalty (only if EKF is initialized)
            if is_ekf_initialized:
                dist_to_prior = np.linalg.norm(t_est[:2] - pred_pose[:2])
                gating_penalty = 2.0 * dist_to_prior
            else:
                gating_penalty = 0.0
            score = ll - gating_penalty
            if score > best_score:
                best_score = score
                best_cand = cand
                best_R = R_est
                best_t = t_est

        # Surprise guard and Compass fallback
        if best_score < self.surprise_threshold or best_cand is None:
            compass_kappa = ekf_tracker.bearing_compass_kappa
            compass_mu = ekf_tracker.bearing_compass_mu
            for cand in passed_candidates:
                bearing = np.arctan2(cand.centroid[1], cand.centroid[0])
                R_est, t_est, ll = self._fit_em_se2(cand.points, gmm, init_pose=init_pose)
                log_prior = compass_kappa * np.cos(bearing - compass_mu) - np.log(2*np.pi*i0(compass_kappa))
                score = ll + log_prior
                if score > best_score:
                    best_score = score
                    best_cand = cand
                    best_R = R_est
                    best_t = t_est

            if best_cand is None:
                ekf_tracker.predict(0.1)
                state = ekf_tracker.get_state()
                state.surprise_triggered = True
                return state, None
            else:
                ekf_tracker.predict(0.1)
                bearing = np.arctan2(best_t[1], best_t[0])
                mu_b = ekf_tracker.bearing_compass_mu
                kappa_b = ekf_tracker.bearing_compass_kappa
                A = kappa_b * np.cos(mu_b) + self.kappa_default * np.cos(bearing)
                B = kappa_b * np.sin(mu_b) + self.kappa_default * np.sin(bearing)
                mu_post = np.arctan2(B, A)
                kappa_post = np.sqrt(A**2 + B**2)
                ekf_tracker.bearing_compass_mu = mu_post
                ekf_tracker.bearing_compass_kappa = np.clip(kappa_post, self.kappa_min, self.kappa_max)

                yaw = np.arctan2(best_R[1,0], best_R[0,0])
                z_meas = np.array([best_t[0], best_t[1], yaw])
                ekf_tracker.update(z_meas)
        else:
            # Normal update
            ekf_tracker.predict(0.1)
            yaw = np.arctan2(best_R[1,0], best_R[0,0])
            z_meas = np.array([best_t[0], best_t[1], yaw])
            ekf_tracker.update(z_meas)
            bearing = np.arctan2(best_t[1], best_t[0])
            ekf_tracker.bearing_compass_mu = bearing
            ekf_tracker.bearing_compass_kappa = self.kappa_default

        state = ekf_tracker.get_state()
        state.surprise_triggered = False
        return state, best_cand

    def _fit_em_se2(self, points: np.ndarray, gmm: List[Dict], init_pose: Tuple[np.ndarray, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        EM registration. If init_pose is provided, starts from that pose instead of identity.
        """
        N = len(points)
        K = len(gmm)
        mu_canon = np.array([g['mu'] for g in gmm])
        Sigma_canon = np.array([g['cov'] for g in gmm])
        weights = np.array([g['weight'] for g in gmm])
        weights = weights / np.sum(weights)

        cand_center = np.mean(points, axis=0)
        rel_points = points - cand_center

        # --- Initialize with prior if available ---
        if init_pose is not None:
            R_init, t_init = init_pose
            t_offset_init = t_init - cand_center
        else:
            R_init = np.eye(3)
            t_offset_init = np.zeros(3)

        R_est = R_init
        t_offset = t_offset_init
        log_likelihood = -999.0

        for _ in range(self.max_em_iterations):
            transformed_mu = (R_est @ mu_canon.T).T + t_offset
            probs = np.zeros((N, K))
            for k in range(K):
                diff = rel_points - transformed_mu[k]
                cov_k = Sigma_canon[k] + np.eye(3)*1e-6
                inv_cov = np.linalg.inv(cov_k)
                maha = np.sum(diff @ inv_cov * diff, axis=1)
                det = np.linalg.det(cov_k)
                probs[:, k] = weights[k] * np.exp(-0.5*maha) / np.sqrt((2*np.pi)**3 * det)

            densities = np.sum(probs, axis=1) + self.outlier_weight
            gamma = probs / densities[:, None]
            log_likelihood = float(np.mean(np.log(densities)))

            P_weight = np.sum(gamma)
            if P_weight < 1e-6:
                break

            center_canon = np.sum(mu_canon * np.sum(gamma, axis=0)[:, None], axis=0) / P_weight
            center_obs = np.sum(rel_points * np.sum(gamma, axis=1)[:, None], axis=0) / P_weight

            H = (rel_points - center_obs).T @ gamma @ (mu_canon - center_canon)
            U, _, Vt = np.linalg.svd(H)
            R_opt = Vt.T @ U.T

            yaw = np.arctan2(R_opt[1,0], R_opt[0,0])
            R_est = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw),  np.cos(yaw), 0.0],
                [0.0,          0.0,         1.0]
            ])
            t_offset = center_obs - R_est @ center_canon

        final_t = cand_center + t_offset
        return R_est, final_t, log_likelihood