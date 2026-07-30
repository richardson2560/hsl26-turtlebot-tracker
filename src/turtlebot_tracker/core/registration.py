"""
registration.py - Direct GMM MAP Registrator with Covariance Rotation & Wilks GLRT.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import chi2

from turtlebot_tracker.core.hierarchical_gmm import HierarchicalGMM
from turtlebot_tracker.datatypes import ClusterCandidate, FrameData, TrackingState


def wrap_to_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class DirectGMMRegistrator:
    """Direct GMM MAP registrator with Trimmed EM, rotated covariance, and Wilks GLRT."""

    def __init__(self, config: dict, canonical_tree_path: str = "config/canonical_tree.json"):
        cfg = config.get("registration", {})
        self.surprise_threshold = cfg.get("surprise_threshold", -12.0)
        self.outlier_weight = cfg.get("outlier_weight", 0.05)
        self.max_em_iterations = cfg.get("max_em_iterations", 8)
        self.sh_intensity_std = cfg.get("sh_intensity_std", 15.0)
        self.mahalanobis_trim_chi2 = cfg.get("mahalanobis_trim_chi2", 11.34)
        self.jump_penalty_weight = cfg.get("jump_penalty_weight", 1.5)
        self.r_max_clamp = cfg.get("r_max_clamp", 2.5)

        tree_path = Path(canonical_tree_path)
        if tree_path.exists():
            self.hg = HierarchicalGMM.load(tree_path)
            self.canonical_gmm = self.hg.original
        else:
            self.canonical_gmm = [
                {'mu': [0.0, 0.0, -0.18], 'cov': (np.eye(3) * 0.01).tolist(), 'sh_c0': 28.0, 'weight': 0.35},
                {'mu': [0.0, 0.0,  0.00], 'cov': (np.eye(3) * 0.01).tolist(), 'sh_c0': 28.0, 'weight': 0.35},
                {'mu': [0.0, 0.0,  0.20], 'cov': (np.eye(3) * 0.01).tolist(), 'sh_c0': 28.0, 'weight': 0.30}
            ]
            self.hg = HierarchicalGMM(self.canonical_gmm)

    def register_and_track(
        self, frame_data: FrameData, candidates: List[ClusterCandidate], ekf_tracker
    ) -> Tuple[TrackingState, Optional[ClusterCandidate]]:
        passed_candidates = [c for c in candidates if c.passed_filters]

        if not passed_candidates:
            ekf_tracker.update_lifecycle(detection_accepted=False, dt=0.1)
            state = ekf_tracker.get_state()
            state.surprise_triggered = True
            return state, None

        pred_pose = ekf_tracker.get_state().pose_se2
        is_ekf_initialized = ekf_tracker.is_initialized

        init_pose = None
        if is_ekf_initialized:
            psi_p = pred_pose[2]
            init_R = np.array([
                [np.cos(psi_p), -np.sin(psi_p), 0.0],
                [np.sin(psi_p),  np.cos(psi_p), 0.0],
                [0.0,            0.0,           1.0]
            ], dtype=np.float64)
            init_t = np.array([pred_pose[0], pred_pose[1], 0.0], dtype=np.float64)
            init_pose = (init_R, init_t)

        best_cand = None
        best_score = -np.inf
        best_R = np.eye(3, dtype=np.float64)
        best_t = np.zeros(3, dtype=np.float64)
        best_log_likelihood = -999.0

        for cand in passed_candidates:
            # Select level M individually per candidate
            N_c = len(cand.points)
            K0 = len(self.canonical_gmm)
            M_level = 2 if N_c < 60 else (3 if N_c < 120 else K0)
            gmm_cand = self.hg.get_level(M_level)

            R_est, t_est, ll = self._fit_em_se2(cand.points, cand.intensity, gmm_cand, init_pose=init_pose)

            if is_ekf_initialized:
                P_pos_inv = np.linalg.inv(ekf_tracker.P[0:2, 0:2] + np.eye(2) * 1e-4)
                pos_diff = t_est[:2] - pred_pose[:2]
                jump_penalty = 0.5 * float(pos_diff.T @ P_pos_inv @ pos_diff)
                yaw_diff = wrap_to_pi(np.arctan2(R_est[1, 0], R_est[0, 0]) - pred_pose[2])
                rot_penalty = (1.0 - np.cos(yaw_diff)) / max(ekf_tracker.P[2, 2], 1e-4)
                map_score = ll - self.jump_penalty_weight * (jump_penalty + rot_penalty)
            else:
                map_score = ll

            cand.map_score = map_score

            if map_score > best_score:
                best_score = map_score
                best_cand = cand
                best_R = R_est
                best_t = t_est
                best_log_likelihood = ll

        if best_cand is None:
            ekf_tracker.update_lifecycle(detection_accepted=False, dt=0.1)
            state = ekf_tracker.get_state()
            state.surprise_triggered = True
            return state, None

        # Wilks GLRT Test against Null Hypothesis (Uniform Noise)
        N_pts = len(best_cand.points)
        vol_container = max(best_cand.v_hull, 0.01)
        log_l0 = N_pts * np.log(self.outlier_weight / vol_container)
        lambda_lrt = 2.0 * (best_log_likelihood - log_l0)
        lrt_threshold = chi2.ppf(0.99, df=3)  # Exact Wilks threshold = 11.3449

        if (best_score < self.surprise_threshold) or (lambda_lrt < lrt_threshold):
            ekf_tracker.update_lifecycle(detection_accepted=False, dt=0.1)
            state = ekf_tracker.get_state()
            state.surprise_triggered = True
            return state, None

        # Accept detection
        yaw_meas = wrap_to_pi(np.arctan2(best_R[1, 0], best_R[0, 0]))
        z_meas = np.array([best_t[0], best_t[1], yaw_meas], dtype=np.float64)

        ekf_tracker.update_lifecycle(detection_accepted=True, dt=0.1)
        ekf_tracker.update(z_meas)

        state = ekf_tracker.get_state()
        state.surprise_triggered = False
        return state, best_cand

    def _fit_em_se2(
        self,
        points: np.ndarray,
        intensity: np.ndarray,
        gmm: List[Dict],
        init_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        N = len(points)
        K = len(gmm)

        mu_canon = np.array([g['mu'] for g in gmm], dtype=np.float64)
        Sigma_canon = np.array([g['cov'] for g in gmm], dtype=np.float64)
        weights = np.array([g['weight'] for g in gmm], dtype=np.float64)
        weights /= np.sum(weights)
        sh_c0_canon = np.array([g.get('sh_c0', 28.0) for g in gmm], dtype=np.float64)

        cand_center = np.mean(points, axis=0)
        rel_points = points - cand_center

        r_i = np.linalg.norm(points, axis=1)
        r_clamped = np.minimum(r_i, self.r_max_clamp)
        cos_eta = np.maximum(np.abs(points[:, 2]) / np.maximum(r_i, 1e-6), 0.1)
        I_corr = np.clip(intensity * (r_clamped**2) / cos_eta, 0.0, 255.0)

        if init_pose is not None:
            R_init, t_init = init_pose
            t_offset = t_init - cand_center
            R_est = R_init.copy()
        else:
            R_est = np.eye(3, dtype=np.float64)
            t_offset = np.zeros(3, dtype=np.float64)

        log_likelihood = -999.0

        for _ in range(self.max_em_iterations):
            transformed_mu = (R_est @ mu_canon.T).T + t_offset
            min_maha_sq = np.full(N, np.inf, dtype=np.float64)
            probs = np.zeros((N, K), dtype=np.float64)

            for k in range(K):
                cov_k = R_est @ Sigma_canon[k] @ R_est.T + np.eye(3) * 1e-5

                diff = rel_points - transformed_mu[k]
                inv_cov = np.linalg.inv(cov_k)
                maha_sq = np.sum(diff @ inv_cov * diff, axis=1)
                min_maha_sq = np.minimum(min_maha_sq, maha_sq)

                det_k = np.maximum(np.linalg.det(cov_k), 1e-12)
                geom_pdf = np.exp(-0.5 * maha_sq) / np.sqrt((2.0 * np.pi)**3 * det_k)

                rad_diff_sq = (I_corr - sh_c0_canon[k])**2
                rad_pdf = np.exp(-rad_diff_sq / (2.0 * self.sh_intensity_std**2)) / (np.sqrt(2.0 * np.pi) * self.sh_intensity_std)

                probs[:, k] = weights[k] * geom_pdf * rad_pdf

            trim_mask = min_maha_sq <= self.mahalanobis_trim_chi2
            if not np.any(trim_mask):
                trim_mask = np.ones(N, dtype=bool)

            densities = np.sum(probs, axis=1) + self.outlier_weight
            gamma = probs / densities[:, None]

            log_likelihood = float(np.mean(np.log(np.maximum(densities[trim_mask], 1e-12))))

            P_weight = np.sum(gamma[trim_mask])
            if P_weight < 1e-5:
                break

            gamma_trimmed = gamma[trim_mask]
            points_trimmed = rel_points[trim_mask]

            center_canon = np.sum(mu_canon * np.sum(gamma_trimmed, axis=0)[:, None], axis=0) / P_weight
            center_obs = np.sum(points_trimmed * np.sum(gamma_trimmed, axis=1)[:, None], axis=0) / P_weight

            H = (points_trimmed - center_obs).T @ gamma_trimmed @ (mu_canon - center_canon)
            U, _, Vt = np.linalg.svd(H)

            d = np.linalg.det(Vt.T @ U.T)
            S_mat = np.diag([1.0, 1.0, np.sign(d)])
            R_opt = Vt.T @ S_mat @ U.T

            yaw = wrap_to_pi(np.arctan2(R_opt[1, 0], R_opt[0, 0]))
            R_est = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw),  np.cos(yaw), 0.0],
                [0.0,          0.0,         1.0]
            ], dtype=np.float64)

            t_offset = center_obs - R_est @ center_canon

        final_t = cand_center + t_offset
        return R_est, final_t, log_likelihood