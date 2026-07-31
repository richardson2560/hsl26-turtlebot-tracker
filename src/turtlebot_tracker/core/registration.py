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

from turtlebot_tracker.core.hierarchical_gmm import HierarchicalGMM
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
#  GMM REGISTRATOR (original, mantiene compatibilidad)
# ============================================================================

class DirectGMMRegistrator:
    """Direct GMM MAP registrator with VMF, Range-Aware Inflation, Multi-start."""

    def __init__(self, config: dict, model_path: str = "config/canonical_turtlebot2.json"):
        cfg = config.get("registration", {})
        self.surprise_threshold = cfg.get("surprise_threshold", -12.0)
        self.outlier_weight = cfg.get("outlier_weight", 0.05)
        self.max_em_iterations = cfg.get("max_em_iterations", 8)
        self.sh_intensity_std = cfg.get("sh_intensity_std", 15.0)
        self.mahalanobis_trim_chi2 = cfg.get("mahalanobis_trim_chi2", 11.34)
        self.jump_penalty_weight = cfg.get("jump_penalty_weight", 1.5)
        self.r_max_clamp = cfg.get("r_max_clamp", 2.5)
        self.default_sigma = cfg.get("default_sigma", 0.005)
        self.n_starts = cfg.get("n_starts", 8)
        self.container_volume = cfg.get("container_volume", 5.0)

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            self.canonical_gmm = data[0] if data else []
            self.hg = HierarchicalGMM([])
            self.hg.tree = data
        elif isinstance(data, dict):
            self.canonical_gmm = data.get("canonical_gaussians", data.get("gaussians", []))
            self.hg = HierarchicalGMM(self.canonical_gmm)
        else:
            raise TypeError("Unsupported model format.")

        if not self.canonical_gmm:
            raise ValueError("No Gaussian components found.")

        self._precompute_splat_data()
        calib_path = Path("config/range_noise_calibration.json")
        if calib_path.exists():
            with open(calib_path, 'r') as f:
                calib_data = json.load(f)
            r_mid = np.array(calib_data['r_mid'])
            sigma_vals = np.array(calib_data['sigma'])
            self.sigma_lookup = interp1d(r_mid, sigma_vals, kind='linear',
                                         fill_value=(sigma_vals[0], sigma_vals[-1]),
                                         bounds_error=False)
        else:
            self.sigma_lookup = lambda r: self.default_sigma

    def _precompute_splat_data(self):
        K = len(self.canonical_gmm)
        self.normals_canon = []
        self.means_range = []
        for g in self.canonical_gmm:
            cov = np.array(g['cov'])
            vals, vecs = np.linalg.eigh(cov)
            normal = vecs[:, 0]
            self.normals_canon.append(normal)
            self.means_range.append(g.get('mean_range', 0.5))

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
            N_c = len(cand.points)
            K0 = len(self.canonical_gmm)
            if N_c < 30:
                M_level = min(2, K0)
            elif N_c < 60:
                M_level = min(3, K0)
            else:
                M_level = K0
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

        N_pts = len(best_cand.points)
        log_l0 = N_pts * np.log(self.outlier_weight / self.container_volume)
        lambda_lrt = 2.0 * (best_log_likelihood - log_l0)
        lrt_threshold = chi2.ppf(0.999, df=3)

        if (best_score < self.surprise_threshold) or (lambda_lrt < lrt_threshold):
            ekf_tracker.update_lifecycle(detection_accepted=False, dt=0.1)
            state = ekf_tracker.get_state()
            state.surprise_triggered = True
            return state, None

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
        if init_pose is not None:
            return self._fit_em_se2_single(points, intensity, gmm, init_pose)

        cand_center = np.mean(points, axis=0)
        best_ll = -np.inf
        best_R = np.eye(3)
        best_t = np.zeros(3)
        angles = np.linspace(0, 2 * np.pi, self.n_starts, endpoint=False)
        for yaw in angles:
            R0 = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw),  np.cos(yaw), 0.0],
                [0.0,          0.0,         1.0]
            ], dtype=np.float64)
            t0 = cand_center.copy()
            R_est, t_est, ll = self._fit_em_se2_single(points, intensity, gmm, (R0, t0))
            if ll > best_ll:
                best_ll = ll
                best_R = R_est
                best_t = t_est
        return best_R, best_t, best_ll

    def _fit_em_se2_single(
        self,
        points: np.ndarray,
        intensity: np.ndarray,
        gmm: List[Dict],
        init_pose: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        N = len(points)
        K = len(gmm)
        if N < 5 or K == 0:
            return np.eye(3), np.zeros(3), -999.0

        mu_can = np.array([g['mu'] for g in gmm], dtype=np.float64)
        Sigma_can = np.array([g['cov'] for g in gmm], dtype=np.float64)
        weights = np.array([g['weight'] for g in gmm], dtype=np.float64)
        weights /= np.sum(weights)
        sh_c0_can = np.array([g.get('sh_c0', 28.0) for g in gmm], dtype=np.float64)
        mu_dir_can = np.array([g.get('mu_dir', [0, 0, 1]) for g in gmm], dtype=np.float64)
        kappa_can = np.array([g.get('kappa', 1.0) for g in gmm], dtype=np.float64)
        mean_range_can = np.array([g.get('mean_range', 0.5) for g in gmm], dtype=np.float64)

        cand_center = np.mean(points, axis=0)
        rel_points = points - cand_center
        r_obs = float(np.linalg.norm(cand_center))
        view_dir = -cand_center / (r_obs + 1e-6)

        r_i = np.linalg.norm(points, axis=1)
        r_clamped = np.minimum(r_i, self.r_max_clamp)
        cos_eta = np.maximum(np.abs(points[:, 2]) / np.maximum(r_i, 1e-6), 0.1)
        I_corr = np.clip(intensity * (r_clamped**2) / cos_eta, 0.0, 255.0)

        R_init, t_init = init_pose
        t_offset = t_init - cand_center
        R_est = R_init.copy()
        log_likelihood = -999.0

        for _ in range(self.max_em_iterations):
            vmf_factors = np.array([
                evaluate_vmf_likelihood(view_dir, R_est @ mu_dir_can[k], kappa_can[k])
                for k in range(K)
            ])
            pi_eff = weights * vmf_factors
            pi_eff /= (np.sum(pi_eff) + 1e-9)

            transformed_mu = (R_est @ mu_can.T).T + t_offset
            min_maha_sq = np.full(N, np.inf, dtype=np.float64)
            probs = np.zeros((N, K), dtype=np.float64)

            for k in range(K):
                cov_k = R_est @ Sigma_can[k] @ R_est.T
                sigma_splat = self.sigma_lookup(mean_range_can[k])
                sigma_obs = self.sigma_lookup(r_obs)
                extra_var = max(sigma_obs**2 - sigma_splat**2, 0.0)
                if extra_var > 0:
                    normal_k = self.normals_canon[k] if k < len(self.normals_canon) else np.array([0,0,1])
                    normal_rot = R_est @ normal_k
                    cov_k = cov_k + extra_var * np.outer(normal_rot, normal_rot)
                cov_k = cov_k + np.eye(3) * 1e-6

                diff = rel_points - transformed_mu[k]
                inv_cov = np.linalg.inv(cov_k)
                maha_sq = np.sum(diff @ inv_cov * diff, axis=1)
                min_maha_sq = np.minimum(min_maha_sq, maha_sq)

                det_k = np.maximum(np.linalg.det(cov_k), 1e-12)
                geom_pdf = np.exp(-0.5 * maha_sq) / np.sqrt((2.0 * np.pi)**3 * det_k)

                rad_diff_sq = (I_corr - sh_c0_can[k])**2
                rad_pdf = np.exp(-rad_diff_sq / (2.0 * self.sh_intensity_std**2)) / (np.sqrt(2.0 * np.pi) * self.sh_intensity_std)

                probs[:, k] = pi_eff[k] * geom_pdf * rad_pdf

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

            center_canon = np.sum(mu_can * np.sum(gamma_trimmed, axis=0)[:, None], axis=0) / P_weight
            center_obs = np.sum(points_trimmed * np.sum(gamma_trimmed, axis=1)[:, None], axis=0) / P_weight

            H = (points_trimmed - center_obs).T @ gamma_trimmed @ (mu_can - center_canon)
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