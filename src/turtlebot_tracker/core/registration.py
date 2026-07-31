"""
registration.py - GPIS-W Registrator (without splats/GMM).
Only uses the Hermite-GPIS-W implicit surface model.
Works with dict-based candidates from online_segmenter.
"""

import numpy as np
from turtlebot_tracker.core.implicit_surface import load_model

def wrap_to_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))

class GPISRegistrator:
    """
    Registrator using Hermite-GPIS-W implicit surface model.
    No GMM / splats are used.
    """

    def __init__(self, config: dict, model_path: str = "config/implicit_model.json"):
        cfg = config.get("registration", {})
        self.score_threshold = cfg.get("score_threshold", 2.0)
        self.max_iter = cfg.get("max_gn_iterations", 4)
        self.n_starts = cfg.get("n_starts", 8)
        self.sigma_r = cfg.get("sigma_lidar", 0.012)
        self.model = load_model(model_path)
        print(f"[GPIS] Model loaded: {self.model.M} primitives, centroid={self.model.centroid}")

    def register_and_track(self, frame_data, candidates, ekf_tracker, dt=0.1):
        """
        Register the best candidate against the implicit model and update EKF.
        If no candidates, just coast (predict only).
        Returns: (state, best_cand, accepted)
        """
        # Always predict first (inertial coasting)
        ekf_tracker.predict(dt)

        if not candidates:
            ekf_tracker.update(None, is_target=False, dt=dt)
            return ekf_tracker.get_state(), None, False

        # Candidates are dicts from online_segmenter: they have keys 'points', 'indices', 'centroid'
        best_cand = None
        best_score = np.inf
        best_pose = None

        for cand in candidates:
            # cand is a dict with 'points' (Nx3 array)
            if 'points' not in cand or len(cand['points']) < 10:
                continue
            score, R, t, ll = self._fit_gpis_se2(cand['points'], init_pose=None)
            if score < best_score:
                best_score = score
                best_cand = cand
                yaw = wrap_to_pi(np.arctan2(R[1, 0], R[0, 0]))
                best_pose = np.array([t[0], t[1], yaw])

        if best_cand is None:
            ekf_tracker.update(None, is_target=False, dt=dt)
            return ekf_tracker.get_state(), None, False

        is_target = best_score < self.score_threshold
        ekf_tracker.z_ground = frame_data.z_ground
        accepted = ekf_tracker.update(best_pose, is_target=is_target, score_val=best_score, dt=dt)

        state = ekf_tracker.get_state()
        return state, best_cand, accepted

    # -------------------------------------------------------------------------
    # Private registration methods (GPIS-W Gauss-Newton)
    # -------------------------------------------------------------------------

    def _fit_gpis_se2(self, points, init_pose=None):
        """
        Fit the GPIS-W model to the given points in SE(2).
        If init_pose is provided, use it; otherwise do a smart multi-start.
        Returns: (score, R, t, log_likelihood)
        """
        if init_pose is not None:
            return self._run_gauss_newton(points, init_pose[0], init_pose[1], max_iter=self.max_iter)

        cand_center = np.mean(points, axis=0)
        angles = np.linspace(0, 2 * np.pi, self.n_starts, endpoint=False)

        # Smart screening: evaluate initial residual for each angle
        init_scores = []
        for yaw0 in angles:
            R0 = np.array([
                [np.cos(yaw0), -np.sin(yaw0), 0.],
                [np.sin(yaw0),  np.cos(yaw0), 0.],
                [0.,            0.,          1.]
            ], dtype=np.float64)
            t0 = np.array([cand_center[0], cand_center[1], self.model.centroid[2]], dtype=np.float64)
            pts_trans = (R0.T @ (points - t0).T).T
            f_vals, _, _ = self.model.evaluate(pts_trans, compute_var=False)
            score_init = float(np.mean(f_vals**2 / (self.sigma_r ** 2)))
            init_scores.append((score_init, R0, t0))

        init_scores.sort(key=lambda x: x[0])
        top_seeds = init_scores[:2]   # only keep the two best initial angles

        best_score = np.inf
        best_R = np.eye(3)
        best_t = np.zeros(3)
        best_ll = -999.0

        for _, R0, t0 in top_seeds:
            score, R, t, ll = self._run_gauss_newton(points, R0, t0, max_iter=3)
            if score < best_score:
                best_score = score
                best_R = R
                best_t = t
                best_ll = ll

        return best_score, best_R, best_t, best_ll

    def _run_gauss_newton(self, points, R0, t0, max_iter=4):
        """
        Gauss-Newton optimization on SE(2) using the implicit model.
        """
        model = self.model
        R = R0.copy()
        yaw = np.arctan2(R[1, 0], R[0, 0])
        x, y = t0[0], t0[1]
        sigma_r2 = self.sigma_r ** 2

        for _ in range(max_iter):
            c, s = np.cos(yaw), np.sin(yaw)
            R_curr = np.array([
                [c, -s, 0.],
                [s,  c, 0.],
                [0., 0., 1.]
            ], dtype=np.float64)
            t_curr = np.array([x, y, model.centroid[2]], dtype=np.float64)

            # Transform points to model coordinates
            pts_trans = (R_curr.T @ (points - t_curr).T).T

            # Evaluate implicit function and its gradient
            f_vals, grad_f, _ = model.evaluate(pts_trans, compute_var=False)

            # Huber weighting
            abs_f = np.abs(f_vals)
            w_huber = np.where(abs_f < 0.03, 1.0, 0.03 / (abs_f + 1e-6))
            w = w_huber / sigma_r2

            # Jacobian d(f)/d(x, y, yaw)
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

        R_final = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.],
            [np.sin(yaw),  np.cos(yaw), 0.],
            [0.,           0.,          1.]
        ], dtype=np.float64)
        t_final = np.array([x, y, model.centroid[2]], dtype=np.float64)

        # Final evaluation
        pts_trans_final = (R_final.T @ (points - t_final).T).T
        f_final, _, _ = model.evaluate(pts_trans_final, compute_var=False)

        score = float(np.mean(f_final**2 / sigma_r2))
        ll = float(-0.5 * np.log(2 * np.pi * sigma_r2) - np.mean(f_final**2) / (2 * sigma_r2))

        return score, R_final, t_final, ll