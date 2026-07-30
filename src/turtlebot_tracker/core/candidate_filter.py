"""
candidate_filter.py - Adaptive Cascaded Filter with Seed Prior.
v_hull_max is derived from physical robot dimensions (Turtlebot2) by default,
and optionally refined from canonical model if available and reliable.
"""

import json
import numpy as np
from scipy.spatial import ConvexHull
import open3d as o3d
from pathlib import Path
from typing import List, Tuple

from turtlebot_tracker.datatypes import ClusterCandidate, SemanticLabel


class CandidateFilter:
    """Cascaded filter evaluating hull, 2D arc curvature, dihedric planes, and volumetricity."""

    def __init__(self, config: dict):
        cfg = config.get("candidate_filter", {})

        # Adaptive parameters
        self.v_nominal = cfg.get("v_nominal", 0.005)
        self.gamma_vol = cfg.get("gamma_vol", 10.0)
        self.rho_base = cfg.get("rho_base", 80.0)
        self.rho_piso = cfg.get("rho_piso", 10.0)
        self.N0 = cfg.get("N0", 30.0)
        self.kappa_N = cfg.get("kappa_N", 5.0)

        # Volumetricity and extents
        self.volumetricity_min = cfg.get("volumetricity_min", 0.03)
        self.dihedral_rho2p_thresh = cfg.get("dihedral_rho2p_threshold", 0.88)
        self.dihedral_angle_min = np.radians(cfg.get("dihedral_angle_min_deg", 55.0))
        self.dihedral_angle_max = np.radians(cfg.get("dihedral_angle_max_deg", 125.0))

        # Extents (soft)
        self.ext_x_mu = cfg.get("ext_x_mu", 0.38)
        self.ext_x_sigma = cfg.get("ext_x_sigma", 0.15)
        self.ext_y_mu = cfg.get("ext_y_mu", 0.38)
        self.ext_y_sigma = cfg.get("ext_y_sigma", 0.15)
        self.ext_z_mu = cfg.get("ext_z_mu", 0.42)
        self.ext_z_sigma = cfg.get("ext_z_sigma", 0.18)

        # --- SEED PRIOR: Physical dimensions of Turtlebot2 ---
        # Approximate dimensions: 0.35 x 0.35 x 0.42 m → volume ≈ 0.05145 m³
        ROBOT_WIDTH = 0.35
        ROBOT_LENGTH = 0.35
        ROBOT_HEIGHT = 0.42
        V_PHYSICAL = ROBOT_WIDTH * ROBOT_LENGTH * ROBOT_HEIGHT  # ≈ 0.05145 m³

        self.kappa_vol = cfg.get("kappa_vol", 1.5)  # Safety margin (1.5x)
        self.v_hull_max = V_PHYSICAL * self.kappa_vol  # ≈ 0.077 m³
        use_prior = "PHYSICAL_SEED"

        # --- Try to refine from canonical model (if reliable) ---
        points_path = Path("config/canonical_points.json")
        if points_path.exists():
            try:
                with open(points_path, 'r') as f:
                    pts_data = json.load(f)
                pts = np.array(pts_data.get("canonical_points", []))
                if len(pts) > 100:  # Enough points for a meaningful hull
                    V_canon = float(ConvexHull(pts).volume)
                    if V_canon > 0.01:  # Sanity check: not a flat plane or noise
                        self.v_hull_max = V_canon * self.kappa_vol
                        use_prior = "CANONICAL_MODEL"
                        print(f"[INFO] v_hull_max from canonical model: {self.v_hull_max:.4f} m³ "
                              f"(κ={self.kappa_vol}, V_canon={V_canon:.4f})")
                    else:
                        print(f"[WARN] Canonical volume too small ({V_canon:.4f}). Using physical prior.")
                else:
                    print(f"[INFO] Canonical points insufficient ({len(pts)} pts). Using physical prior.")
            except Exception as e:
                print(f"[WARN] Could not load canonical model: {e}. Using physical prior.")

        if use_prior == "PHYSICAL_SEED":
            print(f"[INFO] v_hull_max set to PHYSICAL dimensions: {self.v_hull_max:.4f} m³ "
                  f"(κ={self.kappa_vol}, V_physical={V_PHYSICAL:.4f})")

        # Arc rescue (optional)
        self.taubin_r_min = cfg.get("taubin_radius_min", 0.14)
        self.taubin_r_max = cfg.get("taubin_radius_max", 0.22)
        self.taubin_rms_max = cfg.get("taubin_rms_max", 0.025)
        self.enable_arc_rescue = cfg.get("enable_arc_rescue", False)

    def filter_candidates(self, clusters: List[ClusterCandidate]) -> List[ClusterCandidate]:
        """Apply adaptive cascaded filter to each cluster."""
        for cand in clusters:
            pts = cand.points
            num_points = len(pts)

            if num_points < 12:
                cand.passed_filters = False
                continue

            # Radial distance to cluster centroid
            r_centroid = float(np.linalg.norm(cand.centroid))

            # Adaptive hull bounds
            v_min_adaptive = self.v_nominal * (1.0 - np.exp(-self.gamma_vol * num_points / (r_centroid**2 + 1e-6)))
            rho_sol_adaptive = (self.rho_base / (1.0 + np.exp(-(num_points - self.N0) / self.kappa_N))
                                + self.rho_piso)

            try:
                hull = ConvexHull(pts)
                cand.v_hull = float(hull.volume)
                cand.solidity = num_points / np.maximum(cand.v_hull, 1e-5)
            except Exception:
                cand.v_hull = 0.02
                cand.solidity = 30.0

            # --- Core check: volume must be within [v_min, v_max] ---
            hull_pass = (v_min_adaptive <= cand.v_hull <= self.v_hull_max) and (cand.solidity >= rho_sol_adaptive)

            # Taubin 2D Crescent Arc Rescue (fallback for concave clouds)
            _, _, r_fit, rms_fit = self._fit_taubin_circle_2d(pts[:, :2])
            cand.is_arc_valid = ((self.taubin_r_min <= r_fit <= self.taubin_r_max) and
                                 (rms_fit <= self.taubin_rms_max))

            if not hull_pass and self.enable_arc_rescue and cand.is_arc_valid:
                hull_pass = True

            # Dihedral 2-plane RANSAC structural veto
            cand.rho_2p, cand.is_corner = self._evaluate_dihedral_test(pts)

            # Global volumetricity & soft extents
            cov = np.cov(pts.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.maximum(eigvals, 1e-6)
            cand.volumetricity = float(eigvals[0] / np.sum(eigvals))

            extents = np.max(pts, axis=0) - np.min(pts, axis=0)
            log_lik = (
                -0.5 * ((extents[0] - self.ext_x_mu) / self.ext_x_sigma) ** 2 -
                0.5 * ((extents[1] - self.ext_y_mu) / self.ext_y_sigma) ** 2 -
                0.5 * ((extents[2] - self.ext_z_mu) / self.ext_z_sigma) ** 2
            )
            cand.extent_likelihood = float(log_lik)

            cand.passed_filters = (
                hull_pass and
                (not cand.is_corner) and
                (cand.volumetricity >= self.volumetricity_min) and
                (cand.extent_likelihood > -15.0)
            )

        return clusters

    def _fit_taubin_circle_2d(self, points_2d: np.ndarray) -> Tuple[float, float, float, float]:
        """Fit a 2D algebraic circle (Taubin/Pratt method) to the XY projection."""
        num_points = len(points_2d)
        if num_points < 6:
            return 0.0, 0.0, 0.0, 999.0

        x = points_2d[:, 0]
        y = points_2d[:, 1]
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        u = x - x_mean
        v = y - y_mean
        z = u**2 + v**2

        m_uu = np.mean(u**2)
        m_uv = np.mean(u * v)
        m_vv = np.mean(v**2)
        m_uz = np.mean(u * z)
        m_vz = np.mean(v * z)

        cov_matrix = np.array([[m_uu, m_uv], [m_uv, m_vv]], dtype=np.float64)
        rhs_vector = 0.5 * np.array([m_uz, m_vz], dtype=np.float64)

        try:
            center_offset = np.linalg.solve(cov_matrix, rhs_vector)
            a_center = x_mean + center_offset[0]
            b_center = y_mean + center_offset[1]

            r_fit = np.sqrt(center_offset[0]**2 + center_offset[1]**2 + m_uu + m_vv)
            residuals = np.sqrt((x - a_center)**2 + (y - b_center)**2) - r_fit
            rms_fit = float(np.sqrt(np.mean(residuals**2)))

            return float(a_center), float(b_center), float(r_fit), rms_fit
        except np.linalg.LinAlgError:
            return 0.0, 0.0, 0.0, 999.0

    def _evaluate_dihedral_test(self, points: np.ndarray) -> Tuple[float, bool]:
        """Evaluate priority 2-plane RANSAC fit for architectural corners."""
        num_points = len(points)
        if num_points < 20:
            return 0.0, False

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        plane1, inliers1 = pcd.segment_plane(distance_threshold=0.04, ransac_n=3, num_iterations=30)
        if len(inliers1) < 6:
            return 0.0, False

        remainder = pcd.select_by_index(inliers1, invert=True)
        if len(remainder.points) < 6:
            return float(len(inliers1) / num_points), False

        plane2, inliers2 = remainder.segment_plane(distance_threshold=0.04, ransac_n=3, num_iterations=30)

        rho_2p = (len(inliers1) + len(inliers2)) / num_points
        n1 = plane1[:3] / np.linalg.norm(plane1[:3])
        n2 = plane2[:3] / np.linalg.norm(plane2[:3])
        angle = np.arccos(np.clip(np.abs(np.dot(n1, n2)), 0.0, 1.0))

        is_corner = (rho_2p >= self.dihedral_rho2p_thresh) and (self.dihedral_angle_min <= angle <= self.dihedral_angle_max)
        return float(rho_2p), is_corner