"""
candidate_filter.py - Adaptive Cascaded Filter and Dihedral Structural Veto for turtlebot_tracker.

Implements adaptive density hull volume/solidity bounds, Pratt/Taubin 2D crescent arc
fitting rescue for concave point clouds, priority 2-plane dihedric RANSAC structural veto,
and global volumetricity checks.
"""

import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
from scipy.spatial import ConvexHull
import open3d as o3d

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

        # --- Stage 0.5: Canonical volume bound ---
        self.v_hull_max = cfg.get("v_hull_max", 0.45)  # fallback
        self.kappa_vol = cfg.get("kappa_vol", 1.2)

        canonical_path = Path("config/canonical_turtlebot2.json")
        if canonical_path.exists():
            try:
                with open(canonical_path, 'r') as f:
                    data = json.load(f)
                # Compute hull volume from canonical points (if available)
                points_path = Path("config/canonical_points.json")
                if points_path.exists():
                    with open(points_path, 'r') as f:
                        pts_data = json.load(f)
                    pts = np.array(pts_data["canonical_points"])
                    if len(pts) > 10:
                        hull = ConvexHull(pts)
                        V_canon = float(hull.volume)
                        self.v_hull_max = self.kappa_vol * V_canon
                        print(f"[INFO] v_hull_max set from canonical model: {self.v_hull_max:.4f} m³ "
                              f"(κ={self.kappa_vol}, V_canon={V_canon:.4f})")
            except Exception as e:
                print(f"[WARN] Could not load canonical model: {e}. Using fallback v_hull_max.")

        # Arc rescue
        self.taubin_r_min = cfg.get("taubin_radius_min", 0.14)
        self.taubin_r_max = cfg.get("taubin_radius_max", 0.22)
        self.taubin_rms_max = cfg.get("taubin_rms_max", 0.025)
        self.enable_arc_rescue = cfg.get("enable_arc_rescue", False)

    def filter_candidates(self, clusters: List[ClusterCandidate]) -> List[ClusterCandidate]:
        """
        Evaluates the adaptive cascaded filter on segmented clusters.

        Args:
            clusters: List of ClusterCandidate dataclass instances.

        Returns:
            Updated list of ClusterCandidate instances with passed_filters status set.
        """
        for cand in clusters:
            pts = cand.points
            num_points = len(pts)

            if num_points < 12:
                cand.passed_filters = False
                continue

            # Radial distance to cluster centroid
            r_centroid = float(np.linalg.norm(cand.centroid))

            # 1. Adaptive Hull Bounds
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

            hull_pass = ((v_min_adaptive <= cand.v_hull <= self.v_hull_max) and
                         (cand.solidity >= rho_sol_adaptive))

            # Taubin 2D Crescent Arc Rescue (Fallback for concave clouds)
            _, _, r_fit, rms_fit = self._fit_taubin_circle_2d(pts[:, :2])
            cand.is_arc_valid = ((self.taubin_r_min <= r_fit <= self.taubin_r_max) and
                                 (rms_fit <= self.taubin_rms_max))

            if not hull_pass and cand.is_arc_valid:
                hull_pass = True  # Rescued by 2D crescent arc geometry

            # 2. Priority Dihedric 2-Plane RANSAC Structural Veto
            cand.rho_2p, cand.is_corner = self._evaluate_dihedral_test(pts)

            # 3. Global Volumetricity & Soft Extents
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
        """
        Fits a 2D algebraic circle (Kåsa/Pratt method) to the XY projection.

        Args:
            points_2d: Nx2 float array of XY coordinates.

        Returns:
            Tuple of (center_a, center_b, radius_fit, rms_residual).
        """
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
        """
        Evaluates priority 2-plane RANSAC fit for architectural corners.

        Returns:
            Tuple of (rho_2p_fraction, is_corner_boolean).
        """
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