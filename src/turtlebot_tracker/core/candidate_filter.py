"""
candidate_filter.py - Cascaded filter: convex hull (hard), dihedral (hard/priority),
volumetricity + soft extent likelihood.
"""

import numpy as np
from scipy.spatial import ConvexHull
import open3d as o3d
from typing import List
from ..datatypes import ClusterCandidate

class CandidateFilter:
    def __init__(self, config: dict):
        cfg = config["candidate_filter"]
        self.v_hull_min = cfg["v_hull_min"]
        self.v_hull_max = cfg["v_hull_max"]
        self.solidity_min = cfg["solidity_min"]
        self.dihedral_rho2p_thresh = cfg["dihedral_rho2p_threshold"]
        self.dihedral_angle_min = np.radians(cfg["dihedral_angle_min_deg"])
        self.dihedral_angle_max = np.radians(cfg["dihedral_angle_max_deg"])
        self.volumetricity_min = cfg["volumetricity_min"]
        # Soft extent parameters
        self.ext_x_mu, self.ext_x_sigma = cfg["ext_x_mu"], cfg["ext_x_sigma"]
        self.ext_y_mu, self.ext_y_sigma = cfg["ext_y_mu"], cfg["ext_y_sigma"]
        self.ext_z_mu, self.ext_z_sigma = cfg["ext_z_mu"], cfg["ext_z_sigma"]

    def filter_candidates(self, clusters: List[ClusterCandidate]) -> List[ClusterCandidate]:
        for cand in clusters:
            pts = cand.points
            if len(pts) < 15:
                cand.passed_filters = False
                continue

            # Stage 3a: QuickHull Volume & Solidity
            try:
                hull = ConvexHull(pts)
                cand.v_hull = float(hull.volume)
                cand.solidity = len(pts) / np.maximum(cand.v_hull, 1e-6)
            except Exception:
                cand.v_hull = 0.0
                cand.solidity = 0.0

            hull_pass = (self.v_hull_min <= cand.v_hull <= self.v_hull_max) and (cand.solidity >= self.solidity_min)

            # Stage 3b: Dihedral 2-plane RANSAC test
            cand.rho_2p, cand.is_corner = self._evaluate_dihedral_test(pts)

            # Stage 3c: Extents (soft) & Volumetricity
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
                (cand.extent_likelihood > -10.0)   # soft threshold
            )

        return clusters

    def _evaluate_dihedral_test(self, points: np.ndarray) -> tuple[float, bool]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        plane1, inliers1 = pcd.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=50)
        if len(inliers1) < 5:
            return 0.0, False

        remainder = pcd.select_by_index(inliers1, invert=True)
        if len(remainder.points) < 5:
            return float(len(inliers1) / len(points)), False

        plane2, inliers2 = remainder.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=50)

        rho_2p = (len(inliers1) + len(inliers2)) / len(points)
        n1 = plane1[:3] / np.linalg.norm(plane1[:3])
        n2 = plane2[:3] / np.linalg.norm(plane2[:3])
        angle = np.arccos(np.clip(np.abs(np.dot(n1, n2)), 0.0, 1.0))

        is_corner = (rho_2p >= self.dihedral_rho2p_thresh) and (self.dihedral_angle_min <= angle <= self.dihedral_angle_max)
        return float(rho_2p), is_corner