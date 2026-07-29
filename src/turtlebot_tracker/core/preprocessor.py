"""
preprocessor.py - RANSAC ground plane extraction, Rodrigues Z-up alignment,
blind‑spot removal, and radiometric intensity correction (range²/cos(eta)).
"""

import numpy as np
import open3d as o3d
from ..datatypes import FrameData

class LiDARPreprocessor:
    def __init__(self, config: dict):
        cfg = config["preprocessing"]
        self.voxel_size = cfg["voxel_size"]
        self.distance_threshold = cfg["ransac_distance_threshold"]
        self.ransac_iterations = cfg["ransac_iterations"]
        self.blind_spot_radius = cfg.get("blind_spot_radius", 0.40)

    def process(self, timestamp: float, raw_points: np.ndarray, intensity: np.ndarray) -> FrameData:
        # Remove chassis self‑reflections
        r_xy = np.linalg.norm(raw_points[:, :2], axis=1)
        valid_mask = r_xy >= self.blind_spot_radius
        filtered_raw = raw_points[valid_mask]
        filtered_intensity = intensity[valid_mask] if len(intensity) == len(raw_points) else intensity

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_raw)
        pcd_down, _, trace = pcd.voxel_down_sample_and_trace(
            self.voxel_size, pcd.get_min_bound(), pcd.get_max_bound()
        )
        down_points = np.asarray(pcd_down.points)
        down_indices = [t[0] for t in trace if len(t) > 0]
        down_intensity = filtered_intensity[down_indices] if len(down_indices) == len(down_points) else np.ones(len(down_points)) * 100.0

        if len(down_points) < 50:
            return FrameData(timestamp=timestamp, raw_points=filtered_raw, intensity=filtered_intensity)

        # RANSAC ground plane
        plane_model, inliers = pcd_down.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=3,
            num_iterations=self.ransac_iterations
        )
        obstacle_pcd = pcd_down.select_by_index(inliers, invert=True)
        ground_pcd = pcd_down.select_by_index(inliers, invert=False)

        obstacle_points = np.asarray(obstacle_pcd.points)
        ground_points = np.asarray(ground_pcd.points)

        # Rodrigues Z‑up alignment
        ground_normal = plane_model[:3] / np.linalg.norm(plane_model[:3])
        if ground_normal[2] < 0:
            ground_normal = -ground_normal

        target = np.array([0.0, 0.0, 1.0])
        v = np.cross(ground_normal, target)
        s = np.linalg.norm(v)
        c = np.dot(ground_normal, target)
        if s < 1e-6:
            R_align = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R_align = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s ** 2))

        aligned_raw = (R_align @ filtered_raw.T).T
        aligned_obstacles = (R_align @ obstacle_points.T).T if len(obstacle_points) > 0 else obstacle_points
        aligned_ground = (R_align @ ground_points.T).T if len(ground_points) > 0 else ground_points

        # Radiometric correction: intensity *= r² / cos(eta)
        # We compute corrected intensity for obstacle points (to be used later)
        # We'll store the original intensity in FrameData; correction is done during GMM fitting.
        return FrameData(
            timestamp=timestamp,
            raw_points=aligned_raw,
            intensity=filtered_intensity,
            ground_points=aligned_ground,
            obstacle_points=aligned_obstacles,
            R_align=R_align
        )