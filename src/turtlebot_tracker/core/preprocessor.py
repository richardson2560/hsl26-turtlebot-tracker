"""
preprocessor.py - Preprocessor with Static Background Prior (Stage 0.5).
If static prior exists, uses fixed R_align and static veto. Falls back to RANSAC otherwise.
"""

import json
from pathlib import Path
import numpy as np
import open3d as o3d

from turtlebot_tracker.core.static_map import StaticBackgroundMap
from turtlebot_tracker.datatypes import FrameData, SemanticLabel


class LiDARPreprocessor:
    def __init__(self, config: dict):
        cfg = config.get("preprocessing", {})
        self.voxel_size = cfg.get("voxel_size", 0.03)
        self.distance_threshold = cfg.get("ransac_distance_threshold", 0.08)
        self.ransac_iterations = cfg.get("ransac_iterations", 100)
        self.blind_spot_radius = cfg.get("blind_spot_radius", 0.35)

        # Static background map (Stage 0.5)
        self.static_map = StaticBackgroundMap(config)
        prior_path = Path("config/static_map_prior.json")
        self.use_prior = self.static_map.load_from_json(prior_path)

        if self.use_prior:
            self.R_align_fixed = self.static_map.get_R_align()
            self.z_ground_fixed = self.static_map.get_ground_z()
            print(f"[INFO] Using static map prior (ground_z={self.z_ground_fixed:.3f}m, "
                  f"walls={len(self.static_map.wall_planes)})")
        else:
            print("[INFO] No static map prior found. Falling back to online RANSAC.")

    def process(self, timestamp: float, raw_points: np.ndarray, intensity: np.ndarray) -> FrameData:
        if raw_points is None or len(raw_points) == 0:
            return FrameData(
                timestamp=timestamp,
                raw_points=np.empty((0, 3), dtype=np.float32),
                intensity=np.empty((0,), dtype=np.float32)
            )

        # Blind spot filter
        r_xy = np.linalg.norm(raw_points[:, :2], axis=1)
        valid_mask = r_xy >= self.blind_spot_radius
        filtered_raw = raw_points[valid_mask]
        filtered_intensity = (intensity[valid_mask]
                              if len(intensity) == len(raw_points)
                              else np.ones(len(filtered_raw), dtype=np.float32) * 100.0)

        if len(filtered_raw) < 30:
            return FrameData(timestamp=timestamp, raw_points=filtered_raw, intensity=filtered_intensity)

        # Voxel downsampling
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_raw)
        pcd_down, _, trace = pcd.voxel_down_sample_and_trace(
            self.voxel_size, pcd.get_min_bound(), pcd.get_max_bound()
        )
        down_points = np.asarray(pcd_down.points, dtype=np.float32)
        down_intensity = np.zeros(len(down_points), dtype=np.float32)
        for i, idx_list in enumerate(trace):
            if len(idx_list) > 0:
                down_intensity[i] = np.mean(filtered_intensity[idx_list])
            else:
                down_intensity[i] = 100.0

        if len(down_points) < 30:
            return FrameData(timestamp=timestamp, raw_points=filtered_raw, intensity=filtered_intensity)

        # --- Stage 0.5: Static Prior Mode ---
        if self.use_prior:
            # 1. Align with fixed R_align
            aligned_down = (self.R_align_fixed @ down_points.T).T

            # 2. Apply static veto (ground + walls)
            obs_mask, bg_mask, semantic_labels = self.static_map.apply_veto(aligned_down)

            # 3. Extract obstacle and ground points
            obstacle_points = aligned_down[obs_mask]
            obstacle_intensity = down_intensity[obs_mask]
            ground_points = aligned_down[bg_mask]  # includes ground and walls

            # 4. Compute z_ground (fixed)
            z_ground_val = self.z_ground_fixed

            # 5. Optional: check residual to detect if prior is invalid (fallback trigger)
            if len(ground_points) > 0:
                # If residual is large, force fallback
                ground_z_actual = np.mean(aligned_down[bg_mask][:, 2])
                if abs(ground_z_actual - self.z_ground_fixed) > self.distance_threshold * 2:
                    print(f"[WARN] Prior ground drift detected ({ground_z_actual:.3f} vs {self.z_ground_fixed:.3f}). "
                          "Falling back to RANSAC.")
                    self.use_prior = False  # Disable prior for this frame, fallback below

        # --- Fallback: Online RANSAC + Rodrigues (original behavior) ---
        if not self.use_prior:
            # Original RANSAC ground plane extraction
            pcd_down_seg = o3d.geometry.PointCloud()
            pcd_down_seg.points = o3d.utility.Vector3dVector(down_points)
            plane_model, inliers = pcd_down_seg.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=3,
                num_iterations=self.ransac_iterations
            )

            ground_normal = np.array(plane_model[:3], dtype=np.float64)
            ground_d = float(plane_model[3])
            normal_norm = np.linalg.norm(ground_normal)
            if normal_norm > 1e-6:
                ground_normal /= normal_norm
                ground_d /= normal_norm

            if ground_normal[2] < 0:
                ground_normal = -ground_normal
                ground_d = -ground_d

            target_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            v = np.cross(ground_normal, target_up)
            s = np.linalg.norm(v)
            c = np.dot(ground_normal, target_up)

            if s < 1e-6:
                R_align = np.eye(3, dtype=np.float64)
            else:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
                R_align = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s ** 2))

            aligned_down = (R_align @ down_points.T).T

            # Update static map with this frame's ground (for compatibility)
            self.static_map.update_from_frame(ground_normal, ground_d, wall_planes=[])

            # Compute semantic labels (only ground, no walls)
            ground_mask = np.zeros(len(aligned_down), dtype=bool)
            ground_mask[inliers] = True
            semantic_labels = np.full(len(aligned_down), SemanticLabel.CANDIDATE_FREE, dtype=np.int32)
            semantic_labels[ground_mask] = SemanticLabel.GROUND

            obstacle_points = aligned_down[~ground_mask]
            obstacle_intensity = down_intensity[~ground_mask]
            ground_points = aligned_down[ground_mask]
            z_ground_val = float(np.mean(ground_points[:, 2])) if len(ground_points) > 0 else 0.0

        # Build FrameData
        return FrameData(
            timestamp=timestamp,
            raw_points=(self.R_align_fixed @ filtered_raw.T).T if self.use_prior else (R_align @ filtered_raw.T).T,
            intensity=obstacle_intensity,
            ground_points=ground_points,
            obstacle_points=obstacle_points,
            R_align=self.R_align_fixed if self.use_prior else R_align,
            z_ground=z_ground_val,
            semantic_labels=semantic_labels
        )