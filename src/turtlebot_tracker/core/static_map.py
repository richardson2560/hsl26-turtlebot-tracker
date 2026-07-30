"""
static_map.py - Static Background Model (Bounded Walls + Splats).
Supports bounded wall planes (finite rectangles) and infinite planes (ground).
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import open3d as o3d

from turtlebot_tracker.core.hierarchical_gmm import HierarchicalGMM
from turtlebot_tracker.datatypes import SemanticLabel, StaticMapPrimitives


class StaticBackgroundMap:
    def __init__(self, config: dict):
        cfg = config.get("static_map", {})
        self.delta_bg = cfg.get("bg_veto_distance", 0.05)
        self.mahalanobis_threshold = cfg.get("mahalanobis_threshold", 11.34)
        self.is_initialized = False

        # Ground plane (infinite)
        self.R_align = np.eye(3, dtype=np.float64)
        self.ground_z = 0.0
        self.ground_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.ground_distance = 0.0

        # Wall planes (bounded)
        self.wall_planes: List[Dict] = []

        # Splats (volumetric structures)
        self.splats: List[Dict] = []
        self.hierarchical_gmm: Optional[HierarchicalGMM] = None
        self.K = 0

        self.primitives = StaticMapPrimitives()

    def load_from_json(self, json_path: Path) -> bool:
        if not json_path.exists():
            return False

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.R_align = np.array(data.get("R_align", np.eye(3)), dtype=np.float64)
        self.ground_z = data.get("z_ground", 0.0)
        self.ground_normal = np.array(data.get("ground_normal", [0, 0, 1]), dtype=np.float64)
        self.ground_distance = data.get("ground_distance", 0.0)

        self.wall_planes = data.get("wall_planes", [])
        self.splats = data.get("splats", [])
        if self.splats:
            self.hierarchical_gmm = HierarchicalGMM(self.splats)
            self.K = len(self.splats)

        self._update_primitives()
        self.is_initialized = True
        return True

    def _update_primitives(self) -> None:
        normals = [self.ground_normal]
        distances = [self.ground_distance]
        for wall in self.wall_planes:
            normals.append(np.array(wall["normal"]))
            distances.append(wall["distance"])
        self.primitives.normals = np.array(normals, dtype=np.float64)
        self.primitives.distances = np.array(distances, dtype=np.float64)
        self.primitives.is_initialized = True

    def _point_in_wall(self, point: np.ndarray, wall: Dict) -> bool:
        """Check if a 3D point is within the bounded wall rectangle."""
        # 1. Distance to plane
        normal = np.array(wall["normal"])
        dist = np.dot(point, normal) + wall["distance"]
        if abs(dist) > self.delta_bg:
            return False

        # 2. If no bounds (fallback to infinite plane)
        if "center" not in wall:
            return True

        # 3. Project point onto plane
        center = np.array(wall["center"])
        u = np.array(wall["u"])
        v = np.array(wall["v"])
        half_w = wall["half_width"]
        half_h = wall["half_height"]

        diff = point - center
        # Remove normal component
        diff_plane = diff - np.dot(diff, normal) * normal

        # Local coordinates
        coord_u = np.dot(diff_plane, u)
        coord_v = np.dot(diff_plane, v)

        return abs(coord_u) <= half_w and abs(coord_v) <= half_h

    def apply_veto(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = len(points)
        if N == 0 or not self.is_initialized:
            return (np.ones(N, dtype=bool),
                    np.zeros(N, dtype=bool),
                    np.full(N, SemanticLabel.CANDIDATE_FREE, dtype=np.int32))

        bg_mask = np.zeros(N, dtype=bool)

        # 1. Ground veto (infinite plane)
        z_vals = points[:, 2]
        ground_mask = np.abs(z_vals - self.ground_z) <= self.delta_bg * 1.5
        bg_mask |= ground_mask

        # 2. Bounded wall veto
        for wall in self.wall_planes:
            normal = np.array(wall["normal"])
            center = np.array(wall.get("center", [0, 0, 0]))
            # Vectorized check for efficiency
            # Compute plane distances
            dists = points @ normal + wall["distance"]
            plane_hit = np.abs(dists) <= self.delta_bg

            if "center" not in wall:
                # Infinite plane fallback
                bg_mask |= plane_hit
                continue

            # Bounded check for points that passed plane distance
            # Only evaluate points that are near the plane
            idx_near = np.where(plane_hit)[0]
            if len(idx_near) == 0:
                continue

            u = np.array(wall["u"])
            v = np.array(wall["v"])
            half_w = wall["half_width"]
            half_h = wall["half_height"]

            for idx in idx_near:
                point = points[idx]
                diff = point - center
                diff_plane = diff - np.dot(diff, normal) * normal
                coord_u = np.dot(diff_plane, u)
                coord_v = np.dot(diff_plane, v)
                if abs(coord_u) <= half_w and abs(coord_v) <= half_h:
                    bg_mask[idx] = True

        # 3. Splat veto (Mahalanobis)
        if self.K > 0 and self.splats:
            mu = np.array([s['mu'] for s in self.splats], dtype=np.float64)
            dist_min = np.full(N, np.inf, dtype=np.float64)
            for k in range(self.K):
                mu_k = mu[k]
                cov_k = np.array(self.splats[k]['cov'], dtype=np.float64)
                inv_cov = np.linalg.inv(cov_k + np.eye(3) * 1e-6)
                diff = points - mu_k
                maha = np.sum(diff @ inv_cov * diff, axis=1)
                dist_min = np.minimum(dist_min, maha)
            structure_mask = dist_min <= self.mahalanobis_threshold
            bg_mask |= structure_mask

        # Labels
        semantic_labels = np.full(N, SemanticLabel.CANDIDATE_FREE, dtype=np.int32)
        semantic_labels[ground_mask] = SemanticLabel.GROUND
        structure_only = bg_mask & ~ground_mask
        semantic_labels[structure_only] = SemanticLabel.STRUCTURE_WALL

        obs_mask = ~bg_mask
        return obs_mask, bg_mask, semantic_labels

    def get_ground_z(self) -> float:
        return self.ground_z

    def get_R_align(self) -> np.ndarray:
        return self.R_align.copy()