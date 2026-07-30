"""
static_map.py - Static Background Model (Ground + Splats + Fallback Planes).
Supports loading from JSON prior, online RANSAC fallback, and O(N·W) veto.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import open3d as o3d

from turtlebot_tracker.core.hierarchical_gmm import HierarchicalGMM
from turtlebot_tracker.datatypes import SemanticLabel, StaticMapPrimitives


class StaticBackgroundMap:
    """
    Static background model: ground plane + optional wall planes (fallback) + splats.
    """

    def __init__(self, config: dict):
        cfg = config.get("static_map", {})
        self.delta_bg = cfg.get("bg_veto_distance", 0.05)
        self.mahalanobis_threshold = cfg.get("mahalanobis_threshold", 11.34)
        self.is_initialized = False

        # Ground plane (always present)
        self.R_align = np.eye(3, dtype=np.float64)
        self.ground_z = 0.0
        self.ground_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.ground_distance = 0.0

        # Wall planes (fallback, if prior not available)
        self.wall_planes: List[Tuple[np.ndarray, float]] = []

        # Static structure splats (from offline MVI)
        self.splats: List[Dict] = []
        self.hierarchical_gmm: Optional[HierarchicalGMM] = None
        self.K = 0

        # Primitives for compatibility with old static_veto
        self.primitives = StaticMapPrimitives()

    def load_from_json(self, json_path: Path) -> bool:
        """Load static prior from JSON (ground + splats)."""
        if not json_path.exists():
            return False

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.R_align = np.array(data.get("R_align", np.eye(3)), dtype=np.float64)
        self.ground_z = data.get("z_ground", 0.0)
        self.ground_normal = np.array(data.get("ground_normal", [0, 0, 1]), dtype=np.float64)
        self.ground_distance = data.get("ground_distance", 0.0)

        self.wall_planes = []
        for wall in data.get("wall_planes", []):
            n = np.array(wall["normal"], dtype=np.float64)
            d = float(wall["distance"])
            self.wall_planes.append((n, d))

        self.splats = data.get("splats", [])
        if self.splats:
            self.hierarchical_gmm = HierarchicalGMM(self.splats)
            self.K = len(self.splats)
            print(f"[INFO] Loaded {self.K} static structure splats.")
        else:
            self.K = 0

        self._update_primitives()
        self.is_initialized = True
        return True

    def update_from_frame(self, ground_normal: np.ndarray, ground_d: float,
                          wall_planes: Optional[List[Tuple[np.ndarray, float]]] = None) -> None:
        """
        Update ground and wall planes from online RANSAC (fallback mode).
        This is used when no prior is available.
        """
        self.ground_normal = ground_normal / np.linalg.norm(ground_normal)
        self.ground_distance = ground_d
        self.ground_z = 0.0  # Will be set later by preprocessor from actual ground points
        self.wall_planes = wall_planes or []
        self._update_primitives()
        self.is_initialized = True

    def _update_primitives(self) -> None:
        """Update the underlying StaticMapPrimitives for compatibility."""
        normals = [self.ground_normal]
        distances = [self.ground_distance]
        for n, d in self.wall_planes:
            normals.append(n)
            distances.append(d)
        self.primitives.normals = np.array(normals, dtype=np.float64)
        self.primitives.distances = np.array(distances, dtype=np.float64)
        self.primitives.is_initialized = True

    def apply_veto(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply static background veto: ground + wall planes + splats.

        Args:
            points: Nx3 array of XYZ points in world Z-up frame.

        Returns:
            obs_mask: True for non-background (obstacle) points.
            bg_mask: True for background points (ground/walls).
            semantic_labels: integer labels (GROUND, STRUCTURE_WALL, CANDIDATE_FREE).
        """
        N = len(points)
        if N == 0 or not self.is_initialized:
            return (np.ones(N, dtype=bool),
                    np.zeros(N, dtype=bool),
                    np.full(N, SemanticLabel.CANDIDATE_FREE, dtype=np.int32))

        bg_mask = np.zeros(N, dtype=bool)

        # 1. Ground veto (by Z)
        z_vals = points[:, 2]
        ground_mask = np.abs(z_vals - self.ground_z) <= self.delta_bg * 1.5
        bg_mask |= ground_mask

        # 2. Wall planes veto (if any)
        if self.wall_planes:
            normals = np.array([n for n, _ in self.wall_planes], dtype=np.float64)
            distances = np.array([d for _, d in self.wall_planes], dtype=np.float64)
            dist_mat = np.abs(points @ normals.T + distances)
            min_dist = np.min(dist_mat, axis=1)
            wall_mask = min_dist <= self.delta_bg
            bg_mask |= wall_mask

        # 3. Static structure splats veto (Mahalanobis distance)
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

        # Assign semantic labels
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