"""
static_map.py - Static Background Model (Bounded Walls + Ground).
Splats have been removed entirely.
Compatible with preprocessor.py (update_from_frame included).
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import open3d as o3d

from turtlebot_tracker.datatypes import SemanticLabel, StaticMapPrimitives


class StaticBackgroundMap:
    def __init__(self, config: dict):
        cfg = config.get("static_map", {})
        self.delta_bg = cfg.get("bg_veto_distance", 0.05)
        self.mahalanobis_threshold = cfg.get("mahalanobis_threshold", 11.34)  # No usado, se mantiene por compatibilidad
        self.is_initialized = False

        # Ground plane (infinite)
        self.R_align = np.eye(3, dtype=np.float64)
        self.ground_z = 0.0
        self.ground_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.ground_distance = 0.0

        # Wall planes (bounded)
        self.wall_planes: List[Dict] = []

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

        self._update_primitives()
        self.is_initialized = True
        return True

    def update_from_frame(self, ground_normal: np.ndarray, ground_distance: float, wall_planes: List[Dict] = None):
        """
        Actualiza el mapa estático a partir de un frame individual (modo fallback).
        Calcula ground_z a partir del plano y almacena los muros (si se proporcionan).
        """
        self.ground_normal = np.array(ground_normal, dtype=np.float64)
        self.ground_distance = float(ground_distance)
        # Plano: n·p + d = 0 -> z = -d / n_z
        if abs(self.ground_normal[2]) > 1e-6:
            self.ground_z = -self.ground_distance / self.ground_normal[2]
        else:
            self.ground_z = 0.0

        self.wall_planes = wall_planes if wall_planes is not None else []

        self._update_primitives()
        self.is_initialized = True

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