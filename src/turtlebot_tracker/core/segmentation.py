"""
segmentation.py - Vectorized O(N) Range-Image Connected-Components Segmenter.
"""

from collections import deque
from typing import List
import numpy as np

from turtlebot_tracker.datatypes import ClusterCandidate, FrameData


class RangeImageSegmenter:
    """Segments non-ground obstacle point clouds into clusters in O(N) time."""

    def __init__(self, config: dict):
        cfg = config.get("segmentation", {})
        self.azimuth_bins = cfg.get("azimuth_bins", 360)
        self.elevation_bins = cfg.get("elevation_bins", 90)
        self.beta_threshold = np.radians(cfg.get("beta_threshold_deg", 10.0))

    def segment(self, frame_data: FrameData) -> List[ClusterCandidate]:
        points = frame_data.obstacle_points
        intensity = frame_data.intensity
        if points is None or len(points) < 12:
            return []

        num_points = len(points)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.linalg.norm(points, axis=1)

        azimuth = np.arctan2(y, x)
        elevation = np.arcsin(np.clip(z / np.maximum(r, 1e-6), -1.0, 1.0))

        u = np.clip(
            ((azimuth + np.pi) / (2.0 * np.pi) * self.azimuth_bins).astype(np.int32),
            0, self.azimuth_bins - 1
        )
        v = np.clip(
            ((elevation + np.pi / 2.0) / np.pi * self.elevation_bins).astype(np.int32),
            0, self.elevation_bins - 1
        )

        grid = np.full((self.elevation_bins, self.azimuth_bins), -1, dtype=np.int32)
        range_grid = np.zeros((self.elevation_bins, self.azimuth_bins), dtype=np.float32)

        grid[v, u] = np.arange(num_points, dtype=np.int32)
        range_grid[v, u] = r

        visited = np.zeros(num_points, dtype=bool)
        clusters = []
        cluster_id = 0
        delta_alpha = 2.0 * np.pi / self.azimuth_bins

        for idx in range(num_points):
            if visited[idx]:
                continue

            cluster_indices = []
            queue = deque([idx])
            visited[idx] = True

            while queue:
                curr = queue.popleft()
                cluster_indices.append(curr)
                cu, cv = u[curr], v[curr]

                for du in (-1, 0, 1):
                    for dv in (-1, 0, 1):
                        if du == 0 and dv == 0:
                            continue
                        nu = (cu + du) % self.azimuth_bins
                        nv = cv + dv
                        if 0 <= nv < self.elevation_bins:
                            neighbor_idx = grid[nv, nu]
                            if neighbor_idx != -1 and not visited[neighbor_idx]:
                                # CORRECTED: use max/min ranges
                                d_a, d_b = r[curr], range_grid[nv, nu]
                                d1, d2 = max(d_a, d_b), min(d_a, d_b)
                                beta = np.arctan2(
                                    d2 * np.sin(delta_alpha),
                                    d1 - d2 * np.cos(delta_alpha)
                                )
                                if beta > self.beta_threshold:
                                    visited[neighbor_idx] = True
                                    queue.append(neighbor_idx)

            if len(cluster_indices) >= 12:
                c_pts = points[cluster_indices]
                c_int = (intensity[cluster_indices]
                         if len(intensity) == num_points
                         else np.ones(len(cluster_indices), dtype=np.float32) * 100.0)

                candidate = ClusterCandidate(
                    id=cluster_id,
                    points=c_pts,
                    intensity=c_int,
                    centroid=np.mean(c_pts, axis=0)
                )
                clusters.append(candidate)
                cluster_id += 1

        return clusters