"""
segmentation.py - O(N) range‑image segmentation (Bogoslavskyi & Stachniss, 2016).
"""

import numpy as np
from typing import List
from ..datatypes import FrameData, ClusterCandidate

class RangeImageSegmenter:
    def __init__(self, config: dict):
        cfg = config["segmentation"]
        self.azimuth_bins = cfg["azimuth_bins"]
        self.elevation_bins = cfg["elevation_bins"]
        self.beta_threshold = np.radians(cfg["beta_threshold_deg"])

    def segment(self, frame_data: FrameData) -> List[ClusterCandidate]:
        points = frame_data.obstacle_points
        intensity = frame_data.intensity
        if points is None or len(points) < 15:
            return []

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.linalg.norm(points, axis=1)
        azimuth = np.arctan2(y, x)
        elevation = np.arcsin(np.clip(z / np.maximum(r, 1e-6), -1.0, 1.0))

        u = np.clip(((azimuth + np.pi) / (2 * np.pi) * self.azimuth_bins).astype(int), 0, self.azimuth_bins - 1)
        v = np.clip(((elevation + np.pi / 2) / np.pi * self.elevation_bins).astype(int), 0, self.elevation_bins - 1)

        grid = -np.ones((self.elevation_bins, self.azimuth_bins), dtype=int)
        range_grid = np.zeros((self.elevation_bins, self.azimuth_bins))

        for idx in range(len(points)):
            grid[v[idx], u[idx]] = idx
            range_grid[v[idx], u[idx]] = r[idx]

        visited = np.zeros(len(points), dtype=bool)
        clusters = []
        cluster_id = 0

        for idx in range(len(points)):
            if visited[idx]:
                continue

            cluster_indices = []
            queue = [idx]
            visited[idx] = True

            while queue:
                curr = queue.pop(0)
                cluster_indices.append(curr)
                cu, cv = u[curr], v[curr]
                d1 = r[curr]

                # 8‑neighbourhood
                for du in (-1, 0, 1):
                    for dv in (-1, 0, 1):
                        if du == 0 and dv == 0:
                            continue
                        nu = (cu + du) % self.azimuth_bins
                        nv = cv + dv
                        if 0 <= nv < self.elevation_bins:
                            neighbor_idx = grid[nv, nu]
                            if neighbor_idx != -1 and not visited[neighbor_idx]:
                                d2 = range_grid[nv, nu]
                                delta_alpha = 2 * np.pi / self.azimuth_bins
                                beta = np.arctan2(d2 * np.sin(delta_alpha),
                                                  np.abs(d1 - d2 * np.cos(delta_alpha)))
                                if beta > self.beta_threshold:
                                    visited[neighbor_idx] = True
                                    queue.append(neighbor_idx)

            if len(cluster_indices) >= 15:
                c_pts = points[cluster_indices]
                c_int = intensity[cluster_indices] if len(intensity) == len(points) else np.ones(len(cluster_indices)) * 100.0
                candidates = ClusterCandidate(
                    id=cluster_id,
                    points=c_pts,
                    intensity=c_int,
                    centroid=np.mean(c_pts, axis=0)
                )
                clusters.append(candidates)
                cluster_id += 1

        return clusters