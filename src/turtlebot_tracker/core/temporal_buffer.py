"""
temporal_buffer.py - Memory-Leak Free Motion-Compensated Temporal Accumulator in se(2).
"""

from collections import deque
from typing import List, Tuple
import numpy as np
import open3d as o3d

from turtlebot_tracker.datatypes import ClusterCandidate


class MotionCompensatedBuffer:
    """Sliding-window buffer compensating target motion using Lie algebra se(2)."""

    def __init__(self, config: dict):
        cfg = config.get("temporal_buffer", {})
        self.window_size = cfg.get("window_size", 5)
        self.min_points_trigger = cfg.get("min_points_trigger", 30)
        self.max_accumulated_points = 300  # Strict cap to prevent memory leaks

        self.buffer = deque(maxlen=self.window_size)

    def add_frame_candidates(self, candidates: List[ClusterCandidate], timestamp: float) -> None:
        """Stores clean, un-accumulated single-frame copies into buffer."""
        if not candidates:
            return

        all_pts = [cand.points.copy() for cand in candidates]
        all_int = [cand.intensity.copy() for cand in candidates]

        if all_pts:
            concat_pts = np.vstack(all_pts)
            concat_int = np.concatenate(all_int)
            self.buffer.append((timestamp, concat_pts, concat_int))

    def get_accumulated_points(
        self,
        current_candidate: ClusterCandidate,
        target_velocity_se2: np.ndarray,
        current_timestamp: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Accumulates and motion-compensates historical point clouds.
        Downsamples output to N <= 300 points if needed.
        """
        num_curr = len(current_candidate.points)
        if num_curr >= self.min_points_trigger or len(self.buffer) <= 1:
            return current_candidate.points.copy(), current_candidate.intensity.copy()

        accum_pts = [current_candidate.points.copy()]
        accum_int = [current_candidate.intensity.copy()]

        vx, vy, omega = target_velocity_se2

        for ts, pts, intensity in self.buffer:
            tau = current_timestamp - ts
            if tau <= 1e-4 or tau > 1.0:
                continue

            delta_psi = -omega * tau
            cos_psi = np.cos(delta_psi)
            sin_psi = np.sin(delta_psi)

            R_comp = np.array([
                [cos_psi, -sin_psi, 0.0],
                [sin_psi,  cos_psi, 0.0],
                [0.0,          0.0, 1.0]
            ], dtype=np.float64)

            t_comp = np.array([-vx * tau, -vy * tau, 0.0], dtype=np.float64)

            pts_comp = (R_comp @ pts.T).T + t_comp
            accum_pts.append(pts_comp)
            accum_int.append(intensity)

        raw_accum_pts = np.vstack(accum_pts)
        raw_accum_int = np.concatenate(accum_int)

        if len(raw_accum_pts) > self.max_accumulated_points:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(raw_accum_pts)
            pcd_down, _, trace = pcd.voxel_down_sample_and_trace(
                0.02, pcd.get_min_bound(), pcd.get_max_bound()
            )
            down_pts = np.asarray(pcd_down.points, dtype=np.float64)
            down_indices = [t[0] for t in trace if len(t) > 0]
            down_int = (raw_accum_int[down_indices]
                        if len(down_indices) == len(down_pts)
                        else np.ones(len(down_pts), dtype=np.float64) * 100.0)
            return down_pts, down_int

        return raw_accum_pts, raw_accum_int

    def clear(self) -> None:
        self.buffer.clear()