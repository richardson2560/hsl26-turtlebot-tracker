import numpy as np
import open3d as o3d
from typing import Tuple

class PointCloudPreprocessor:
    """Handles voxel filtering and RANSAC ground plane segmentation."""
    
    def __init__(self, voxel_size: float = 0.03, distance_threshold: float = 0.08):
        self.voxel_size = voxel_size
        self.distance_threshold = distance_threshold

    def process(self, raw_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (obstacle_points, ground_points)."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_points)
        
        # 1. Voxel Downsampling
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        if len(pcd_down.points) < 50:
            return np.empty((0, 3)), np.empty((0, 3))

        # 2. RANSAC Ground Plane Fitting
        plane_model, inliers = pcd_down.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=3,
            num_iterations=150
        )
        
        obstacle_pcd = pcd_down.select_by_index(inliers, invert=True)
        ground_pcd = pcd_down.select_by_index(inliers, invert=False)
        
        return np.asarray(obstacle_pcd.points), np.asarray(ground_pcd.points)