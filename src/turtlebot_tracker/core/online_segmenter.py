"""
online_segmenter.py - Expert Online Veto with Anisotropic Padding and Z-Gating.
Fixed AxisError in _is_inside_stretched_shell.
"""

import numpy as np
import open3d as o3d

class OnlineSegmenter:
    def __init__(self, prior_data, metadata, 
                 blind_spot_radius=0.45, 
                 iou_threshold=0.6,
                 robot_max_height=0.55):
        
        # Load Alignment and Ground Context
        self.R_align = np.array(metadata['R_align'])
        self.z_ground = metadata['z_ground']
        self.ground_thick = metadata['ground_thickness']
        
        # Load Static Memory
        self.shells = prior_data.get('shells', [])
        self.static_splats = prior_data.get('static_splats', [])
        
        # Thresholds
        self.blind_spot_radius = blind_spot_radius
        self.iou_threshold = iou_threshold
        self.robot_max_height = robot_max_height
        
        # Anisotropic Padding Parameters
        self.point_margin = 0.03
        self.thickness_margin = 0.04
        self.continuity_padding = 0.15

    def _is_inside_stretched_shell(self, local_pts, extents):
        """Checks if points are inside a shell stretched to close topology gaps."""
        # Ensure local_pts is 2D (N, 3)
        pts = np.atleast_2d(local_pts)
        
        thickness_axis = np.argmin(extents)
        stretched_ext = np.array(extents) + self.continuity_padding
        stretched_ext[thickness_axis] = extents[thickness_axis] + self.thickness_margin
        
        # Check bounds
        in_box = np.all(np.abs(pts) < stretched_ext, axis=1)
        return in_box

    def _match_static_splat(self, cluster_pts, cluster_centroid):
        """Checks if a new cluster is actually a known static object from the prior."""
        if not self.static_splats: return False
        for splat in self.static_splats:
            mu = np.array(splat['mu'])
            dist = np.linalg.norm(cluster_centroid - mu)
            if dist > 0.4: continue
            
            splat_max_scale = np.max(splat['scales'])
            cluster_max_extent = np.max(np.max(cluster_pts, axis=0) - np.min(cluster_pts, axis=0)) / 2.0
            if abs(splat_max_scale - cluster_max_extent) < 0.2:
                return True
        return False

    def classify_and_cluster(self, raw_points):
        # 1. Alignment
        pts_world = (self.R_align @ raw_points.T).T
        n_pts = len(pts_world)
        labels = np.full(n_pts, 3, dtype=np.int32) 

        # 2. Blind Spot
        dist_sq = np.sum(pts_world[:, :2]**2, axis=1)
        labels[dist_sq < (self.blind_spot_radius**2)] = -1

        # 3. Ground Veto
        ground_mask = (np.abs(pts_world[:, 2] - self.z_ground) < (self.ground_thick + self.point_margin)) & (labels != -1)
        labels[ground_mask] = 0

        # 4. Structural Shell Veto
        target_indices = np.where(labels == 3)[0]
        if len(target_indices) > 0:
            rem_pts = pts_world[target_indices]
            for shell in self.shells:
                d = rem_pts - np.array(shell['center'])
                local_pts = d @ np.array(shell['axes'])
                # Call refined function
                in_box = self._is_inside_stretched_shell(local_pts, shell['extents'])
                labels[target_indices[in_box]] = 1

        # 5. Residual Clustering
        target_mask = (labels == 3)
        target_idx_global = np.where(target_mask)[0]
        target_pts = pts_world[target_mask]

        if len(target_pts) < 12:
            return pts_world, labels, []

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_pts))
        res_labels = np.array(pcd.cluster_dbscan(eps=0.25, min_points=12))

        # 6. Refinement
        unique_ids = np.unique(res_labels)
        potential_clusters = []
        
        for cid in unique_ids:
            if cid == -1: continue
            c_mask = (res_labels == cid)
            c_indices = target_idx_global[c_mask]
            c_pts = target_pts[c_mask]
            c_centroid = np.mean(c_pts, axis=0)

            if (c_centroid[2] - self.z_ground) > self.robot_max_height:
                labels[c_indices] = 1
                continue

            pts_near_wall = 0
            for shell in self.shells:
                d = c_pts - np.array(shell['center'])
                local = d @ np.array(shell['axes'])
                in_box = self._is_inside_stretched_shell(local, shell['extents'])
                pts_near_wall += np.sum(in_box)
                if pts_near_wall / len(c_pts) > self.iou_threshold: break
            
            if (pts_near_wall / len(c_pts)) > self.iou_threshold:
                labels[c_indices] = 1
                continue

            if self._match_static_splat(c_pts, c_centroid):
                labels[c_indices] = 2
                continue

            potential_clusters.append({
                "points": c_pts, "indices": c_indices, "centroid": c_centroid
            })

        return pts_world, labels, potential_clusters