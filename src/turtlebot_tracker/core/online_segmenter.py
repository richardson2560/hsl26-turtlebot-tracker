"""
online_segmenter.py - Expert Online Veto with Anisotropic Padding and Z-Gating.
"""

import numpy as np
import open3d as o3d

class OnlineSegmenter:
    def __init__(self, prior_data, metadata, 
                 blind_spot_radius=0.4, 
                 iou_threshold=0.6,
                 robot_max_height=0.55): # Turtlebot2 is ~0.45m tall
        
        self.R_align = np.array(metadata['R_align'])
        self.z_ground = metadata['z_ground']
        self.ground_thick = metadata['ground_thickness']
        self.shells = prior_data.get('shells', [])
        self.splats = prior_data.get('static_splats', [])
        
        self.blind_spot_radius = blind_spot_radius
        self.iou_threshold = iou_threshold
        self.robot_max_height = robot_max_height
        
        # Expert Tuning for Cracks
        self.point_margin = 0.03       # Tight margin for point-level veto
        self.thickness_margin = 0.04   # Keep walls thin (Normal direction)
        self.continuity_padding = 0.12 # Stretch boxes to close cracks (Width/Height direction)

    def _is_inside_stretched_shell(self, local_pts, extents):
        """
        Applies anisotropic stretching to the box extents.
        Stretches the major axes (Width/Height) more than the minor axis (Thickness).
        """
        # Find which axis is the thickness (the smallest one)
        thickness_axis = np.argmin(extents)
        
        # Create stretched extents
        stretched_ext = np.array(extents) + self.continuity_padding
        # Restore original thickness (don't make walls fat)
        stretched_ext[thickness_axis] = extents[thickness_axis] + self.thickness_margin
        
        return np.all(np.abs(local_pts) < stretched_ext, axis=1)

    def classify_and_cluster(self, raw_points):
        # 1. World Alignment
        pts_world = (self.R_align @ raw_points.T).T
        n_pts = len(pts_world)
        labels = np.full(n_pts, 3, dtype=np.int32) 

        # 2. Radial Blind Spot
        dist_sq = np.sum(pts_world[:, :2]**2, axis=1)
        labels[dist_sq < (self.blind_spot_radius**2)] = -1

        # 3. Ground Veto (Infinite Plane)
        ground_mask = (np.abs(pts_world[:, 2] - self.z_ground) < (self.ground_thick + self.point_margin)) & (labels != -1)
        labels[ground_mask] = 0

        # 4. Structural Shell Veto (With Anisotropic Padding)
        target_indices = np.where(labels == 3)[0]
        if len(target_indices) > 0:
            rem_pts = pts_world[target_indices]
            for shell in self.shells:
                d = rem_pts - np.array(shell['center'])
                local_pts = d @ np.array(shell['axes'])
                extents = np.array(shell['extents'])
                
                # Check point against stretched box
                in_box = self._is_inside_stretched_shell(local_pts, extents)
                labels[target_indices[in_box]] = 1

        # 5. DBSCAN on residuals
        target_mask = (labels == 3)
        target_idx_global = np.where(target_mask)[0]
        target_pts = pts_world[target_mask]

        if len(target_pts) < 12:
            return pts_world, labels, []

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_pts))
        res_labels = np.array(pcd.cluster_dbscan(eps=0.22, min_points=12))

        # 6. Cluster Refinement (IoU + Z-Gate + Splats)
        unique_ids = np.unique(res_labels)
        final_valid_clusters = []
        
        for cid in unique_ids:
            if cid == -1: continue
            
            c_mask = (res_labels == cid)
            c_indices = target_idx_global[c_mask]
            c_pts = target_pts[c_mask]
            c_centroid = np.mean(c_pts, axis=0)

            # CRITERION A: Height Gate (Relative to floor)
            height_above_floor = c_centroid[2] - self.z_ground
            if height_above_floor > self.robot_max_height or height_above_floor < -0.1:
                labels[c_indices] = 1 # Re-classify as structure noise
                continue

            # CRITERION B: IoU Refined Wall Check (Using stretched boxes)
            pts_near_wall = 0
            for shell in self.shells:
                d = c_pts - np.array(shell['center'])
                local = d @ np.array(shell['axes'])
                if np.any(self._is_inside_stretched_shell(local, shell['extents'])):
                    pts_near_wall += np.sum(self._is_inside_stretched_shell(local, shell['extents']))
                
                if pts_near_wall / len(c_pts) > self.iou_threshold: break
            
            if (pts_near_wall / len(c_pts)) > self.iou_threshold:
                labels[c_indices] = 1
                continue

            # CRITERION C: Known Static Object (Splat)
            # (Keep existing _match_static_splat logic)
            if self._match_static_splat(c_pts, c_centroid):
                labels[c_indices] = 2
                continue

            final_valid_clusters.append({"points": c_pts, "indices": c_indices, "id": cid})

        return pts_world, labels, final_valid_clusters

    def _match_static_splat(self, cluster_pts, cluster_centroid):
        if not self.splats: return False
        for splat in self.splats:
            mu = np.array(splat['mu'])
            dist = np.linalg.norm(cluster_centroid - mu)
            if dist > 0.4: continue
            
            splat_max_scale = np.max(splat['scales'])
            cluster_max_extent = np.max(np.max(cluster_pts, axis=0) - np.min(cluster_pts, axis=0)) / 2.0
            if abs(splat_max_scale - cluster_max_extent) < 0.18:
                return True
        return False