import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from typing import List, Dict

class ClusterGaussianFitter:
    """Extracts candidate clusters and fits robust 3D Gaussian components without matrix singularities."""
    
    def __init__(self, eps: float = 0.22, min_points: int = 20, max_points: int = 2000, cov_floor: float = 1e-4):
        self.eps = eps
        self.min_points = min_points
        self.max_points = max_points
        self.cov_floor = cov_floor

    def extract_clusters_and_fit_gaussians(self, obstacle_points: np.ndarray, num_gaussians: int = 3) -> List[Dict]:
        """Segments obstacle points using DBSCAN and fits regularized 3D Gaussians."""
        if len(obstacle_points) < self.min_points:
            return []

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obstacle_points)
        
        labels = np.array(pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points, print_progress=False))
        max_label = labels.max()
        
        candidate_models = []
        
        for i in range(max_label + 1):
            cluster_pts = obstacle_points[labels == i]
            
            if len(cluster_pts) < self.min_points or len(cluster_pts) > self.max_points:
                continue
                
            # Dimensional filter for Turtlebot2 bounding extent
            extents = cluster_pts.max(axis=0) - cluster_pts.min(axis=0)
            if extents[0] > 0.85 or extents[1] > 0.85 or extents[2] > 0.85:
                continue

            # Robust K-Means Centroid Seeding + Sample Covariance Regularization
            k_comp = min(num_gaussians, max(1, len(cluster_pts) // 15))
            
            if k_comp == 1:
                mu_list = [cluster_pts.mean(axis=0)]
                assignments = np.zeros(len(cluster_pts), dtype=int)
            else:
                kmeans = KMeans(n_clusters=k_comp, n_init=3, random_state=42).fit(cluster_pts)
                mu_list = kmeans.cluster_centers_
                assignments = kmeans.labels_

            gaussian_components = []
            for k in range(k_comp):
                pts_k = cluster_pts[assignments == k]
                if len(pts_k) < 3:
                    pts_k = cluster_pts  # Fallback to entire cluster

                mu = mu_list[k]
                cov_raw = np.cov(pts_k.T) if len(pts_k) > 3 else np.eye(3) * 0.01
                if cov_raw.ndim < 2:
                    cov_raw = np.eye(3) * 0.01

                # Enforce Isotropic Covariance Floor: Σ_robust = Σ_sample + ε_cov * I_3
                cov_robust = cov_raw + np.eye(3) * self.cov_floor
                
                # Eigenvalue decomposition for principal semi-axes
                eigvals, eigvecs = np.linalg.eigh(cov_robust)
                eigvals = np.maximum(eigvals, self.cov_floor)
                scales = np.sqrt(eigvals)

                gaussian_components.append({
                    'mu': mu,
                    'cov': cov_robust,
                    'scales': scales,
                    'rotation': eigvecs,
                    'weight': len(pts_k) / len(cluster_pts)
                })

            candidate_models.append({
                'cluster_pts': cluster_pts,
                'centroid': cluster_pts.mean(axis=0),
                'gaussians': gaussian_components,
                'num_points': len(cluster_pts)
            })
            
        return candidate_models