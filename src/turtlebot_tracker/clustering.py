import numpy as np
import open3d as o3d
from sklearn.mixture import GaussianMixture
from typing import List, Dict

class ClusterGaussianFitter:
    """Segments obstacles into Euclidean clusters and fits 3D Gaussian Components."""
    
    def __init__(self, eps: float = 0.20, min_points: int = 15, max_points: int = 2500):
        self.eps = eps
        self.min_points = min_points
        self.max_points = max_points

    def extract_clusters_and_fit_gaussians(self, obstacle_points: np.ndarray, num_gaussians_per_cluster: int = 4) -> List[Dict]:
        """Clusters obstacle points and fits GMM representations to each candidate cluster."""
        if len(obstacle_points) < self.min_points:
            return []

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obstacle_points)
        
        # DBSCAN Clustering
        labels = np.array(pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points, print_progress=False))
        max_label = labels.max()
        
        candidate_models = []
        
        for i in range(max_label + 1):
            cluster_mask = (labels == i)
            cluster_pts = obstacle_points[cluster_mask]
            
            if len(cluster_pts) < self.min_points or len(cluster_pts) > self.max_points:
                continue
                
            # Filter clusters by physical Turtlebot2 dimensional bounds (e.g., width <= 0.8m, height <= 0.8m)
            bbox_min = cluster_pts.min(axis=0)
            bbox_max = cluster_pts.max(axis=0)
            extents = bbox_max - bbox_min
            
            if extents[0] > 0.9 or extents[1] > 0.9 or extents[2] > 0.9:
                continue  # Too large to be a Turtlebot2
                
            # Fit GMM
            k_comp = min(num_gaussians_per_cluster, len(cluster_pts) // 5)
            if k_comp < 1:
                k_comp = 1
                
            gmm = GaussianMixture(n_components=k_comp, covariance_type='full', max_iter=50, random_state=42)
            gmm.fit(cluster_pts)
            
            gaussian_components = []
            for k in range(k_comp):
                mu = gmm.means_[k]
                cov = gmm.covariances_[k]
                weight = gmm.weights_[k]
                
                # Spectral decomposition for scales
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.maximum(eigvals, 1e-6)  # Positive definite guarantee
                scales = np.sqrt(eigvals)
                
                gaussian_components.append({
                    'mu': mu,
                    'cov': cov,
                    'scales': scales,
                    'rotation': eigvecs,
                    'weight': weight
                })
                
            candidate_models.append({
                'cluster_pts': cluster_pts,
                'centroid': cluster_pts.mean(axis=0),
                'gaussians': gaussian_components,
                'num_points': len(cluster_pts)
            })
            
        return candidate_models