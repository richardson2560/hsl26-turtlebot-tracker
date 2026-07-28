import numpy as np
import ot
from typing import Dict, List, Tuple

class RigidPoseEstimator:
    """Extracts SE(3) Rigid Transformation via Weighted Procrustes SVD."""
    
    @staticmethod
    def estimate_pose(canonical_gaussians: List[Dict], observed_gaussians: List[Dict], P_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns: (R_matrix [3,3], t_vector [3])
        """
        K = len(canonical_gaussians)
        M = len(observed_gaussians)
        
        P_weight = np.sum(P_mat)
        if P_weight < 1e-6:
            return np.eye(3), np.zeros(3)

        mu_canon = np.array([g['mu'] for g in canonical_gaussians])  # [K, 3]
        mu_obs = np.array([g['mu'] for g in observed_gaussians])      # [M, 3]

        # Weighted Centroids
        w_canon_sum = np.sum(P_mat, axis=1)  # [K]
        w_obs_sum = np.sum(P_mat, axis=0)    # [M]
        
        center_canon = np.sum(mu_canon * w_canon_sum[:, None], axis=0) / P_weight
        center_obs = np.sum(mu_obs * w_obs_sum[:, None], axis=0) / P_weight

        # Center Coordinates
        mu_canon_centered = mu_canon - center_canon
        mu_obs_centered = mu_obs - center_obs

        # Compute Cross-Covariance Matrix H [3, 3]
        H = np.zeros((3, 3), dtype=np.float64)
        for i in range(K):
            for j in range(M):
                if P_mat[i, j] > 1e-4:
                    H += P_mat[i, j] * np.outer(mu_canon_centered[i], mu_obs_centered[j])

        # SVD Decomposition
        U, S, Vt = np.linalg.svd(H)
        V = Vt.T
        
        d = np.linalg.det(V @ U.T)
        S_matrix = np.diag([1.0, 1.0, np.sign(d)])
        
        R_opt = V @ S_matrix @ U.T
        t_opt = center_obs - R_opt @ center_canon

        return R_opt, t_opt