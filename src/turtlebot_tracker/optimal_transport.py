"""
optimal_transport.py - Exact Local-Shape Optimal Transport Solver (EMD)
"""

import numpy as np
import ot

class OptimalTransportMatcher:
    """Computes Translation-Invariant Bures-Wasserstein Optimal Transport using Exact EMD."""
    
    def __init__(self, gamma_prune: float = 0.2, gamma_spawn: float = 0.2):
        self.gamma_prune = gamma_prune
        self.gamma_spawn = gamma_spawn

    @staticmethod
    def compute_bures_wasserstein_distance(g1: dict, g2: dict) -> float:
        """Computes Bures-Wasserstein distance between local zero-centered Gaussians."""
        mu_dist_sq = np.sum((np.array(g1['mu']) - np.array(g2['mu'])) ** 2)
        scale_dist_sq = np.sum((np.array(g1['scales']) - np.array(g2['scales'])) ** 2)
        return float(mu_dist_sq + scale_dist_sq)

    def match_models(self, canonical_gaussians: list, observed_gaussians: list) -> tuple:
        """
        Computes Zero-Centered Shape Distance using Exact Earth Mover's Distance (EMD).
        Returns: (shape_cost, coupling_matrix_P)
        """
        K = len(canonical_gaussians)
        M = len(observed_gaussians)

        # Zero-center observed components relative to their local mean centroid
        obs_center = np.mean([g['mu'] for g in observed_gaussians], axis=0)
        zero_centered_obs = []
        for g in observed_gaussians:
            g_centered = g.copy()
            g_centered['mu'] = np.array(g['mu']) - obs_center
            zero_centered_obs.append(g_centered)

        # Build Relative Local Shape Cost Matrix C_shape [K, M]
        C_shape = np.zeros((K, M), dtype=np.float64)
        for i in range(K):
            for j in range(M):
                C_shape[i, j] = self.compute_bures_wasserstein_distance(canonical_gaussians[i], zero_centered_obs[j])

        # Augmented Matrix for Unbalanced Transport
        C_tilde = np.zeros((K + 1, M + 1), dtype=np.float64)
        C_tilde[:K, :M] = C_shape
        C_tilde[:K, M] = self.gamma_prune
        C_tilde[K, :M] = self.gamma_spawn
        C_tilde[K, M] = 0.0

        w_canon = np.array([g['weight'] for g in canonical_gaussians], dtype=np.float64)
        w_obs = np.array([g['weight'] for g in observed_gaussians], dtype=np.float64)
        
        a = np.append(w_canon, np.sum(w_obs))
        b = np.append(w_obs, np.sum(w_canon))
        
        # Normalize weights to sum to 1.0 for Exact EMD
        a_norm = a / np.sum(a)
        b_norm = b / np.sum(b)

        # Exact EMD Solver (Fast, exact, 0 warnings)
        P_tilde = ot.emd(a_norm, b_norm, C_tilde)
        P_mat = P_tilde[:K, :M]
        shape_cost = np.sum(P_tilde * C_tilde)

        return float(shape_cost), P_mat