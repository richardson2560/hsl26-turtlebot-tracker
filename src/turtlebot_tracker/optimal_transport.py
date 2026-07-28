import numpy as np
import ot
from typing import Dict, List, Tuple

class OptimalTransportMatcher:
    """Computes Bures-Wasserstein UOT between Canonical and Observed Gaussian Models."""
    
    def __init__(self, gamma_prune: float = 0.25, gamma_spawn: float = 0.25, reg_sinkhorn: float = 0.05):
        self.gamma_prune = gamma_prune
        self.gamma_spawn = gamma_spawn
        self.reg_sinkhorn = reg_sinkhorn

    def compute_bures_wasserstein_distance(self, g1: Dict, g2: Dict) -> float:
        """Computes closed-form Bures-Wasserstein distance between two 3D Gaussians."""
        mu_dist_sq = np.sum((g1['mu'] - g2['mu']) ** 2)
        scale_dist_sq = np.sum((g1['scales'] - g2['scales']) ** 2)
        return float(mu_dist_sq + scale_dist_sq)

    def match_models(self, canonical_gaussians: List[Dict], observed_gaussians: List[Dict]) -> Tuple[float, np.ndarray]:
        """
        Solves Unbalanced Optimal Transport with Ghost Nodes.
        Returns: (transport_cost, coupling_matrix_P)
        """
        K = len(canonical_gaussians)
        M = len(observed_gaussians)
        
        # Build Base Cost Matrix C [K, M]
        C_base = np.zeros((K, M), dtype=np.float64)
        for i in range(K):
            for j in range(M):
                C_base[i, j] = self.compute_bures_wasserstein_distance(canonical_gaussians[i], observed_gaussians[j])

        # Construct Augmented Cost Matrix C_tilde [(K+1), (M+1)]
        C_tilde = np.zeros((K + 1, M + 1), dtype=np.float64)
        C_tilde[:K, :M] = C_base
        C_tilde[:K, M] = self.gamma_prune
        C_tilde[K, :M] = self.gamma_spawn
        C_tilde[K, M] = 0.0

        # Marginal Weights
        w_canon = np.array([g['weight'] for g in canonical_gaussians], dtype=np.float64)
        w_obs = np.array([g['weight'] for g in observed_gaussians], dtype=np.float64)
        
        a = np.append(w_canon, np.sum(w_obs))
        b = np.append(w_obs, np.sum(w_canon))
        
        # Solve Entropic Sinkhorn
        P_tilde = ot.sinkhorn(a, b, C_tilde, reg=self.reg_sinkhorn, numItermax=100, stopThr=1e-3)
        
        # Extract Canonical-Observed submatrix
        P_mat = P_tilde[:K, :M]
        total_cost = np.sum(P_tilde * C_tilde)
        
        return float(total_cost), P_mat