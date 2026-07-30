"""
hierarchical_gmm.py - Runnalls KL-Divergence Hierarchical Mixture Reduction Tree.

Maintains a pre-computed reduction tree of canonical Gaussian Splats (Runnalls, 2007).
Selects model granularity level M based on observed point count N in O(1) time.
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Union
import numpy as np


class HierarchicalGMM:
    """Manages KL-divergence hierarchical reduction tree for canonical GMM splats."""

    def __init__(self, gmm: List[Dict]):
        """
        Args:
            gmm: List of dictionaries with keys 'mu', 'cov', 'weight', 'sh_c0'.
        """
        self.original = deepcopy(gmm)
        self.tree = self._build_tree(self.original) if len(gmm) > 0 else []

    @staticmethod
    def _moment_match(g1: Dict, g2: Dict) -> Dict:
        """Merges two Gaussian components using exact first and second-order moment matching."""
        w1, w2 = g1['weight'], g2['weight']
        w = w1 + w2
        mu1, mu2 = np.array(g1['mu'], dtype=np.float64), np.array(g2['mu'], dtype=np.float64)
        mu = (w1 * mu1 + w2 * mu2) / w

        cov1, cov2 = np.array(g1['cov'], dtype=np.float64), np.array(g2['cov'], dtype=np.float64)
        delta1 = mu1 - mu
        delta2 = mu2 - mu
        cov = (w1 * (cov1 + np.outer(delta1, delta1)) +
               w2 * (cov2 + np.outer(delta2, delta2))) / w

        merged = {
            'mu': mu.tolist(),
            'cov': cov.tolist(),
            'weight': float(w)
        }

        if 'scales' in g1 and 'scales' in g2:
            eigvals, eigvecs = np.linalg.eigh(cov)
            merged['scales'] = np.sqrt(np.maximum(eigvals, 1e-5)).tolist()
            merged['rotation'] = eigvecs.tolist()

        if 'sh_c0' in g1 and 'sh_c0' in g2:
            merged['sh_c0'] = float((w1 * g1['sh_c0'] + w2 * g2['sh_c0']) / w)

        return merged

    @staticmethod
    def _kl_cost(g1: Dict, g2: Dict) -> float:
        """Computes Kullback-Leibler information loss cost for merging two components."""
        mu1, mu2 = np.array(g1['mu'], dtype=np.float64), np.array(g2['mu'], dtype=np.float64)
        cov1, cov2 = np.array(g1['cov'], dtype=np.float64), np.array(g2['cov'], dtype=np.float64)
        w1, w2 = g1['weight'], g2['weight']

        w_merged = w1 + w2
        merged = HierarchicalGMM._moment_match(g1, g2)
        cov_m = np.array(merged['cov'], dtype=np.float64)

        det_1 = np.maximum(np.linalg.det(cov1), 1e-12)
        det_2 = np.maximum(np.linalg.det(cov2), 1e-12)
        det_m = np.maximum(np.linalg.det(cov_m), 1e-12)

        # Runnalls KL cost closed form
        cost = 0.5 * (w_merged * np.log(det_m) - w1 * np.log(det_1) - w2 * np.log(det_2))
        return float(cost)

    def _build_tree(self, gmm: List[Dict]) -> List[List[Dict]]:
        """Greedily builds KL reduction tree from K_0 down to 1 component."""
        tree = [deepcopy(gmm)]
        current = deepcopy(gmm)

        while len(current) > 1:
            best_cost = np.inf
            best_pair = (0, 1)

            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    cost = self._kl_cost(current[i], current[j])
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (i, j)

            merged = self._moment_match(current[best_pair[0]], current[best_pair[1]])
            new_list = [current[k] for k in range(len(current)) if k not in best_pair]
            new_list.append(merged)
            current = new_list
            tree.append(deepcopy(current))

        return tree

    def get_level(self, M: int) -> List[Dict]:
        """
        Retrieves the M-component GMM level from the pre-computed tree.

        Args:
            M: Desired number of Gaussian components.

        Returns:
            List of GMM dictionaries at granular level M.
        """
        if not self.tree:
            return deepcopy(self.original)

        M = max(1, min(M, len(self.original)))
        idx = len(self.original) - M
        idx = max(0, min(idx, len(self.tree) - 1))
        return deepcopy(self.tree[idx])

    def save(self, path: Union[str, Path]) -> None:
        """Saves hierarchical tree to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.tree, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'HierarchicalGMM':
        """Loads hierarchical tree from JSON file."""
        with open(path, 'r') as f:
            tree = json.load(f)
        inst = cls([])
        inst.tree = tree
        inst.original = tree[0] if len(tree) > 0 else []
        return inst