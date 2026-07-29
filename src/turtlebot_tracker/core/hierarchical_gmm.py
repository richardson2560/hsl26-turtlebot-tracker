"""
hierarchical_gmm.py - Build KL‑based hierarchical reduction tree (Runnalls, 2007).
"""

import json
import numpy as np
from copy import deepcopy
from typing import List, Dict

class HierarchicalGMM:
    def __init__(self, gmm: List[Dict]):
        """
        gmm: list of dicts with keys 'mu', 'cov', 'weight' (and optionally 'scales', 'rotation', 'sh_c0').
        """
        self.original = deepcopy(gmm)
        self.tree = self._build_tree(self.original)

    @staticmethod
    def _moment_match(g1: Dict, g2: Dict) -> Dict:
        w1, w2 = g1['weight'], g2['weight']
        w = w1 + w2
        mu1, mu2 = np.array(g1['mu']), np.array(g2['mu'])
        mu = (w1 * mu1 + w2 * mu2) / w
        cov1, cov2 = np.array(g1['cov']), np.array(g2['cov'])
        delta1 = mu1 - mu
        delta2 = mu2 - mu
        cov = (w1 * (cov1 + np.outer(delta1, delta1)) +
               w2 * (cov2 + np.outer(delta2, delta2))) / w
        merged = {'mu': mu.tolist(), 'cov': cov.tolist(), 'weight': w}
        if 'scales' in g1 and 'scales' in g2:
            eigvals, eigvecs = np.linalg.eigh(cov)
            merged['scales'] = np.sqrt(np.maximum(eigvals, 1e-5)).tolist()
            merged['rotation'] = eigvecs.tolist()
        if 'sh_c0' in g1 and 'sh_c0' in g2:
            merged['sh_c0'] = (w1 * g1['sh_c0'] + w2 * g2['sh_c0']) / w
        return merged

    @staticmethod
    def _kl_cost(g1: Dict, g2: Dict) -> float:
        mu1, mu2 = np.array(g1['mu']), np.array(g2['mu'])
        cov1, cov2 = np.array(g1['cov']), np.array(g2['cov'])
        cov_mean = (cov1 + cov2) / 2
        try:
            mahal = mu1 @ np.linalg.solve(cov_mean, mu2)
        except:
            mahal = np.linalg.norm(mu1 - mu2) ** 2
        eig1 = np.sqrt(np.maximum(np.linalg.eigvalsh(cov1), 1e-5))
        eig2 = np.sqrt(np.maximum(np.linalg.eigvalsh(cov2), 1e-5))
        scale_diff = np.sum((eig1 - eig2) ** 2)
        return mahal + scale_diff + 0.1 * np.log(g1['weight'] + g2['weight'])

    def _build_tree(self, gmm: List[Dict]) -> List[List[Dict]]:
        tree = [deepcopy(gmm)]
        current = deepcopy(gmm)
        while len(current) > 1:
            best_cost = np.inf
            best_pair = (0, 1)
            for i in range(len(current)):
                for j in range(i+1, len(current)):
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
        M = max(1, min(M, len(self.original)))
        idx = len(self.original) - M
        idx = max(0, min(idx, len(self.tree)-1))
        return deepcopy(self.tree[idx])

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.tree, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'HierarchicalGMM':
        with open(path, 'r') as f:
            tree = json.load(f)
        inst = cls([])
        inst.tree = tree
        inst.original = tree[0]
        return inst