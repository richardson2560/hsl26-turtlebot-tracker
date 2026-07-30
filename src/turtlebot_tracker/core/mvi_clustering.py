"""
mvi_clustering.py - Minimum Volume Increase (MVI) clustering using determinantal volume expansion.
Shared module for offline model building.
"""

import heapq
import numpy as np
from sklearn.neighbors import NearestNeighbors


def numpy_to_native(obj):
    """Converts numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class MVIHierarchicalClustering:
    """Minimum Volume Increase (MVI) clustering using determinantal volume expansion."""

    def __init__(self, points: np.ndarray, k_neighbors: int = 10, volume_threshold: float = 1.8):
        self.points = points
        self.N = len(points)
        self.k_neighbors = k_neighbors
        self.volume_threshold = volume_threshold

        self.n = np.ones(self.N, dtype=np.int32)
        self.mu = points.copy()
        self.cov = np.array([np.eye(3, dtype=np.float64) * 1e-6 for _ in range(self.N)])
        self.volume = np.array([np.sqrt(np.maximum(1e-12, np.linalg.det(c))) for c in self.cov])

        self.parent = np.arange(self.N, dtype=np.int32)
        self.rank = np.zeros(self.N, dtype=np.int32)
        self._build_graph()

    def _build_graph(self) -> None:
        neigh = NearestNeighbors(n_neighbors=self.k_neighbors, algorithm='kd_tree')
        neigh.fit(self.points)
        distances, indices = neigh.kneighbors(self.points)
        self.neighbors = [set() for _ in range(self.N)]
        self.edges = []
        for i in range(self.N):
            for j, d in zip(indices[i], distances[i]):
                if i < j and d < 0.10:
                    self.edges.append((i, j))
                    self.neighbors[i].add(j)
                    self.neighbors[j].add(i)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> int:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return rx
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
            return ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
            return rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
            return rx

    def _merge_stats(self, ri: int, rj: int):
        n_i, n_j = self.n[ri], self.n[rj]
        mu_i, mu_j = self.mu[ri], self.mu[rj]
        cov_i, cov_j = self.cov[ri], self.cov[rj]
        n_m = n_i + n_j
        mu_m = (n_i * mu_i + n_j * mu_j) / n_m
        delta_i = mu_i - mu_m
        delta_j = mu_j - mu_m
        cov_m = (n_i * (cov_i + np.outer(delta_i, delta_i)) + n_j * (cov_j + np.outer(delta_j, delta_j))) / n_m
        cov_m += np.eye(3) * 1e-8
        vol_m = np.sqrt(np.maximum(1e-12, np.linalg.det(cov_m)))
        return n_m, mu_m, cov_m, vol_m

    def _merge_cost(self, ri: int, rj: int) -> float:
        if ri == rj:
            return float('inf')
        _, _, _, vol_m = self._merge_stats(ri, rj)
        return float(vol_m - self.volume[ri] - self.volume[rj])

    def cluster(self, max_components: int = 6) -> dict:
        heap = []
        for i, j in self.edges:
            ri, rj = self.find(i), self.find(j)
            if ri != rj:
                cost = self._merge_cost(ri, rj)
                heapq.heappush(heap, (cost, ri, rj))

        active_clusters = self.N

        while heap:
            if max_components is not None and active_clusters <= max_components:
                break

            cost, ri, rj = heapq.heappop(heap)
            if self.find(ri) != ri or self.find(rj) != rj or ri == rj:
                continue

            actual_cost = self._merge_cost(ri, rj)
            if actual_cost > self.volume_threshold and active_clusters <= max_components:
                continue

            new_root = self.union(ri, rj)
            n_m, mu_m, cov_m, vol_m = self._merge_stats(ri, rj)
            self.n[new_root] = n_m
            self.mu[new_root] = mu_m
            self.cov[new_root] = cov_m
            self.volume[new_root] = vol_m
            active_clusters -= 1

            new_neighbors = self.neighbors[ri].union(self.neighbors[rj])
            new_neighbors.discard(new_root)
            cleaned = set()
            for nb in new_neighbors:
                root_nb = self.find(nb)
                if root_nb != new_root:
                    cleaned.add(root_nb)
            self.neighbors[new_root] = cleaned

            for nb in cleaned:
                if nb != new_root:
                    new_cost = self._merge_cost(new_root, nb)
                    heapq.heappush(heap, (new_cost, new_root, nb))

        final_clusters = {}
        for i in range(self.N):
            root = self.find(i)
            final_clusters.setdefault(root, []).append(i)

        return final_clusters