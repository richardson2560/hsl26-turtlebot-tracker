"""
tools/build_canonical_model.py - Build canonical GMM via MVI (AMVIDC) clustering,
with radiometric intensity correction and hierarchical tree export.
"""

import argparse
import json
import heapq
import numpy as np
import open3d as o3d
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from turtlebot_tracker.core.hierarchical_gmm import HierarchicalGMM

def numpy_to_native(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

class MVIHierarchicalClustering:
    """Minimum Volume Increase clustering using covariance determinant."""
    def __init__(self, points: np.ndarray, k_neighbors=10, volume_threshold=1.8):
        self.points = points
        self.N = len(points)
        self.k_neighbors = k_neighbors
        self.volume_threshold = volume_threshold

        self.n = np.ones(self.N, dtype=np.int32)
        self.mu = points.copy()
        self.cov = np.array([np.eye(3)*1e-6 for _ in range(self.N)])
        self.volume = np.array([np.sqrt(np.maximum(1e-12, np.linalg.det(c))) for c in self.cov])

        self.parent = np.arange(self.N, dtype=np.int32)
        self.rank = np.zeros(self.N, dtype=np.int32)
        self._build_graph()

    def _build_graph(self):
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
        print(f"[MVI GRAPH] Built {len(self.edges)} local edges for {self.N} points.")

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return rx
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

    def _merge_stats(self, ri, rj):
        n_i, n_j = self.n[ri], self.n[rj]
        mu_i, mu_j = self.mu[ri], self.mu[rj]
        cov_i, cov_j = self.cov[ri], self.cov[rj]
        n_m = n_i + n_j
        mu_m = (n_i*mu_i + n_j*mu_j)/n_m
        delta_i = mu_i - mu_m
        delta_j = mu_j - mu_m
        cov_m = (n_i*(cov_i + np.outer(delta_i, delta_i)) + n_j*(cov_j + np.outer(delta_j, delta_j))) / n_m
        cov_m += np.eye(3)*1e-8
        vol_m = np.sqrt(np.maximum(1e-12, np.linalg.det(cov_m)))
        return n_m, mu_m, cov_m, vol_m

    def _merge_cost(self, ri, rj):
        if ri == rj: return float('inf')
        _, _, _, vol_m = self._merge_stats(ri, rj)
        return vol_m - self.volume[ri] - self.volume[rj]

    def cluster(self, max_components=None):
        print(f"[MVI CLUSTERING] Starting with {self.N} points. Target max: {max_components if max_components else 'auto'}")

        heap = []
        for i, j in self.edges:
            ri, rj = self.find(i), self.find(j)
            if ri != rj:
                cost = self._merge_cost(ri, rj)
                heapq.heappush(heap, (cost, ri, rj))

        step = 0
        active_clusters = self.N

        while heap:
            if max_components is not None and active_clusters <= max_components:
                print(f"  [MVI] Stopped at {active_clusters} components.")
                break

            cost, ri, rj = heapq.heappop(heap)
            if self.find(ri) != ri or self.find(rj) != rj:
                continue
            if ri == rj:
                continue

            actual_cost = self._merge_cost(ri, rj)
            force_merge = (max_components is not None and active_clusters > max_components)
            if not force_merge and actual_cost > self.volume_threshold:
                continue

            new_root = self.union(ri, rj)
            n_m, mu_m, cov_m, vol_m = self._merge_stats(ri, rj)
            self.n[new_root] = n_m
            self.mu[new_root] = mu_m
            self.cov[new_root] = cov_m
            self.volume[new_root] = vol_m

            step += 1
            active_clusters -= 1

            if step % 500 == 0 or force_merge:
                print(f"  Step {step}: merged {ri} & {rj}, active: {active_clusters}, cost: {actual_cost:.4f} {'(forced)' if force_merge else ''}")

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

            self.neighbors[ri] = set()
            self.neighbors[rj] = set()

        # Post‑process: if still above max_components, force merges
        if max_components is not None and active_clusters > max_components:
            print(f"  [MVI] Forcing merges to reach {max_components} components...")
            roots = list(set(self.find(i) for i in range(self.N)))
            while len(roots) > max_components:
                best_cost = float('inf')
                best_pair = (roots[0], roots[1])
                for a in range(len(roots)):
                    for b in range(a+1, len(roots)):
                        ri, rj = roots[a], roots[b]
                        cost = self._merge_cost(ri, rj)
                        if cost < best_cost:
                            best_cost = cost
                            best_pair = (ri, rj)
                ri, rj = best_pair
                ri_new = self.find(ri)
                rj_new = self.find(rj)
                if ri_new == rj_new:
                    roots = list(set(self.find(i) for i in range(self.N)))
                    continue
                new_root = self.union(ri_new, rj_new)
                n_m, mu_m, cov_m, vol_m = self._merge_stats(ri_new, rj_new)
                self.n[new_root] = n_m
                self.mu[new_root] = mu_m
                self.cov[new_root] = cov_m
                self.volume[new_root] = vol_m
                active_clusters -= 1
                step += 1
                print(f"  Step {step} (forced): merged {ri_new} & {rj_new}, active: {active_clusters}, cost: {best_cost:.4f}")
                roots = list(set(self.find(i) for i in range(self.N)))

        print(f"[MVI CLUSTERING] Converged. Active clusters: {active_clusters}")

        final_clusters = {}
        for i in range(self.N):
            root = self.find(i)
            final_clusters.setdefault(root, []).append(i)

        return final_clusters

def compute_radiometric_sh_c0(points: np.ndarray, intensity: np.ndarray) -> float:
    if len(points) == 0: return 28.2
    r_sq = np.sum(points**2, axis=1)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=15))
    normals = np.asarray(pcd.normals)
    if len(normals) == len(points):
        ray_vectors = -points / (np.sqrt(r_sq)[:, None] + 1e-6)
        cos_eta = np.abs(np.sum(ray_vectors * normals, axis=1))
        cos_eta = np.maximum(cos_eta, 0.1)
    else:
        cos_eta = np.ones(len(points)) * 0.5
    corrected = intensity * (r_sq / cos_eta)
    return float(0.28209479177 * np.mean(corrected))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", type=str, default="data/outputs/canonical_raw_candidate.npz")
    parser.add_argument("--output_splats", type=str, default="config/canonical_turtlebot2.json")
    parser.add_argument("--output_points", type=str, default="config/canonical_points.json")
    parser.add_argument("--voxel_size", type=float, default=0.006)
    parser.add_argument("--k_neighbors", type=int, default=10)
    parser.add_argument("--volume_threshold", type=float, default=1.8)
    parser.add_argument("--max_components", type=int, default=None)
    parser.add_argument("--min_cluster_size", type=int, default=25)
    args = parser.parse_args()

    npz_path = Path(args.input_npz)
    if not npz_path.exists():
        print(f"[ERROR] Input NPZ not found: {npz_path}")
        return

    print("\n=======================================================")
    print(" MVI CLUSTERING + SEPARATE JSON EXPORT")
    print("=======================================================\n")
    data = np.load(npz_path)
    raw_points = data["points"]
    raw_intensity = data["intensity"]
    print(f"[INFO] Loaded {len(raw_points):,} points.")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(raw_points))
    pcd_down, _, trace = pcd.voxel_down_sample_and_trace(args.voxel_size, pcd.get_min_bound(), pcd.get_max_bound())
    pts = np.asarray(pcd_down.points)
    down_indices = [t[0] for t in trace if len(t)>0]
    if len(down_indices) == len(pts):
        intensities = raw_intensity[down_indices]
    else:
        intensities = np.ones(len(pts)) * 100.0

    centroid = np.mean(pts, axis=0)
    pts_rel = pts - centroid

    clusterer = MVIHierarchicalClustering(pts_rel, k_neighbors=args.k_neighbors, volume_threshold=args.volume_threshold)
    final_clusters = clusterer.cluster(max_components=args.max_components)

    splats = []
    points_list = []
    labels_list = []
    total_pts = len(pts_rel)
    for domain_id, indices in final_clusters.items():
        if len(indices) < args.min_cluster_size:
            continue
        cluster_pts = pts_rel[indices]
        cluster_int = intensities[indices]
        mu_k = np.mean(cluster_pts, axis=0)
        cov_k = np.cov(cluster_pts.T) + np.eye(3)*1e-6
        eigvals, eigvecs = np.linalg.eigh(cov_k)
        eigvals = np.maximum(eigvals, 1e-5)
        scales_k = np.sqrt(eigvals)
        extents = np.max(cluster_pts, axis=0) - np.min(cluster_pts, axis=0)
        scales_k = np.minimum(scales_k, extents/2.0 + 0.01)
        weight_k = float(len(indices) / total_pts)
        c0_sh = compute_radiometric_sh_c0(cluster_pts, cluster_int)

        splats.append({
            "mu": mu_k.tolist(),
            "cov": cov_k.tolist(),
            "scales": scales_k.tolist(),
            "rotation": eigvecs.tolist(),
            "weight": weight_k,
            "sh_c0": float(c0_sh)
        })
        domain_id_int = int(domain_id)
        points_list.extend(cluster_pts.tolist())
        labels_list.extend([domain_id_int] * len(cluster_pts))

    # Save splats
    splat_path = Path(args.output_splats)
    splat_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splat_path, 'w') as f:
        json.dump({
            "canonical_gaussians": splats,
            "num_components": len(splats),
            "algorithm": "MVI_Hierarchical_Clustering"
        }, f, indent=4, default=numpy_to_native)
    print(f"[SAVED] Splats ({len(splats)} components) -> {splat_path}")

    # Save points
    points_path = Path(args.output_points)
    points_path.parent.mkdir(parents=True, exist_ok=True)
    with open(points_path, 'w') as f:
        json.dump({
            "canonical_points": points_list,
            "canonical_point_labels": labels_list,
            "num_points": len(points_list)
        }, f, indent=4, default=numpy_to_native)
    print(f"[SAVED] Points ({len(points_list):,} pts) + labels -> {points_path}")

    # Build hierarchical tree
    hg = HierarchicalGMM(splats)
    tree_path = Path("config/canonical_tree.json")
    hg.save(tree_path)
    print(f"[SAVED] Hierarchical tree -> {tree_path}")

    print("\n[SUCCESS] Canonical model and tree built.")
    print("  -> Run 'python tools/run_online_tracker.py' to process bags.")

if __name__ == "__main__":
    main()