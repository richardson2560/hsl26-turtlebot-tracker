"""
build_canonical_model.py - MVI Offline Extraction and Runnalls Tree Generation.

Fits organic 3D Gaussian Splats using Minimum Volume Increase (MVI) clustering,
computes range-corrected c0_SH radiometric coefficients, and exports canonical JSONs.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import open3d as o3d

sys.path.append(str(Path(__file__).parent.parent / "src"))

from turtlebot_tracker.core.hierarchical_gmm import HierarchicalGMM
from turtlebot_tracker.core.mvi_clustering import MVIHierarchicalClustering, numpy_to_native
from turtlebot_tracker.core.radiometric import compute_radiometric_sh_c0

def main():
    parser = argparse.ArgumentParser(description="MVI Offline Canonical Model Generator")
    parser.add_argument("--input_npz", type=str, default="data/outputs/canonical_raw_candidate.npz")
    parser.add_argument("--output_splats", type=str, default="config/canonical_turtlebot2.json")
    parser.add_argument("--output_points", type=str, default="config/canonical_points.json")
    parser.add_argument("--voxel_size", type=float, default=0.006)
    parser.add_argument("--k_neighbors", type=int, default=10)
    parser.add_argument("--volume_threshold", type=float, default=1.0)
    parser.add_argument("--max_components", type=int, default=12)
    parser.add_argument("--min_cluster_size", type=int, default=25)
    args = parser.parse_args()

    npz_path = Path(args.input_npz)
    if not npz_path.exists():
        print(f"[ERROR] Candidate NPZ not found: {npz_path}")
        print("Run 'python tools/select_canonical_candidate.py --auto' first.")
        return

    print("\n=======================================================")
    print(" MVI OFFLINE CANONICAL MODEL GENERATOR")
    print("=======================================================\n")

    data = np.load(npz_path)
    raw_points = data["points"]
    raw_intensity = data["intensity"]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(raw_points))
    pcd_down, _, trace = pcd.voxel_down_sample_and_trace(args.voxel_size, pcd.get_min_bound(), pcd.get_max_bound())
    pts = np.asarray(pcd_down.points, dtype=np.float64)
    down_indices = [t[0] for t in trace if len(t) > 0]
    intensities = (raw_intensity[down_indices]
                   if len(down_indices) == len(pts)
                   else np.ones(len(pts)) * 100.0)

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
        cov_k = np.cov(cluster_pts.T) + np.eye(3) * 1e-6
        eigvals, eigvecs = np.linalg.eigh(cov_k)
        eigvals = np.maximum(eigvals, 1e-5)
        scales_k = np.sqrt(eigvals)

        extents = np.max(cluster_pts, axis=0) - np.min(cluster_pts, axis=0)
        scales_k = np.minimum(scales_k, extents / 2.0 + 0.01)
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

        points_list.extend(cluster_pts.tolist())
        labels_list.extend([int(domain_id)] * len(cluster_pts))

    # Save splats JSON
    splat_path = Path(args.output_splats)
    splat_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splat_path, 'w') as f:
        json.dump({
            "canonical_gaussians": splats,
            "num_components": len(splats),
            "algorithm": "MVI_Determinantal_Clustering"
        }, f, indent=4, default=numpy_to_native)
    print(f"[SAVED] Canonical Splats ({len(splats)} components) -> {splat_path}")

    # Save points JSON
    points_path = Path(args.output_points)
    with open(points_path, 'w') as f:
        json.dump({
            "canonical_points": points_list,
            "canonical_point_labels": labels_list,
            "num_points": len(points_list)
        }, f, indent=4, default=numpy_to_native)
    print(f"[SAVED] Canonical Points ({len(points_list):,} pts) -> {points_path}")

    # Build Runnalls Tree JSON
    hg = HierarchicalGMM(splats)
    tree_path = Path("config/canonical_tree.json")
    hg.save(tree_path)
    print(f"[SAVED] Runnalls Hierarchical Tree -> {tree_path}")


if __name__ == "__main__":
    main()