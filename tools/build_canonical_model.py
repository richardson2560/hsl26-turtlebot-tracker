"""
build_canonical_model.py - Expert Actor Modeler (Robot).
Extracts Label 3 (Robot) and fits high-fidelity GMM + Radiometric C0.
Outputs compatible with online pipeline (canonical_turtlebot2.json).
"""

import argparse
import json
import sys
import numpy as np
import open3d as o3d
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from turtlebot_tracker.core.mvi_clustering import MVIHierarchicalClustering, numpy_to_native
from turtlebot_tracker.core.radiometric import compute_radiometric_sh_c0
from turtlebot_tracker.core.hierarchical_gmm import HierarchicalGMM


def main():
    parser = argparse.ArgumentParser(description="Expert Robot Canonical Modeler")
    parser.add_argument("--input_npz", type=str, default="data/outputs/static_full.npz",
                        help="Input NPZ with points, labels, intensity")
    parser.add_argument("--output_splats", type=str, default="config/canonical_turtlebot2.json",
                        help="Output JSON for canonical splats (online compatible)")
    parser.add_argument("--output_points", type=str, default="config/canonical_points.json",
                        help="Output JSON for canonical points")
    parser.add_argument("--voxel_size", type=float, default=0.006,
                        help="Voxel size for robot model (fine)")
    parser.add_argument("--k_neighbors", type=int, default=10)
    parser.add_argument("--volume_threshold", type=float, default=1.8,
                        help="MVI volume threshold (lower = more splats)")
    parser.add_argument("--max_components", type=int, default=10,
                        help="Max number of Gaussian components")
    parser.add_argument("--min_cluster_size", type=int, default=25)
    args = parser.parse_args()

    # Load data
    data = np.load(args.input_npz)
    points = data['points']
    labels = data['labels']
    intensities = data['intensity']

    # Isolate Robot (Label 3)
    robot_mask = (labels == 3)
    robot_pts = points[robot_mask]
    robot_int = intensities[robot_mask]

    if len(robot_pts) < 100:
        print("[ERROR] Not enough robot points (Label 3) found. Run select_canonical_candidate.py first.")
        return

    print(f"[INFO] Isolated {len(robot_pts):,} robot points.")

    # Centering
    centroid = np.mean(robot_pts, axis=0)
    pts_centered = robot_pts - centroid

    # Voxel downsampling
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_centered))
    pcd_down, _, trace = pcd.voxel_down_sample_and_trace(
        args.voxel_size, pcd.get_min_bound(), pcd.get_max_bound()
    )
    pts_final = np.asarray(pcd_down.points)
    # Map intensities to downsampled points
    int_final = np.array([np.mean(robot_int[t]) for t in trace if len(t) > 0])
    if len(int_final) != len(pts_final):
        int_final = np.ones(len(pts_final)) * 100.0

    print(f"[INFO] Downsampled to {len(pts_final):,} points for MVI.")

    # MVI Clustering
    clusterer = MVIHierarchicalClustering(
        pts_final,
        k_neighbors=args.k_neighbors,
        volume_threshold=args.volume_threshold
    )
    clusters = clusterer.cluster(max_components=args.max_components)

    splats = []
    pts_list = []
    lbl_list = []
    total_pts = len(pts_final)

    for domain_id, indices in clusters.items():
        if len(indices) < args.min_cluster_size:
            continue
        c_pts = pts_final[indices]
        c_int = int_final[indices]

        mu = np.mean(c_pts, axis=0)
        cov = np.cov(c_pts.T) + np.eye(3) * 1e-6
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-7)
        scales = np.sqrt(vals)
        weight = float(len(indices) / total_pts)
        sh_c0 = compute_radiometric_sh_c0(c_pts, c_int)

        splats.append({
            "mu": mu.tolist(),
            "cov": cov.tolist(),
            "scales": scales.tolist(),
            "rotation": vecs.tolist(),
            "weight": weight,
            "sh_c0": float(sh_c0)
        })
        pts_list.extend(c_pts.tolist())
        lbl_list.extend([int(domain_id)] * len(c_pts))

    print(f"[INFO] Generated {len(splats)} robot splats.")

    # --- Save compatible JSONs ---
    # 1. Canonical Splats (for online candidate filter and registration)
    splat_path = Path(args.output_splats)
    splat_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splat_path, 'w') as f:
        json.dump({
            "canonical_gaussians": splats,
            "num_components": len(splats),
            "algorithm": "MVI_Hierarchical_Clustering",
            "centroid_offset": centroid.tolist()  # Optional extra info
        }, f, indent=4, default=numpy_to_native)
    print(f"[SAVED] Canonical Splats -> {splat_path}")

    # 2. Canonical Points (for visualization)
    points_path = Path(args.output_points)
    with open(points_path, 'w') as f:
        json.dump({
            "canonical_points": pts_list,
            "canonical_point_labels": lbl_list,
            "num_points": len(pts_list)
        }, f, indent=4, default=numpy_to_native)
    print(f"[SAVED] Canonical Points ({len(pts_list):,}) -> {points_path}")

    # 3. Runnalls Tree (optional but useful)
    if splats:
        hg = HierarchicalGMM(splats)
        tree_path = Path("config/canonical_tree.json")
        hg.save(tree_path)
        print(f"[SAVED] Runnalls Tree -> {tree_path}")

    print("\n[SUCCESS] Robot model built. Ready for online tracking.")


if __name__ == "__main__":
    main()