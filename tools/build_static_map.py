"""
tools/build_static_map.py - Build static background model (planes + splats).
Extracts dominant wall planes via RANSAC, then fits splats to residual points.
"""

import argparse
import json
import sys
import yaml
import numpy as np
import open3d as o3d
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from turtlebot_tracker.core.hierarchical_gmm import HierarchicalGMM
from turtlebot_tracker.core.mvi_clustering import MVIHierarchicalClustering, numpy_to_native
from turtlebot_tracker.core.radiometric import compute_radiometric_sh_c0


def extract_planes(points: np.ndarray, max_planes: int, distance_threshold: float = 0.05,
                   min_inliers: int = 50, max_angle_from_vertical: float = 30.0) -> list:
    """
    Extract dominant vertical wall planes using iterative RANSAC.
    Returns list of dicts: {'normal': [...], 'distance': ...}
    The number of planes extracted is limited by max_planes (upper bound).
    """
    if len(points) < 100:
        return []

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    planes = []
    remaining = pcd

    for _ in range(max_planes):
        if len(remaining.points) < min_inliers:
            break

        plane_model, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=200
        )
        if len(inliers) < min_inliers:
            break

        n = np.array(plane_model[:3], dtype=np.float64)
        n = n / np.linalg.norm(n)
        d = float(plane_model[3])

        # Ensure normal points away from origin (positive d)
        if d < 0:
            n = -n
            d = -d

        # Check if vertical (normal mostly horizontal)
        # Angle between normal and vertical (0,0,1)
        angle_deg = np.degrees(np.arccos(np.abs(n[2])))
        if angle_deg > max_angle_from_vertical:
            # Not a vertical wall; skip this plane and continue on remaining
            remaining = remaining.select_by_index(inliers, invert=True)
            continue

        planes.append({"normal": n.tolist(), "distance": d})
        remaining = remaining.select_by_index(inliers, invert=True)

    return planes


def main():
    parser = argparse.ArgumentParser(
        description="Build static map prior: ground plane + wall planes + splats for non-planar structures."
    )
    parser.add_argument("--input_npz", type=str, default="data/outputs/background_candidate.npz",
                        help="Background points NPZ (from select_canonical_candidate.py)")
    parser.add_argument("--metadata", type=str, default="data/outputs/static_metadata.json",
                        help="Metadata JSON with R_align, z_ground, etc.")
    parser.add_argument("--config", type=str, default="config/default_params.yaml",
                        help="Configuration YAML")
    parser.add_argument("--output", type=str, default="config/static_map_prior.json",
                        help="Output JSON path")
    parser.add_argument("--max_wall_planes", type=int, default=4,
                        help="Maximum number of wall planes to extract (upper bound)")
    parser.add_argument("--max_splats", type=int, default=15,
                        help="Maximum number of splats for non-planar structures (upper bound)")
    parser.add_argument("--voxel_size", type=float, default=0.025,
                        help="Voxel size for MVI (increase for fewer splats)")
    parser.add_argument("--plane_distance_threshold", type=float, default=0.05,
                        help="RANSAC distance threshold for plane extraction")
    parser.add_argument("--plane_min_inliers", type=int, default=100,
                        help="Minimum inliers to accept a wall plane")
    parser.add_argument("--splat_volume_threshold", type=float, default=2.0,
                        help="MVI volume threshold for splat clustering (higher = fewer splats)")
    args = parser.parse_args()

    # Check input files
    npz_path = Path(args.input_npz)
    if not npz_path.exists():
        print(f"[ERROR] Background NPZ not found: {npz_path}")
        return

    meta_path = Path(args.metadata)
    if not meta_path.exists():
        print(f"[ERROR] Metadata not found: {meta_path}")
        return

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    print(f"\n=======================================================")
    print(" BUILD STATIC MAP PRIOR (PLANES + SPLATS)")
    print("=======================================================\n")

    data = np.load(npz_path)
    bg_points = data["points"]
    bg_intensity = data["intensity"]

    R_align = np.array(metadata["R_align"])
    z_ground = metadata["z_ground"]
    ground_normal = np.array(metadata["ground_normal"])
    ground_distance = metadata["ground_distance"]

    print(f"[INFO] Loaded {len(bg_points):,} background points.")
    print(f"[INFO] z_ground = {z_ground:.3f} m")

    # ----- Extract wall planes from background points (aligned, Z-up) -----
    # Use points that are not ground (z > z_ground + 0.05)
    non_ground_mask = bg_points[:, 2] > z_ground + 0.05
    wall_candidate_points = bg_points[non_ground_mask]

    print(f"[INFO] Using {len(wall_candidate_points):,} points for wall plane extraction (max {args.max_wall_planes} planes).")

    wall_planes = extract_planes(
        wall_candidate_points,
        max_planes=args.max_wall_planes,
        distance_threshold=args.plane_distance_threshold,
        min_inliers=args.plane_min_inliers,
        max_angle_from_vertical=30.0
    )
    print(f"[INFO] Extracted {len(wall_planes)} wall planes.")

    # ----- Remove points belonging to wall planes from the background -----
    wall_mask = np.zeros(len(bg_points), dtype=bool)
    for plane in wall_planes:
        n = np.array(plane["normal"])
        d = plane["distance"]
        dist = np.abs(bg_points @ n + d)
        wall_mask |= dist <= args.plane_distance_threshold * 1.5  # margin

    # Residual points: not on walls and not on ground
    residual_mask = ~wall_mask & (bg_points[:, 2] > z_ground + 0.05)
    residual_points = bg_points[residual_mask]
    residual_intensity = bg_intensity[residual_mask]

    print(f"[INFO] Residual non-planar points after removing walls: {len(residual_points):,}")

    # ----- Generate splats from residual points using MVI -----
    if len(residual_points) > 200:
        pcd_res = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(residual_points))
        pcd_res_down, _, _ = pcd_res.voxel_down_sample_and_trace(
            args.voxel_size, pcd_res.get_min_bound(), pcd_res.get_max_bound()
        )
        res_pts = np.asarray(pcd_res_down.points)
        print(f"[INFO] Residual points after voxel: {len(res_pts):,}")

        if len(res_pts) > 200:
            # Run MVI clustering on residual points
            clusterer = MVIHierarchicalClustering(
                res_pts,
                k_neighbors=10,
                volume_threshold=args.splat_volume_threshold
            )
            final_clusters = clusterer.cluster(max_components=args.max_splats)

            splats = []
            total_pts = len(res_pts)
            for domain_id, indices in final_clusters.items():
                if len(indices) < 30:
                    continue
                cluster_pts = res_pts[indices]
                mu_k = np.mean(cluster_pts, axis=0)
                cov_k = np.cov(cluster_pts.T) + np.eye(3) * 1e-6
                eigvals, eigvecs = np.linalg.eigh(cov_k)
                eigvals = np.maximum(eigvals, 1e-5)
                scales_k = np.sqrt(eigvals)
                extents = np.max(cluster_pts, axis=0) - np.min(cluster_pts, axis=0)
                scales_k = np.minimum(scales_k, extents / 2.0 + 0.01)
                weight_k = float(len(indices) / total_pts)

                # Radiometric coefficient (optional, set to 0 if not needed)
                sh_c0 = 0.0

                splats.append({
                    "mu": mu_k.tolist(),
                    "cov": cov_k.tolist(),
                    "scales": scales_k.tolist(),
                    "rotation": eigvecs.tolist(),
                    "weight": weight_k,
                    "sh_c0": sh_c0
                })
            print(f"[INFO] Generated {len(splats)} splats for non-planar structures.")
        else:
            print("[WARN] Not enough residual points for MVI. Using empty splats.")
            splats = []
    else:
        print("[WARN] Not enough residual points for MVI. Using empty splats.")
        splats = []

    # Build hierarchical tree for splats (optional)
    tree = None
    if splats:
        hg = HierarchicalGMM(splats)
        tree = hg.tree

    # Prepare final JSON
    prior_data = {
        "R_align": R_align.tolist(),
        "z_ground": float(z_ground),
        "ground_normal": metadata["ground_normal"],
        "ground_distance": metadata["ground_distance"],
        "wall_planes": wall_planes,
        "splats": splats,
        "hierarchical_tree": tree,
        "description": "Static map prior: ground plane + wall planes + splats for non-planar structures."
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(prior_data, f, indent=4, default=numpy_to_native)

    print(f"\n[SUCCESS] Static map prior saved to: {output_path}")
    print(f"  - {len(wall_planes)} wall planes")
    print(f"  - {len(splats)} structure splats")
    print("\nNext: Run 'python tools/run_online_tracker.py' to use the prior.")


if __name__ == "__main__":
    main()