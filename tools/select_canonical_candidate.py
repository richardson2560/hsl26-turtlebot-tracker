"""
select_canonical_candidate.py - Semantic Accumulator & Labeler (Expert).
Labels: 0=Ground, 1=Structure, 2=Static Candidate, 3=Robot.
Visualization uses draw_geometries with consistent colors and centroid markers.
"""

import argparse
import json
import sys
import yaml
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial import KDTree

sys.path.append(str(Path(__file__).parent.parent / "src"))

from turtlebot_tracker.core.candidate_filter import CandidateFilter
from turtlebot_tracker.core.preprocessor import LiDARPreprocessor
from turtlebot_tracker.datatypes import ClusterCandidate
from turtlebot_tracker.io.mcap_loader import MCAPLiDARLoader

# ---- Color palette (consistent with console table) ----
COLOR_NAMES = ["Green", "Yellow", "Cyan", "Magenta", "Orange", "Red", "Purple", "Lime"]
COLOR_PALETTE = [
    [0.0, 1.0, 0.0],   # Green
    [1.0, 0.9, 0.0],   # Yellow
    [0.0, 0.8, 1.0],   # Cyan
    [0.8, 0.0, 1.0],   # Magenta
    [1.0, 0.4, 0.0],   # Orange
    [1.0, 0.2, 0.2],   # Red
    [0.6, 0.2, 0.8],   # Purple
    [0.6, 0.8, 0.2],   # Lime
]


def get_alignment_matrix(normal):
    """Rodrigues rotation to align normal to Z+."""
    target = np.array([0, 0, 1.0])
    v = np.cross(normal, target)
    s = np.linalg.norm(v)
    c = np.dot(normal, target)
    if s < 1e-6:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s ** 2))


def auto_select_robot(valid_candidates):
    """Select candidate closest to Turtlebot2 dimensions (~0.38m)."""
    target_extents = np.array([0.38, 0.38, 0.42])
    best_idx = 0
    best_score = float('inf')
    for i, cand in enumerate(valid_candidates):
        ext = np.max(cand.points, axis=0) - np.min(cand.points, axis=0)
        score = np.linalg.norm(ext - target_extents)
        if score < best_score:
            best_score = score
            best_idx = i
    return best_idx


def main():
    parser = argparse.ArgumentParser(description="Expert Semantic Accumulator")
    parser.add_argument("--config", type=str, default="config/default_params.yaml")
    parser.add_argument("--output_npz", type=str, default="data/outputs/static_full.npz")
    parser.add_argument("--output_metadata", type=str, default="data/outputs/static_metadata.json")
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--voxel_clustering", type=float, default=0.015)
    parser.add_argument("--auto", action="store_true", help="Skip visual selection")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    static_bags = sorted(list(Path("data/bags").glob("*static_1m*")))
    if not static_bags:
        print("[ERROR] No static_1m bag found.")
        return

    loader = MCAPLiDARLoader(str(static_bags[0]))
    preprocessor = LiDARPreprocessor(cfg)
    candidate_filter = CandidateFilter(cfg)

    obs_pts, ground_pts, intensities = [], [], []

    print(f"[INFO] Accumulating {args.max_frames} frames from {static_bags[0].name}...")
    for idx, (ts, pts, intensity) in enumerate(loader.stream_point_clouds()):
        if idx >= args.max_frames:
            break
        frame = preprocessor.process(ts, pts, intensity)
        if frame.obstacle_points is not None and len(frame.obstacle_points) > 10:
            obs_pts.append(frame.obstacle_points)
            intensities.append(frame.intensity[:len(frame.obstacle_points)])
        if frame.ground_points is not None and len(frame.ground_points) > 10:
            ground_pts.append(frame.ground_points)

    if not obs_pts:
        print("[ERROR] No obstacle points accumulated.")
        return

    full_obs = np.vstack(obs_pts)
    full_int = np.concatenate(intensities)
    full_ground = np.vstack(ground_pts) if ground_pts else None

    print(f"[INFO] Accumulated {len(full_obs):,} obstacle points.")

    # --- Ground Plane and Alignment ---
    if full_ground is not None and len(full_ground) > 100:
        pcd_g = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(full_ground))
        plane_model, inliers = pcd_g.segment_plane(0.05, 3, 250)
        normal = np.array(plane_model[:3])
        normal = normal / np.linalg.norm(normal)
        if normal[2] < 0:
            normal = -normal
        d = float(plane_model[3]) / np.linalg.norm(plane_model[:3])
        if d < 0:
            normal = -normal
            d = -d
        R_align = get_alignment_matrix(normal)

        aligned_obs = (R_align @ full_obs.T).T
        aligned_ground = (R_align @ full_ground.T).T
        ground_z_vals = aligned_ground[inliers, 2]
        z_ground = float(np.mean(ground_z_vals))
        ground_thick = cfg.get('segmentation', {}).get('ground_thickness', 0.05)
    else:
        print("[ERROR] Could not extract ground plane.")
        return

    # --- DBSCAN Clustering ---
    pcd_obs = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned_obs))
    pcd_down, _, trace = pcd_obs.voxel_down_sample_and_trace(
        args.voxel_clustering, pcd_obs.get_min_bound(), pcd_obs.get_max_bound()
    )
    down_pts = np.asarray(pcd_down.points)
    dbscan_labels = np.array(pcd_down.cluster_dbscan(eps=0.20, min_points=25))

    clusters = []
    for i in range(dbscan_labels.max() + 1):
        mask = dbscan_labels == i
        c_pts = down_pts[mask]
        if len(c_pts) < 25:
            continue
        c_int = np.ones(len(c_pts)) * 100.0
        clusters.append(ClusterCandidate(
            id=i,
            points=c_pts,
            intensity=c_int,
            centroid=np.mean(c_pts, axis=0)
        ))

    evaluated = candidate_filter.filter_candidates(clusters)

    # --- Label propagation ---
    label_map = {-1: 1}  # Noise -> Structure
    valid_candidates = []
    for cand in evaluated:
        label_map[cand.id] = 2 if cand.passed_filters else 1
        if cand.passed_filters:
            valid_candidates.append(cand)

    down_labels = np.array([label_map.get(idx, 1) for idx in dbscan_labels])
    tree = KDTree(down_pts)
    _, nn_idx = tree.query(aligned_obs)
    obs_labels = down_labels[nn_idx]

    # --- Robot Selection ---
    if not valid_candidates:
        print("[WARN] No valid candidates found. Using fallback.")
        robot_candidate = None
    else:
        if args.auto:
            selected_id = auto_select_robot(valid_candidates)
            print(f"[AUTO] Selected robot candidate {selected_id} based on size.")
        else:
            # --- Print candidate table ---
            print("\n" + "=" * 65)
            print(f"{'ID':^5} | {'COLOR':^12} | {'POINTS':^8} | {'EXTENTS (X, Y, Z)'}")
            print("-" * 65)
            for i, cand in enumerate(valid_candidates):
                ext = np.max(cand.points, axis=0) - np.min(cand.points, axis=0)
                color_name = COLOR_NAMES[i % len(COLOR_NAMES)]
                print(f"{i:^5} | {color_name:^12} | {len(cand.points):^8} | "
                      f"({ext[0]:.3f}, {ext[1]:.3f}, {ext[2]:.3f})")
            print("=" * 65)

            # --- Visualization with draw_geometries (compatible) ---
            geometries = []

            # Ground reference (dark gray)
            if len(aligned_ground) > 0:
                pcd_bg = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned_ground))
                pcd_bg = pcd_bg.voxel_down_sample(0.05)
                pcd_bg.paint_uniform_color([0.15, 0.15, 0.15])
                geometries.append(pcd_bg)

            # Candidates with colors and centroid markers
            for i, cand in enumerate(valid_candidates):
                pcd_c = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cand.points))
                color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
                pcd_c.paint_uniform_color(color)
                geometries.append(pcd_c)

                # Add a sphere at centroid for easy spotting
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                sphere.translate(cand.centroid)
                sphere.paint_uniform_color([1.0, 1.0, 1.0])  # white marker
                geometries.append(sphere)

            # Coordinate frame
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            geometries.append(coord)

            print("\n[ACTION] Inspect the 3D window. Each cluster has a unique color matching the table.")
            print("[INFO] Close the window to enter your selection in the terminal.")

            # Use draw_geometries (compatible with all Open3D versions)
            o3d.visualization.draw_geometries(
                geometries,
                window_name="Select Robot Candidate (see console table)",
                width=1280,
                height=720,
                lookat=[0, 0, 0],
                front=[0, 0, -1],
                up=[0, -1, 0],
                zoom=0.6
            )

            user_input = input(f"\nEnter Robot ID (0-{len(valid_candidates)-1}): ")
            try:
                selected_id = int(user_input)
                selected_id = max(0, min(selected_id, len(valid_candidates)-1))
            except ValueError:
                selected_id = auto_select_robot(valid_candidates)
                print(f"[AUTO] Invalid input, using auto-selected {selected_id}.")

        robot_candidate = valid_candidates[selected_id]

    # --- Mark Robot (Label 3) ---
    if robot_candidate is not None:
        centroid = robot_candidate.centroid
        obs_tree = KDTree(aligned_obs)
        candidate_indices = obs_tree.query_ball_point(centroid, r=0.5)
        for idx in candidate_indices:
            if obs_labels[idx] == 2:  # promote only candidates
                obs_labels[idx] = 3
        robot_centroid = centroid
    else:
        robot_centroid = [0, 0, 0]

    # --- Combine Ground + Obstacles ---
    all_points = np.vstack([aligned_ground, aligned_obs])
    all_labels = np.concatenate([np.zeros(len(aligned_ground), dtype=np.int32), obs_labels])
    all_int = np.concatenate([np.full(len(aligned_ground), 10.0), full_int])

    print(f"\n[INFO] Final label distribution:")
    print(f"  Ground    : {np.sum(all_labels == 0):,}")
    print(f"  Structure : {np.sum(all_labels == 1):,}")
    print(f"  Candidate : {np.sum(all_labels == 2):,}")
    print(f"  Robot     : {np.sum(all_labels == 3):,}")

    # --- Save ---
    out_npz = Path(args.output_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz,
             points=all_points.astype(np.float32),
             labels=all_labels.astype(np.int32),
             intensity=all_int.astype(np.float32))
    print(f"[SAVED] Semantic NPZ -> {out_npz}")

    meta = {
        "R_align": R_align.tolist(),
        "z_ground": z_ground,
        "ground_normal": normal.tolist(),
        "ground_distance": d,
        "ground_thickness": ground_thick,
        "robot_centroid": robot_centroid.tolist(),
        "label_counts": {
            "ground": int(np.sum(all_labels == 0)),
            "structure": int(np.sum(all_labels == 1)),
            "candidate": int(np.sum(all_labels == 2)),
            "robot": int(np.sum(all_labels == 3))
        }
    }
    out_meta = Path(args.output_metadata)
    with open(out_meta, 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"[SAVED] Metadata -> {out_meta}")

    print("\n[SUCCESS] Static accumulation complete.")
    print("Next steps:")
    print("  python tools/build_canonical_model.py")
    print("  python tools/build_static_map.py")


if __name__ == "__main__":
    main()