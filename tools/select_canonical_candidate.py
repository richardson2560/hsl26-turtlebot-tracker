"""
select_canonical_candidate.py - Dense Static Accumulation and Candidate Extraction.
Now also saves background points and metadata for static map building.
"""

import argparse
import json
from pathlib import Path
import sys
import yaml
import numpy as np
import open3d as o3d

sys.path.append(str(Path(__file__).parent.parent / "src"))

from turtlebot_tracker.core.candidate_filter import CandidateFilter
from turtlebot_tracker.core.preprocessor import LiDARPreprocessor
from turtlebot_tracker.datatypes import ClusterCandidate
from turtlebot_tracker.io.mcap_loader import MCAPLiDARLoader

COLOR_PALETTE = [
    {"name": "BRIGHT_GREEN",  "rgb": [0.0, 1.0, 0.0]},
    {"name": "BRIGHT_YELLOW", "rgb": [1.0, 0.95, 0.0]},
    {"name": "CYAN",          "rgb": [0.0, 0.8, 1.0]},
    {"name": "MAGENTA",       "rgb": [0.8, 0.0, 1.0]},
    {"name": "ORANGE",        "rgb": [1.0, 0.4, 0.0]},
    {"name": "RED",           "rgb": [1.0, 0.2, 0.2]}
]

def build_bounding_box_lineset(centroid: np.ndarray, extents: np.ndarray, color: list) -> o3d.geometry.LineSet:
    dx, dy, dz = extents[0] / 2.0, extents[1] / 2.0, extents[2] / 2.0
    corners = np.array([
        [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
        [-dx, -dy, dz],  [dx, -dy, dz],  [dx, dy, dz],  [-dx, dy, dz]
    ])
    world_corners = corners + centroid
    lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(world_corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    return line_set

def main():
    parser = argparse.ArgumentParser(description="Dense static accumulation and candidate extraction")
    parser.add_argument("--config", type=str, default="config/default_params.yaml")
    parser.add_argument("--output_robot", type=str, default="data/outputs/robot_candidate.npz")
    parser.add_argument("--output_background", type=str, default="data/outputs/background_candidate.npz")
    parser.add_argument("--output_metadata", type=str, default="data/outputs/static_metadata.json")
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    static_bags = sorted(list(Path("data/bags").glob("*static_1m*")))
    if not static_bags:
        static_bags = sorted(list(Path("data/bags").glob("*")))

    print(f"\n=======================================================")
    print(f" ACCUMULATING DENSE STATIC CLOUD FROM: {static_bags[0].name}")
    print(f"=======================================================\n")

    loader = MCAPLiDARLoader(str(static_bags[0]))
    preprocessor = LiDARPreprocessor(cfg)
    candidate_filter = CandidateFilter(cfg)

    accumulated_obstacles = []
    accumulated_intensities = []
    accumulated_ground = []

    for idx, (ts, pts, intensity) in enumerate(loader.stream_point_clouds()):
        if idx >= args.max_frames:
            break
        frame_data = preprocessor.process(ts, pts, intensity)
        if frame_data.obstacle_points is not None and len(frame_data.obstacle_points) > 10:
            accumulated_obstacles.append(frame_data.obstacle_points)
            accumulated_intensities.append(frame_data.intensity[:len(frame_data.obstacle_points)])
        if frame_data.ground_points is not None:
            accumulated_ground.append(frame_data.ground_points)

    if not accumulated_obstacles:
        print("[ERROR] Failed to accumulate points.")
        return

    full_obstacles = np.vstack(accumulated_obstacles)
    full_intensities = np.concatenate(accumulated_intensities)
    full_ground = np.vstack(accumulated_ground) if accumulated_ground else None

    print(f"[INFO] Multi-Frame Dense Point Cloud Accumulated: {len(full_obstacles):,} total points")

    # --- Get ground alignment from the last preprocessor (or recompute) ---
    # We'll reuse the R_align and z_ground from the last frame (assuming static)
    # For robustness, we could recompute from accumulated ground, but this is fine.
    # We'll get it from the preprocessor's internal state if available, or recompute.
    # For now, recompute from accumulated ground:
    if full_ground is not None and len(full_ground) > 100:
        pcd_g = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(full_ground))
        plane_model, inliers = pcd_g.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=200)
        n = np.array(plane_model[:3], dtype=np.float64)
        n = n / np.linalg.norm(n)
        d = float(plane_model[3])
        if n[2] < 0:
            n = -n
            d = -d
        target = np.array([0.0, 0.0, 1.0])
        v = np.cross(n, target)
        s = np.linalg.norm(v)
        c = np.dot(n, target)
        if s < 1e-6:
            R_align = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R_align = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s ** 2))
        z_ground = float(np.mean(full_ground[inliers, 2]))
    else:
        print("[ERROR] Could not extract ground plane from accumulated ground.")
        return

    # Align all obstacle points
    aligned_obs = (R_align @ full_obstacles.T).T

    # --- Candidate detection (same as before) ---
    pcd_obs = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned_obs))
    pcd_obs_down, _, trace = pcd_obs.voxel_down_sample_and_trace(0.006, pcd_obs.get_min_bound(), pcd_obs.get_max_bound())
    down_pts = np.asarray(pcd_obs_down.points, dtype=np.float64)
    down_indices = [t[0] for t in trace if len(t) > 0]
    down_intensities = full_intensities[down_indices] if len(down_indices) == len(down_pts) else np.ones(len(down_pts)) * 100.0

    pcd_clustered = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(down_pts))
    labels = np.array(pcd_clustered.cluster_dbscan(eps=0.20, min_points=30))

    all_clusters = []
    for i in range(labels.max() + 1):
        c_mask = (labels == i)
        c_pts = down_pts[c_mask]
        c_int = down_intensities[c_mask]
        if len(c_pts) < 30:
            continue
        c_obj = ClusterCandidate(
            id=i,
            points=c_pts,
            intensity=c_int,
            centroid=np.mean(c_pts, axis=0)
        )
        all_clusters.append(c_obj)

    evaluated_clusters = candidate_filter.filter_candidates(all_clusters)

    robot_candidates = []
    structure_clusters = []
    for cand in evaluated_clusters:
        extents = np.max(cand.points, axis=0) - np.min(cand.points, axis=0)
        if cand.passed_filters:
            cand_id = len(robot_candidates)
            target_diff = np.linalg.norm(extents - np.array([0.38, 0.38, 0.42]))
            palette_entry = COLOR_PALETTE[cand_id % len(COLOR_PALETTE)]
            cand_data = {
                "id": cand_id,
                "points_down": cand.points,
                "intensity_down": cand.intensity,
                "centroid": cand.centroid,
                "extents": extents,
                "target_diff": target_diff,
                "color_name": palette_entry["name"],
                "color_rgb": palette_entry["rgb"]
            }
            robot_candidates.append(cand_data)
        else:
            structure_clusters.append(cand)

    if not robot_candidates:
        print("[WARNING] Fallback to top local clusters.")
        for idx, cand in enumerate(evaluated_clusters[:5]):
            extents = np.max(cand.points, axis=0) - np.min(cand.points, axis=0)
            palette_entry = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
            robot_candidates.append({
                "id": idx,
                "points_down": cand.points,
                "intensity_down": cand.intensity,
                "centroid": cand.centroid,
                "extents": extents,
                "target_diff": np.linalg.norm(extents - np.array([0.38, 0.38, 0.42])),
                "color_name": palette_entry["name"],
                "color_rgb": palette_entry["rgb"]
            })

    auto_selected = min(robot_candidates, key=lambda c: c["target_diff"])

    # --- Visualization (optional, for manual selection) ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Select Robot Candidate", width=1280, height=720)
    if full_ground is not None:
        pcd_ground = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(full_ground))
        pcd_ground_down = pcd_ground.voxel_down_sample(voxel_size=0.04)
        pcd_ground_down.paint_uniform_color([0.2, 0.25, 0.35])
        vis.add_geometry(pcd_ground_down)
    for struct in structure_clusters:
        pcd_s = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(struct.points))
        pcd_s.paint_uniform_color([0.35, 0.35, 0.40])
        vis.add_geometry(pcd_s)
    for cand in robot_candidates:
        pcd_c = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cand["points_down"]))
        pcd_c.paint_uniform_color(cand["color_rgb"])
        vis.add_geometry(pcd_c)
        bbox = build_bounding_box_lineset(cand["centroid"], cand["extents"], cand["color_rgb"])
        vis.add_geometry(bbox)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(coord_frame)

    print("\n-------------------------------------------------------")
    print(f" Auto-recommended Candidate #{auto_selected['id']} ({auto_selected['color_name']}).")
    print(" CLOSE THE 3D WINDOW when ready to proceed.")
    print("-------------------------------------------------------")
    vis.run()
    vis.destroy_window()

    if args.auto:
        selected_id = auto_selected["id"]
        print(f"\n[AUTO-MODE] Selected Candidate #{selected_id} ({auto_selected['color_name']})")
    else:
        user_input = input(f"\nEnter Candidate ID [0-{len(robot_candidates)-1}] [ENTER for #{auto_selected['id']}]: ").strip()
        if user_input.isdigit() and int(user_input) < len(robot_candidates):
            selected_id = int(user_input)
        else:
            selected_id = auto_selected["id"]

    selected_cand = robot_candidates[selected_id]

    # --- Extract robot and background ---
    c_centroid = selected_cand["centroid"]
    c_extents = selected_cand["extents"]
    min_box = c_centroid - (c_extents / 2.0) - 0.05
    max_box = c_centroid + (c_extents / 2.0) + 0.05

    in_robot_mask = np.all((aligned_obs >= min_box) & (aligned_obs <= max_box), axis=1)
    robot_points = aligned_obs[in_robot_mask]
    robot_intensities = full_intensities[in_robot_mask]
    background_points = aligned_obs[~in_robot_mask]
    background_intensities = full_intensities[~in_robot_mask]

    print(f"\n[SELECTED] Candidate #{selected_id} ({selected_cand['color_name']})")
    print(f"[EXTRACTED] Robot: {len(robot_points):,} points, Background: {len(background_points):,} points")

    # --- Save outputs ---
    out_robot = Path(args.output_robot)
    out_robot.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_robot, points=robot_points, intensity=robot_intensities)
    print(f"[SAVED] Robot NPZ -> {out_robot}")

    out_bg = Path(args.output_background)
    out_bg.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_bg, points=background_points, intensity=background_intensities)
    print(f"[SAVED] Background NPZ -> {out_bg}")

    # Metadata
    metadata = {
        "R_align": R_align.tolist(),
        "z_ground": float(z_ground),
        "ground_normal": n.tolist(),
        "ground_distance": float(d),
        "robot_centroid": c_centroid.tolist(),
        "robot_extents": c_extents.tolist()
    }
    out_meta = Path(args.output_metadata)
    with open(out_meta, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"[SAVED] Metadata -> {out_meta}")

    print("\n[SUCCESS] Static accumulation complete.")
    print("Next steps:")
    print("  python tools/build_canonical_model.py --input_npz data/outputs/robot_candidate.npz")
    print("  python tools/build_static_map.py --input_npz data/outputs/background_candidate.npz --metadata data/outputs/static_metadata.json")


if __name__ == "__main__":
    main()