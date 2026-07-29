"""
tools/select_canonical_candidate.py - Script 1: Dense Accumulation & Raw Feature Exporter
Accumulates static frames, isolates candidate clusters, and exports XYZ coordinates, raw intensities,
and range distances to data/outputs/canonical_raw_candidate.npz.
Usage: python tools/select_canonical_candidate.py
"""

import argparse
import yaml
import numpy as np
import open3d as o3d
from pathlib import Path
from turtlebot_tracker.io.mcap_loader import MCAPLiDARLoader
from turtlebot_tracker.core.preprocessor import LiDARPreprocessor
from turtlebot_tracker.core.candidate_filter import CandidateFilter
from turtlebot_tracker.datatypes import ClusterCandidate

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
    parser = argparse.ArgumentParser(description="Dense accumulation and feature export")
    parser.add_argument("--bags_dir", type=str, default="data/bags", help="Path to bags directory")
    parser.add_argument("--config", type=str, default="config/default_params.yaml", help="Path to config")
    parser.add_argument("--output_npz", type=str, default="data/outputs/canonical_raw_candidate.npz", help="Output NPZ path")
    parser.add_argument("--max_frames", type=int, default=100, help="Number of static frames to accumulate")
    parser.add_argument("--auto", action="store_true", help="Auto-select candidate closest to target size")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    bags_dir = Path(args.bags_dir)
    static_bags = list(bags_dir.glob("*static_1m*"))
    if not static_bags:
        static_bags = list(bags_dir.glob("*"))

    if not static_bags:
        print("[ERROR] Static bag static_1m not found in", bags_dir)
        return

    static_bag_path = static_bags[0]
    print(f"\n=======================================================")
    print(f" SCRIPT 1: ACCUMULATING DENSE STATIC CLOUD FROM: {static_bag_path.name}")
    print(f"=======================================================\n")

    loader = MCAPLiDARLoader(str(static_bag_path))
    preprocessor = LiDARPreprocessor(cfg)
    candidate_filter = CandidateFilter(cfg)

    accumulated_obstacles = []
    accumulated_ground = []
    accumulated_intensities = []

    for idx, (ts, pts, intensity) in enumerate(loader.stream_point_clouds()):
        if idx >= args.max_frames:
            break
        frame_data = preprocessor.process(ts, pts, intensity)
        if frame_data.obstacle_points is not None and len(frame_data.obstacle_points) > 10:
            accumulated_obstacles.append(frame_data.obstacle_points)
            accumulated_intensities.append(frame_data.intensity[:len(frame_data.obstacle_points)])
        if frame_data.ground_points is not None and idx % 2 == 0:
            accumulated_ground.append(frame_data.ground_points)

    if not accumulated_obstacles:
        print("[ERROR] Failed to accumulate points.")
        return

    full_dense_obstacles = np.vstack(accumulated_obstacles)
    full_dense_intensities = np.concatenate(accumulated_intensities) if accumulated_intensities else np.ones(len(full_dense_obstacles)) * 100.0
    dense_ground = np.vstack(accumulated_ground) if accumulated_ground else None

    print(f"[INFO] Multi-Frame Dense Point Cloud Accumulated: {len(full_dense_obstacles):,} total points")

    # Fine voxel downsampling (6mm) for uniform surface coverage
    pcd_obs = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(full_dense_obstacles))
    pcd_obs_down, _, trace = pcd_obs.voxel_down_sample_and_trace(0.006, pcd_obs.get_min_bound(), pcd_obs.get_max_bound())
    down_pts = np.asarray(pcd_obs_down.points)

    down_indices = [t[0] for t in trace if len(t) > 0]
    down_intensities = full_dense_intensities[down_indices] if len(down_indices) == len(down_pts) else np.ones(len(down_pts)) * 100.0

    pcd_clustered = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(down_pts))
    labels = np.array(pcd_clustered.cluster_dbscan(eps=0.20, min_points=30))

    max_label = labels.max()
    all_clusters = []

    for i in range(max_label + 1):
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

    print("\n-------------------------------------------------------")
    print(" SEMANTICALLY CLASSIFIED ROBOT CANDIDATES:")
    print("-------------------------------------------------------")

    for cand in evaluated_clusters:
        extents = np.max(cand.points, axis=0) - np.min(cand.points, axis=0)
        
        if cand.passed_filters:
            cand_id = len(robot_candidates)
            target_diff = np.linalg.norm(extents - np.array([0.45, 0.45, 0.48]))
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

            print(f" [Candidate #{cand_id} - Color: {palette_entry['name']:14s}] "
                  f"Points: {len(cand.points):5d} | "
                  f"Extents [X,Y,Z]: [{extents[0]:.2f}, {extents[1]:.2f}, {extents[2]:.2f}] m | "
                  f"Centroid: [{cand.centroid[0]:.2f}, {cand.centroid[1]:.2f}, {cand.centroid[2]:.2f}] m")
        else:
            structure_clusters.append(cand)

    print(f"\n[SEMANTIC SUMMARY] Robot Candidates: {len(robot_candidates)} | Walls/Structures: {len(structure_clusters)}")

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
                "target_diff": np.linalg.norm(extents - np.array([0.45, 0.45, 0.48])),
                "color_name": palette_entry["name"],
                "color_rgb": palette_entry["rgb"]
            })

    auto_selected = min(robot_candidates, key=lambda c: c["target_diff"])

    # Open3D Native Inspection Window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Script 1: Select Robot Candidate Cluster", width=1280, height=720)

    if dense_ground is not None:
        pcd_ground = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dense_ground))
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
    print(" [INSPECTION WINDOW OPENED]")
    print(f" Auto-recommended Candidate #{auto_selected['id']} ({auto_selected['color_name']}).")
    print(" CLOSE THE 3D WINDOW (or press 'Q') when ready to enter selection.")
    print("-------------------------------------------------------")

    vis.run()
    vis.destroy_window()

    if args.auto:
        selected_id = auto_selected["id"]
        print(f"\n[AUTO-MODE] Selected Candidate #{selected_id} ({auto_selected['color_name']})")
    else:
        user_input = input(f"\nEnter Candidate ID [0-{len(robot_candidates)-1}] to extract [Press ENTER for #{auto_selected['id']} ({auto_selected['color_name']})]: ").strip()
        if user_input.isdigit() and int(user_input) < len(robot_candidates):
            selected_id = int(user_input)
        else:
            selected_id = auto_selected["id"]

    selected_cand = robot_candidates[selected_id]
    
    # Extract ALL raw accumulated points and intensities belonging to chosen bounding box
    c_centroid = selected_cand["centroid"]
    c_extents = selected_cand["extents"]
    min_box = c_centroid - (c_extents / 2.0) - 0.05
    max_box = c_centroid + (c_extents / 2.0) + 0.05

    in_box_mask = np.all((full_dense_obstacles >= min_box) & (full_dense_obstacles <= max_box), axis=1)
    dense_candidate_points = full_dense_obstacles[in_box_mask]
    dense_candidate_intensities = full_dense_intensities[in_box_mask]

    print(f"\n[SELECTED] Candidate #{selected_id} ({selected_cand['color_name']})")
    print(f"[EXTRACTED] Exporting {len(dense_candidate_points):,} dense points with intensity features.")

    out_npz_path = Path(args.output_npz)
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz_path, points=dense_candidate_points, intensity=dense_candidate_intensities)

    print(f"[SUCCESS] Saved dense candidate NPZ to: {out_npz_path}")
    print("Next step: Run 'python tools/build_canonical_model.py' for Organic Crystal Growth GMM fitting!")

if __name__ == "__main__":
    main()