#!/usr/bin/env python3
"""
visualize_online_segmentation.py - Final KISS visualizer with snap-on-target.
"""

import argparse
import json
import sys
import time
import types
import numpy as np
import open3d as o3d
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from turtlebot_tracker.io.mcap_loader import MCAPLiDARLoader
from turtlebot_tracker.core.online_segmenter import OnlineSegmenter
from turtlebot_tracker.core.registration import GPISRegistrator
from turtlebot_tracker.core.tracking import SE2ManifoldEKF
from turtlebot_tracker.datatypes import LifecycleState


def create_oriented_bbox(center, R_mat, extent=[0.45, 0.45, 0.48], color=[0.0, 1.0, 0.0]):
    dx, dy, dz = extent[0] / 2.0, extent[1] / 2.0, extent[2] / 2.0
    corners_local = np.array([
        [-dx, -dy, -dz], [ dx, -dy, -dz], [ dx,  dy, -dz], [-dx,  dy, -dz],
        [-dx, -dy,  dz], [ dx, -dy,  dz], [ dx,  dy,  dz], [-dx,  dy,  dz]
    ], dtype=np.float64)
    corners_world = (R_mat @ corners_local.T).T + center
    lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return line_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=str, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--model", type=str, default="config/implicit_model.json")
    parser.add_argument("--prior", type=str, default="config/static_map_prior.json")
    parser.add_argument("--metadata", type=str, default="data/outputs/static_metadata.json")
    args = parser.parse_args()

    with open(args.prior, 'r') as f:
        prior = json.load(f)
    with open(args.metadata, 'r') as f:
        meta = json.load(f)

    Z_GROUND = meta['z_ground']
    print(f"[INFO] Loaded z_ground: {Z_GROUND:.3f} m")

    # Core components
    registrator = GPISRegistrator({"registration": {"score_threshold": 2.0}}, args.model)
    segmenter = OnlineSegmenter(prior, meta)

    # EKF with KISS settings
    config = {"tracking": {}}
    ekf = SE2ManifoldEKF(config)
    ekf.z_ground = Z_GROUND

    # Load bag
    p = Path(args.bag)
    if p.is_dir():
        mcap_files = list(p.glob("*.mcap"))
        if not mcap_files:
            print(f"[ERROR] No .mcap file found in {p}")
            return
        mcap_path = str(mcap_files[0])
    else:
        mcap_path = str(p)

    loader = MCAPLiDARLoader(mcap_path)

    # Open3D setup
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="MOCD-Lite: KISS Final", width=1280, height=720)
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.02, 0.02, 0.02])
    render_opt.point_size = 3.0

    # Static background shells
    for s in prior.get('shells', []):
        extents = np.array(s['extents']) * 2.0
        if np.any(extents <= 0):
            continue
        box = o3d.geometry.OrientedBoundingBox(s['center'], np.array(s['axes']), extents)
        wire = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
        wire.paint_uniform_color([0.3, 0.3, 0.3])
        vis.add_geometry(wire)

    # Dynamic geometries
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    target_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
    target_marker.paint_uniform_color([1.0, 0.0, 0.0])
    vis.add_geometry(target_marker)

    bbox_geom = o3d.geometry.LineSet()
    vis.add_geometry(bbox_geom)

    # Trajectory as a LineSet (breadcrumbs)
    trajectory_line = o3d.geometry.LineSet()
    trajectory_line.paint_uniform_color([1.0, 0.0, 0.0])
    vis.add_geometry(trajectory_line)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(coord_frame)

    print("\n" + "=" * 120)
    print(f"{'FRAME':^6} | {'STATUS':^14} | {'HEALTH':^6} | {'ACCEPTED':^8} | "
          f"{'POS (X, Y)':^14} | {'N_PTS':^5} | {'YAW':^6}")
    print("-" * 120)

    frame_idx = 0
    last_timestamp = None

    for ts, pts, intensity in loader.stream_point_clouds():
        loop_start = time.perf_counter()

        dt = 0.1 if last_timestamp is None else max(0.01, ts - last_timestamp)
        last_timestamp = ts

        # 1. Segment
        pts_world, labels, clusters = segmenter.classify_and_cluster(pts)

        # 2. Register (GPIS-W) and update EKF
        frame_data = types.SimpleNamespace(z_ground=Z_GROUND)
        state, best_cand, accepted = registrator.register_and_track(
            frame_data, clusters, ekf, dt=dt
        )

        # 3. Colorize point cloud
        colors = np.zeros((len(pts_world), 3))
        colors[labels == 0] = [0.1, 0.1, 0.3]   # Ground
        colors[labels == 1] = [0.2, 0.2, 0.2]   # Wall
        colors[labels == 2] = [0.7, 0.1, 0.1]   # Static object

        for c in clusters:
            colors[c['indices']] = [0.0, 0.8, 1.0]  # Cyan for candidates

        if best_cand is not None:
            if accepted:
                colors[best_cand['indices']] = [0.0, 1.0, 0.0]   # Green
            else:
                colors[best_cand['indices']] = [1.0, 1.0, 0.0]   # Yellow

        pcd.points = o3d.utility.Vector3dVector(pts_world)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(pcd)

        # 4. Update marker, bbox, and trajectory
        if state.lifecycle_state in (LifecycleState.ACTIVE_TRACKING, LifecycleState.COASTING_LOST):
            pose = state.pose_se2
            z_abs = state.z

            # Marker color: green if tracking, orange if coasting
            if state.lifecycle_state == LifecycleState.ACTIVE_TRACKING:
                target_marker.paint_uniform_color([0.0, 1.0, 0.0])
            else:
                target_marker.paint_uniform_color([1.0, 0.5, 0.0])

            target_marker.translate([pose[0], pose[1], z_abs], relative=False)
            vis.update_geometry(target_marker)

            # Bounding box
            yaw = pose[2]
            R_mat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw),  np.cos(yaw), 0.0],
                [0.0,          0.0,         1.0]
            ], dtype=np.float64)
            center = np.array([pose[0], pose[1], z_abs], dtype=np.float64)
            bbox = create_oriented_bbox(center, R_mat, color=[0.0, 1.0, 0.0])
            bbox_geom.points = bbox.points
            bbox_geom.lines = bbox.lines
            bbox_geom.colors = bbox.colors
            vis.update_geometry(bbox_geom)

            # Trajectory (breadcrumbs) – fixed dimension issue
            if len(ekf.trajectory_log) > 1:
                # Build 3D points: use state.z for height (Z is constant for the session)
                traj_3d = np.array([
                    [p[0], p[1], z_abs] for p in ekf.trajectory_log
                ], dtype=np.float64)
                lines = [[i, i+1] for i in range(len(traj_3d) - 1)]
                trajectory_line.points = o3d.utility.Vector3dVector(traj_3d)
                trajectory_line.lines = o3d.utility.Vector2iVector(lines)
                # Color is already set to red
                vis.update_geometry(trajectory_line)

        else:
            # Hide marker and bbox
            target_marker.translate([100, 100, 100], relative=False)
            vis.update_geometry(target_marker)
            bbox_geom.points = o3d.utility.Vector3dVector([])
            bbox_geom.lines = o3d.utility.Vector2iVector([])
            vis.update_geometry(bbox_geom)

        # 5. Console log
        status_str = state.lifecycle_state.name
        health = state.reliability_score if hasattr(state, 'reliability_score') else ekf.health
        accepted_str = "YES" if accepted else "NO"
        n_pts = len(best_cand['points']) if best_cand is not None else 0

        if best_cand is not None:
            pose = state.pose_se2
            yaw_deg = np.degrees(pose[2])
            print(f"{frame_idx:6d} | {status_str:14s} | {health:6d} | "
                  f"{accepted_str:^8} | ({pose[0]:5.2f},{pose[1]:5.2f}) | "
                  f"{n_pts:5d} | {yaw_deg:6.1f}°")
        else:
            print(f"{frame_idx:6d} | {status_str:14s} | {health:6d} | "
                  f"{'---':^8} | {'---':^14} | {'---':^5} | {'---':^6}")

        vis.poll_events()
        vis.update_renderer()

        frame_idx += 1
        elapsed = time.perf_counter() - loop_start
        time.sleep(max(0, (1.0 / args.fps) - elapsed))

    vis.destroy_window()
    print("\n[DONE] Visualization finished.")


if __name__ == "__main__":
    main()