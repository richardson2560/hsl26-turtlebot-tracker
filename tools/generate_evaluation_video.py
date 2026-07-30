"""
tools/generate_evaluation_video.py - Evaluation Video Generator
Uses corrected Z and camera orientation.
"""

import argparse
import yaml
import numpy as np
import open3d as o3d
from pathlib import Path
from turtlebot_tracker.io.mcap_loader import MCAPLiDARLoader
from turtlebot_tracker.core.preprocessor import LiDARPreprocessor
from turtlebot_tracker.core.segmentation import RangeImageSegmenter
from turtlebot_tracker.core.candidate_filter import CandidateFilter
from turtlebot_tracker.core.registration import DirectGMMRegistrator
from turtlebot_tracker.core.tracking import SE2ManifoldEKF
from turtlebot_tracker.datatypes import LifecycleState

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/default_params.yaml")
    parser.add_argument("--output", type=str, default="data/outputs/evaluation_demo.mp4")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    loader = MCAPLiDARLoader(args.bag)
    preprocessor = LiDARPreprocessor(cfg)
    segmenter = RangeImageSegmenter(cfg)
    candidate_filter = CandidateFilter(cfg)
    registrator = DirectGMMRegistrator(cfg)
    ekf = SE2ManifoldEKF(cfg)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Evaluation", width=1280, height=720, visible=False)

    print(f"[INFO] Generating video from: {args.bag}")
    image_list = []

    for frame_idx, (ts, pts, intensity) in enumerate(loader.stream_point_clouds()):
        frame_data = preprocessor.process(ts, pts, intensity)
        clusters = segmenter.segment(frame_data)
        candidates = candidate_filter.filter_candidates(clusters)
        state, target = registrator.register_and_track(frame_data, candidates, ekf)

        vis.clear_geometries()

        # Coordinate frame
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        vis.add_geometry(coord, reset_bounding_box=False)

        # Obstacles
        if frame_data.obstacle_points is not None and len(frame_data.obstacle_points) > 0:
            pcd_obs = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(frame_data.obstacle_points))
            pcd_obs.paint_uniform_color([0.3, 0.3, 0.3])
            vis.add_geometry(pcd_obs, reset_bounding_box=False)

        # Target
        if target is not None:
            pcd_target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target.points))
            pcd_target.paint_uniform_color([1.0, 0.95, 0.0])
            vis.add_geometry(pcd_target, reset_bounding_box=False)

        # BBox
        is_tracking = state.lifecycle_state in (LifecycleState.ACTIVE_TRACKING, LifecycleState.COASTING_LOST)
        if is_tracking:
            x, y = state.pose_se2[0], state.pose_se2[1]
            yaw = state.pose_se2[2]
            z = state.z
            dx, dy, dz = 0.225, 0.225, 0.24
            corners_local = np.array([
                [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
                [-dx, -dy, dz],  [dx, -dy, dz],  [dx, dy, dz],  [-dx, dy, dz]
            ])
            R = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                          [np.sin(yaw),  np.cos(yaw), 0.0],
                          [0.0,          0.0,         1.0]])
            world_corners = (R @ corners_local.T).T + np.array([x, y, z])
            lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(world_corners),
                lines=o3d.utility.Vector2iVector(lines)
            )
            line_set.paint_uniform_color([0.0, 1.0, 0.0])
            vis.add_geometry(line_set, reset_bounding_box=False)

        # Camera setup (same as visualizer)
        ctr = vis.get_view_control()
        if is_tracking:
            look_center = [state.pose_se2[0], state.pose_se2[1], 0.0]
        elif len(pts) > 0:
            look_center = [np.mean(pts[:, 0]), np.mean(pts[:, 1]), 0.0]
        else:
            look_center = [0.0, 0.0, 0.0]
        ctr.set_lookat(look_center)
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_up([0.0, -1.0, 0.0])
        ctr.set_zoom(0.6)

        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(do_render=True)
        img_np = (np.asarray(img) * 255).astype(np.uint8)
        image_list.append(img_np)

    vis.destroy_window()

    try:
        import cv2
        height, width, _ = image_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, 10.0, (width, height))
        for img in image_list:
            out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"[SUCCESS] Video saved to {output_path}")
    except ImportError:
        print(f"[INFO] OpenCV not installed. Captured {len(image_list)} frames.")

if __name__ == "__main__":
    main()