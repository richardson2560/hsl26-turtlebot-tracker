"""
tools/generate_evaluation_video.py - Evaluation Video Generator for Hackathon Submission
Usage: python tools/generate_evaluation_video.py --bag data/bags/rosbag2_2026_06_27-18_43-mov_01
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

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation demo video")
    parser.add_argument("--bag", type=str, required=True, help="Path to bag folder or .mcap file")
    parser.add_argument("--config", type=str, default="config/default_params.yaml", help="Path to config")
    parser.add_argument("--output", type=str, default="data/outputs/evaluation_demo.mp4", help="Output MP4 video path")
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
    vis.create_window(window_name="StarLine Hackathon 2026 - Evaluation Demo", width=1280, height=720, visible=False)

    print(f"[INFO] Generating evaluation video from: {args.bag}")
    image_list = []

    for frame_idx, (ts, pts, intensity) in enumerate(loader.stream_point_clouds()):
        frame_data = preprocessor.process(ts, pts, intensity)
        clusters = segmenter.segment(frame_data)
        candidates = candidate_filter.filter_candidates(clusters)
        state, target = registrator.register_and_track(frame_data, candidates, ekf)

        vis.clear_geometries()

        # Render raw obstacles
        if frame_data.obstacle_points is not None:
            pcd_obs = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(frame_data.obstacle_points))
            pcd_obs.paint_uniform_color([0.3, 0.3, 0.3])
            vis.add_geometry(pcd_obs, reset_bounding_box=False)

        # Render target points
        if target is not None:
            pcd_target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target.points))
            pcd_target.paint_uniform_color([1.0, 0.95, 0.0])
            vis.add_geometry(pcd_target, reset_bounding_box=False)

        # Render SE(2) Bounding Box
        if state is not None:
            x, y, yaw = state.pose_se2
            dx, dy, dz = 0.225, 0.225, 0.24
            corners = np.array([
                [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
                [-dx, -dy, dz],  [dx, -dy, dz],  [dx, dy, dz],  [-dx, dy, dz]
            ])
            R = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
            world_corners = (R @ corners.T).T + np.array([x, y, -0.05])
            lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
            
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(world_corners),
                lines=o3d.utility.Vector2iVector(lines)
            )
            line_set.paint_uniform_color([0.0, 1.0, 0.0])
            vis.add_geometry(line_set, reset_bounding_box=False)

        vis.poll_events()
        vis.update_renderer()
        
        # Capture frame buffer
        img = vis.capture_screen_float_buffer(do_render=True)
        img_np = (np.asarray(img) * 255).astype(np.uint8)
        image_list.append(img_np)

    vis.destroy_window()

    # Save to MP4 using OpenCV or imageio if installed, fallback to image frame count print
    try:
        import cv2
        height, width, _ = image_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, 10.0, (width, height))
        for img in image_list:
            out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"[SUCCESS] Demo video successfully generated at: {output_path}")
    except ImportError:
        print(f"[INFO] OpenCV not installed. Captured {len(image_list)} frames for video generation.")

if __name__ == "__main__":
    main()