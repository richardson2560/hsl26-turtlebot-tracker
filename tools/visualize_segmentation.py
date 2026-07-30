"""
visualize_segmentation.py - Phase 1: Segmentation and Semantic Labeling Validation.
Uses persistent geometries to prevent blank screen issues in Open3D 0.17+.
View: Bottom-up (Z-negative looking towards Z-positive).
"""

import argparse
import sys
import time
import yaml
import numpy as np
import open3d as o3d
from pathlib import Path

# Add project source to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from turtlebot_tracker.core.candidate_filter import CandidateFilter
from turtlebot_tracker.core.preprocessor import LiDARPreprocessor
from turtlebot_tracker.core.registration import DirectGMMRegistrator
from turtlebot_tracker.core.segmentation import RangeImageSegmenter
from turtlebot_tracker.core.tracking import SE2ManifoldEKF
from turtlebot_tracker.datatypes import LifecycleState, SemanticLabel
from turtlebot_tracker.io.mcap_loader import MCAPLiDARLoader

class SegmentationVisualizer:
    def __init__(self, bag_path: str, config_path: str, fps: float = 5.0):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Data Loading
        self.loader = MCAPLiDARLoader(bag_path)
        self.frames = list(self.loader.stream_point_clouds())
        if not self.frames:
            raise RuntimeError("No frames loaded from the source file.")

        # Playback State
        self.current_idx = 0
        self.paused = False
        self.fps = fps
        self.frame_delay = 1.0 / fps
        self.point_size = 5.0
        self.running = True
        self._first_render = True

        # Tracker Components
        self.preproc = LiDARPreprocessor(self.config)
        self.segmenter = RangeImageSegmenter(self.config)
        self.filter_ = CandidateFilter(self.config)
        self.registrator = DirectGMMRegistrator(self.config)
        self.ekf = SE2ManifoldEKF(self.config)

        # Open3D Setup
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="LiDAR Semantic Segmentation - Bottom View", width=1280, height=720)
        
        # Initialize Persistent Geometries (prevents flickering/blank screen)
        self.pcd_ground = o3d.geometry.PointCloud()
        self.pcd_walls = o3d.geometry.PointCloud()
        self.pcd_candidates = o3d.geometry.PointCloud()
        self.pcd_target = o3d.geometry.PointCloud()
        self.bbox_tracker = o3d.geometry.LineSet()
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

        # Add all geometries once
        self.vis.add_geometry(self.pcd_ground)
        self.vis.add_geometry(self.pcd_walls)
        self.vis.add_geometry(self.pcd_candidates)
        self.vis.add_geometry(self.pcd_target)
        self.vis.add_geometry(self.bbox_tracker)
        self.vis.add_geometry(self.coord_frame)

        # Visual Options
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.08, 0.08, 0.08])
        opt.point_size = self.point_size
        opt.light_on = False

        # Register Controls
        self.vis.register_key_callback(32, self.toggle_pause)      # Space
        self.vis.register_key_callback(262, self.next_frame_key)   # Right Arrow
        self.vis.register_key_callback(263, self.prev_frame_key)   # Left Arrow
        self.vis.register_key_callback(82, self.reset)             # 'R'
        self.vis.register_key_callback(81, self.quit)              # 'Q'
        self.vis.register_key_callback(61, self.inc_point_size)    # '+' or '='
        self.vis.register_key_callback(45, self.dec_point_size)    # '-'

    def toggle_pause(self, vis):
        self.paused = not self.paused
        print(f"\n[INFO] Paused: {self.paused}")

    def quit(self, vis):
        self.running = False
        self.vis.destroy_window()

    def reset(self, vis):
        self.current_idx = 0
        self.ekf = SE2ManifoldEKF(self.config)
        print("\n[INFO] Resetting sequence...")

    def inc_point_size(self, vis):
        self.point_size = min(self.point_size + 1.0, 20.0)
        self.vis.get_render_option().point_size = self.point_size

    def dec_point_size(self, vis):
        self.point_size = max(self.point_size - 1.0, 1.0)
        self.vis.get_render_option().point_size = self.point_size

    def next_frame_key(self, vis):
        self.current_idx = (self.current_idx + 1) % len(self.frames)
        self.paused = True
        self.render_frame()

    def prev_frame_key(self, vis):
        self.current_idx = (self.current_idx - 1) % len(self.frames)
        self.paused = True
        self.render_frame()

    def set_camera_view(self):
        """Sets the camera to look from Z-negative towards Z-positive (Bottom View)."""
        ctr = self.vis.get_view_control()
        ctr.set_lookat([0.0, 0.0, 0.0])
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_up([0.0, 1.0, 0.0])
        ctr.set_zoom(8.0)

    def render_frame(self):
        # 1. Pipeline processing
        ts, pts, intensity = self.frames[self.current_idx]
        frame_data = self.preproc.process(ts, pts, intensity)
        clusters = self.segmenter.segment(frame_data)
        candidates = self.filter_.filter_candidates(clusters)
        state, best_cand = self.registrator.register_and_track(frame_data, candidates, self.ekf)

        # 2. Update Ground
        if frame_data.ground_points is not None and len(frame_data.ground_points) > 0:
            self.pcd_ground.points = o3d.utility.Vector3dVector(frame_data.ground_points)
            self.pcd_ground.paint_uniform_color([0.1, 0.3, 0.6])
        else:
            self.pcd_ground.points = o3d.utility.Vector3dVector([])

        # 3. Update Walls
        if frame_data.semantic_labels is not None:
            wall_mask = frame_data.semantic_labels == SemanticLabel.STRUCTURE_WALL
            if np.any(wall_mask):
                self.pcd_walls.points = o3d.utility.Vector3dVector(frame_data.obstacle_points[wall_mask])
                self.pcd_walls.paint_uniform_color([0.4, 0.4, 0.4])
            else:
                self.pcd_walls.points = o3d.utility.Vector3dVector([])

        # 4. Update Candidates (rejected = red, valid = cyan)
        all_pts, all_colors = [], []
        for cand in candidates:
            if best_cand and cand.id == best_cand.id: continue
            all_pts.extend(cand.points)
            color = [0.8, 0.2, 0.2] if not cand.passed_filters else [0.0, 0.7, 0.7]
            all_colors.extend([color] * len(cand.points))
        
        self.pcd_candidates.points = o3d.utility.Vector3dVector(all_pts)
        self.pcd_candidates.colors = o3d.utility.Vector3dVector(all_colors)

        # 5. Update Target (Yellow)
        if best_cand:
            self.pcd_target.points = o3d.utility.Vector3dVector(best_cand.points)
            self.pcd_target.paint_uniform_color([1.0, 0.9, 0.0])
        else:
            self.pcd_target.points = o3d.utility.Vector3dVector([])

        # 6. Update Tracker Bounding Box
        if state.lifecycle_state in (LifecycleState.ACTIVE_TRACKING, LifecycleState.COASTING_LOST):
            center = np.array([state.pose_se2[0], state.pose_se2[1], state.z])
            yaw = state.pose_se2[2]
            dx, dy, dz = 0.25, 0.25, 0.25 # Semi-extents
            R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            
            local_corners = np.array([[-dx,-dy,-dz],[dx,-dy,-dz],[dx,dy,-dz],[-dx,dy,-dz],
                                     [-dx,-dy,dz],[dx,-dy,dz],[dx,dy,dz],[-dx,dy,dz]])
            world_corners = (R @ local_corners.T).T + center
            lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
            
            self.bbox_tracker.points = o3d.utility.Vector3dVector(world_corners)
            self.bbox_tracker.lines = o3d.utility.Vector2iVector(lines)
            self.bbox_tracker.paint_uniform_color([0, 1, 0])
        else:
            self.bbox_tracker.points = o3d.utility.Vector3dVector([])

        # Trigger Open3D Updates
        self.vis.update_geometry(self.pcd_ground)
        self.vis.update_geometry(self.pcd_walls)
        self.vis.update_geometry(self.pcd_candidates)
        self.vis.update_geometry(self.pcd_target)
        self.vis.update_geometry(self.bbox_tracker)

        # Apply camera lock on first render
        if self._first_render:
            self.set_camera_view()
            self._first_render = False

        print(f"\rFrame {self.current_idx:03d}/{len(self.frames)-1} | "
              f"State: {state.lifecycle_state.name:15s} | "
              f"NIS: {state.nis:.2f} | FPS Target: {self.fps}", end="")

    def run(self):
        last_time = time.time()
        while self.running:
            if not self.paused:
                self.render_frame()
                self.current_idx = (self.current_idx + 1) % len(self.frames)
                if self.current_idx == 0: 
                    self.ekf = SE2ManifoldEKF(self.config) # Restart EKF on loop
                
                # Precise FPS Control
                elapsed = time.time() - last_time
                time.sleep(max(0, self.frame_delay - elapsed))
                last_time = time.time()
            else:
                # Keep window responsive when paused
                self.vis.poll_events()
                self.vis.update_renderer()
                time.sleep(0.05)

            if not self.vis.poll_events(): # Handle window close button
                break

        self.vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="LiDAR Phase 1 Visualization")
    parser.add_argument("--bag", type=str, required=True, help="Path to bag directory or mcap file")
    parser.add_argument("--config", type=str, default="config/default_params.yaml")
    parser.add_argument("--fps", type=float, default=7.0)
    args = parser.parse_args()

    # Find file
    p = Path(args.bag)
    mcap_file = list(p.glob("*.mcap"))[0] if p.is_dir() else p
    
    try:
        app = SegmentationVisualizer(str(mcap_file), args.config, args.fps)
        app.run()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

if __name__ == "__main__":
    main()