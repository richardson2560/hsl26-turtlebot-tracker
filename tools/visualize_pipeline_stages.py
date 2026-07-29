"""
tools/visualize_pipeline_stages.py - Interactive stage‑by‑stage visualizer.
Keyboard:
  Left/Right arrows : previous/next frame
  1-7              : toggle layers (raw, ground, clusters, candidates, bbox, trajectory)
Usage: python tools/visualize_pipeline_stages.py --bag data/bags/.../ --config config/default_params.yaml
"""

import argparse
import sys
import yaml
import numpy as np
import open3d as o3d
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from turtlebot_tracker.io.mcap_loader import MCAPLiDARLoader
from turtlebot_tracker.core.preprocessor import LiDARPreprocessor
from turtlebot_tracker.core.segmentation import RangeImageSegmenter
from turtlebot_tracker.core.candidate_filter import CandidateFilter
from turtlebot_tracker.core.registration import DirectGMMRegistrator
from turtlebot_tracker.core.tracking import SE2ManifoldEKF

def create_wireframe_box(center, R_mat, extent=[0.45, 0.45, 0.48], color=[0.0, 1.0, 0.0]):
    dx, dy, dz = extent[0]/2.0, extent[1]/2.0, extent[2]/2.0
    corners_local = np.array([
        [-dx,-dy,-dz], [ dx,-dy,-dz], [ dx, dy,-dz], [-dx, dy,-dz],
        [-dx,-dy, dz], [ dx,-dy, dz], [ dx, dy, dz], [-dx, dy, dz]
    ])
    corners_world = (R_mat @ corners_local.T).T + center
    lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return line_set

class PipelineVisualizer:
    def __init__(self, bag_path, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.loader = MCAPLiDARLoader(bag_path)
        self.frames = list(self.loader.stream_point_clouds())
        self.current_idx = 0

        self.preproc = LiDARPreprocessor(self.config)
        self.segmenter = RangeImageSegmenter(self.config)
        self.filter_ = CandidateFilter(self.config)
        self.registrator = DirectGMMRegistrator(self.config)
        self.ekf = SE2ManifoldEKF(self.config)

        # Layer toggles
        self.show_raw = True
        self.show_ground = True
        self.show_clusters = True
        self.show_candidates = True
        self.show_bbox = True
        self.show_trajectory = True

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.register_key_callback(262, self.next_frame)
        self.vis.register_key_callback(263, self.prev_frame)
        self.vis.register_key_callback(49, self.toggle_raw)
        self.vis.register_key_callback(50, self.toggle_ground)
        self.vis.register_key_callback(51, self.toggle_clusters)
        self.vis.register_key_callback(52, self.toggle_candidates)
        self.vis.register_key_callback(53, lambda v: None)  # placeholder
        self.vis.register_key_callback(54, self.toggle_bbox)
        self.vis.register_key_callback(55, self.toggle_trajectory)

        self.vis.create_window(window_name="MOCD-Lite/SE(2) Stage Inspector", width=1440, height=810)
        self.render_frame()

    def render_frame(self):
        self.vis.clear_geometries()
        ts, pts, intensity = self.frames[self.current_idx]
        frame_data = self.preproc.process(ts, pts, intensity)
        clusters = self.segmenter.segment(frame_data)
        candidates = self.filter_.filter_candidates(clusters)
        state, best_cand = self.registrator.register_and_track(frame_data, candidates, self.ekf)

        first_geometry = True

        # 1. Raw points (dim gray)
        if self.show_raw and len(frame_data.raw_points) > 0:
            pcd_raw = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(frame_data.raw_points))
            pcd_raw.paint_uniform_color([0.3, 0.3, 0.3])
            self.vis.add_geometry(pcd_raw, reset_bounding_box=first_geometry)
            first_geometry = False

        # 2. Ground (transparent blue)
        if self.show_ground and frame_data.ground_points is not None and len(frame_data.ground_points) > 0:
            pcd_ground = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(frame_data.ground_points))
            pcd_ground.paint_uniform_color([0.1, 0.3, 0.6])
            self.vis.add_geometry(pcd_ground, reset_bounding_box=first_geometry)
            first_geometry = False

        # 3. Clusters & Candidates
        if self.show_clusters or self.show_candidates:
            for cand in clusters:
                # If showing clusters, color them by pass/fail. If showing only candidates, show green only.
                if self.show_clusters:
                    color = [1.0, 0.0, 0.0] if not cand.passed_filters else [0.0, 0.8, 0.0]
                else:
                    if not cand.passed_filters:
                        continue
                    color = [0.0, 1.0, 0.0]
                
                pcd_c = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cand.points))
                pcd_c.paint_uniform_color(color)
                self.vis.add_geometry(pcd_c, reset_bounding_box=first_geometry)
                first_geometry = False

        # 4. Bounding Box (SE(2) green wireframe)
        if state is not None and self.show_bbox and np.isfinite(state.pose_se2[0]):
            t = state.pose_se2[:3]
            yaw = state.pose_se2[2]
            R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw),  np.cos(yaw), 0],
                          [0, 0, 1]])
            bbox = create_wireframe_box(t, R, extent=[0.45, 0.45, 0.48], color=[0.0, 1.0, 0.0])
            self.vis.add_geometry(bbox, reset_bounding_box=first_geometry)
            first_geometry = False

        # 5. Trajectory history (Red dots)
        if self.show_trajectory and len(self.ekf.trajectory_log) > 0:
            traj_pts = np.array(self.ekf.trajectory_log)
            # Filter out invalid points to avoid scattering
            valid_mask = np.isfinite(traj_pts[:, 0]) & np.isfinite(traj_pts[:, 1])
            if np.sum(valid_mask) > 0:
                traj_pts = traj_pts[valid_mask]
                pcd_traj = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(traj_pts))
                pcd_traj.paint_uniform_color([1.0, 0.0, 0.0])
                self.vis.add_geometry(pcd_traj, reset_bounding_box=first_geometry)
                first_geometry = False

        # Coordinate frame at origin
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        self.vis.add_geometry(coord_frame, reset_bounding_box=first_geometry)

        self.vis.poll_events()
        self.vis.update_renderer()

        # --- FIX CAMERA: Top-down view looking in negative Z ---
        if not hasattr(self, '_camera_set'):
            ctr = self.vis.get_view_control()
            # Center the view on the current robot pose if available, otherwise on the raw points
            if state is not None and np.isfinite(state.pose_se2[0]):
                center = state.pose_se2[:3].copy()
                center[2] = 0.0
            elif len(pts) > 0:
                center = np.mean(pts, axis=0)
            else:
                center = [0.0, 0.0, 0.0]
            
            ctr.set_lookat(center)
            # Camera looks from +Z towards -Z (top-down)
            ctr.set_front([0.0, 0.0, -1.0])
            ctr.set_up([0.0, -1.0, 0.0])  # Y points down on screen, X right (bird's eye map)
            ctr.set_zoom(0.6)  # Not too high
            self._camera_set = True

    # --- Callbacks ---
    def next_frame(self, vis):
        if self.current_idx < len(self.frames) - 1:
            self.current_idx += 1
            self._camera_set = False  # Reset camera to recenter on new frame
            self.render_frame()

    def prev_frame(self, vis):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._camera_set = False
            self.render_frame()

    def toggle_raw(self, vis):    self.show_raw = not self.show_raw; self._camera_set = False; self.render_frame()
    def toggle_ground(self, vis): self.show_ground = not self.show_ground; self._camera_set = False; self.render_frame()
    def toggle_clusters(self, vis): self.show_clusters = not self.show_clusters; self._camera_set = False; self.render_frame()
    def toggle_candidates(self, vis): self.show_candidates = not self.show_candidates; self._camera_set = False; self.render_frame()
    def toggle_bbox(self, vis):   self.show_bbox = not self.show_bbox; self._camera_set = False; self.render_frame()
    def toggle_trajectory(self, vis): self.show_trajectory = not self.show_trajectory; self._camera_set = False; self.render_frame()

    def run(self):
        self.vis.run()
        self.vis.destroy_window()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=str, required=True, help="Path to bag directory or .mcap file")
    parser.add_argument("--config", type=str, default="config/default_params.yaml", help="Config YAML")
    args = parser.parse_args()

    bag_path = Path(args.bag)
    if bag_path.is_dir():
        mcap_files = list(bag_path.glob("*.mcap"))
        if not mcap_files:
            print(f"[ERROR] No .mcap found in {bag_path}")
            return
        bag_path = mcap_files[0]

    if not bag_path.exists():
        print(f"[ERROR] Bag not found: {bag_path}")
        return

    vis = PipelineVisualizer(str(bag_path), args.config)
    vis.run()

if __name__ == "__main__":
    main()