"""
visualize_pipeline_stages.py - Open3D Interactive Visualizer with 8-Layer Toggles.
Fixed camera orientation: Z-up, top-down view.
"""

import argparse
from pathlib import Path
import sys
import yaml
import numpy as np
import open3d as o3d

sys.path.append(str(Path(__file__).parent.parent / "src"))

from turtlebot_tracker.core.candidate_filter import CandidateFilter
from turtlebot_tracker.core.preprocessor import LiDARPreprocessor
from turtlebot_tracker.core.registration import DirectGMMRegistrator
from turtlebot_tracker.core.segmentation import RangeImageSegmenter
from turtlebot_tracker.core.tracking import SE2ManifoldEKF
from turtlebot_tracker.datatypes import LifecycleState, SemanticLabel
from turtlebot_tracker.io.mcap_loader import MCAPLiDARLoader


def create_wireframe_box(center: np.ndarray, R_mat: np.ndarray,
                         extent=[0.45, 0.45, 0.48], color=[0.0, 1.0, 0.0]) -> o3d.geometry.LineSet:
    dx, dy, dz = extent[0] / 2.0, extent[1] / 2.0, extent[2] / 2.0
    corners_local = np.array([
        [-dx, -dy, -dz], [ dx, -dy, -dz], [ dx,  dy, -dz], [-dx,  dy, -dz],
        [-dx, -dy,  dz], [ dx, -dy,  dz], [ dx,  dy,  dz], [-dx,  dy,  dz]
    ], dtype=np.float64)
    corners_world = (R_mat @ corners_local.T).T + center
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return line_set


class PipelineVisualizer:
    def __init__(self, bag_path: str, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.loader = MCAPLiDARLoader(bag_path)
        self.frames = list(self.loader.stream_point_clouds())
        if not self.frames:
            raise RuntimeError("No frames loaded from bag. Check path and content.")
        self.current_idx = 0

        self.preproc = LiDARPreprocessor(self.config)
        self.segmenter = RangeImageSegmenter(self.config)
        self.filter_ = CandidateFilter(self.config)
        self.registrator = DirectGMMRegistrator(self.config)
        self.ekf = SE2ManifoldEKF(self.config)

        self.show_raw = True
        self.show_ground = True
        self.show_walls = True
        self.show_rejected = True
        self.show_candidates = True
        self.show_target = True
        self.show_bbox = True
        self.show_trajectory = True

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.register_key_callback(262, self.next_frame)
        self.vis.register_key_callback(263, self.prev_frame)
        self.vis.register_key_callback(49, self.toggle_raw)
        self.vis.register_key_callback(50, self.toggle_ground)
        self.vis.register_key_callback(51, self.toggle_walls)
        self.vis.register_key_callback(52, self.toggle_rejected)
        self.vis.register_key_callback(53, self.toggle_candidates)
        self.vis.register_key_callback(54, self.toggle_target)
        self.vis.register_key_callback(55, self.toggle_bbox)
        self.vis.register_key_callback(56, self.toggle_trajectory)

        self.vis.create_window(window_name="turtlebot_tracker - 8 Layer Inspector", width=1440, height=810)
        self._camera_initialized = False
        self.render_frame()

    def render_frame(self) -> None:
        self.vis.clear_geometries()
        ts, pts, intensity = self.frames[self.current_idx]

        frame_data = self.preproc.process(ts, pts, intensity)
        clusters = self.segmenter.segment(frame_data)
        candidates = self.filter_.filter_candidates(clusters)
        state, best_cand = self.registrator.register_and_track(frame_data, candidates, self.ekf)

        # Always add a coordinate frame for reference
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.vis.add_geometry(coord, reset_bounding_box=False)

        # Add a ground plane grid for orientation (if no points)
        if len(frame_data.raw_points) == 0:
            grid = o3d.geometry.LineSet.create_from_points(
                points=o3d.utility.Vector3dVector([
                    [-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]
                ]),
                lines=o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0]])
            )
            grid.paint_uniform_color([0.5, 0.5, 0.5])
            self.vis.add_geometry(grid, reset_bounding_box=False)

        # 1. Raw cloud
        if self.show_raw and len(frame_data.raw_points) > 0:
            pcd_raw = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(frame_data.raw_points))
            pcd_raw.paint_uniform_color([0.25, 0.25, 0.25])
            self.vis.add_geometry(pcd_raw, reset_bounding_box=False)

        # 2. Ground
        if self.show_ground and frame_data.ground_points is not None and len(frame_data.ground_points) > 0:
            pcd_g = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(frame_data.ground_points))
            pcd_g.paint_uniform_color([0.1, 0.25, 0.5])
            self.vis.add_geometry(pcd_g, reset_bounding_box=False)

        # 3. Walls
        if self.show_walls and frame_data.semantic_labels is not None:
            wall_mask = frame_data.semantic_labels == SemanticLabel.STRUCTURE_WALL
            if np.any(wall_mask):
                pcd_w = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(frame_data.obstacle_points[wall_mask]))
                pcd_w.paint_uniform_color([0.4, 0.4, 0.45])
                self.vis.add_geometry(pcd_w, reset_bounding_box=False)

        # 4-5. Candidates
        for cand in candidates:
            if not cand.passed_filters:
                if self.show_rejected:
                    pcd_r = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cand.points))
                    pcd_r.paint_uniform_color([1.0, 0.0, 0.0])
                    self.vis.add_geometry(pcd_r, reset_bounding_box=False)
            else:
                if self.show_candidates and (best_cand is None or cand.id != best_cand.id):
                    pcd_c = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cand.points))
                    pcd_c.paint_uniform_color([0.0, 0.8, 1.0])
                    self.vis.add_geometry(pcd_c, reset_bounding_box=False)

        # 6. Target
        if self.show_target and best_cand is not None:
            pcd_t = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(best_cand.points))
            pcd_t.paint_uniform_color([1.0, 0.95, 0.0])
            self.vis.add_geometry(pcd_t, reset_bounding_box=False)

        # 7. BBox
        is_tracking = state.lifecycle_state in (LifecycleState.ACTIVE_TRACKING, LifecycleState.COASTING_LOST)
        if self.show_bbox and is_tracking:
            center = np.array([state.pose_se2[0], state.pose_se2[1], state.z], dtype=np.float64)
            yaw = state.pose_se2[2]
            R_mat = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                              [np.sin(yaw),  np.cos(yaw), 0.0],
                              [0.0,          0.0,         1.0]], dtype=np.float64)
            bbox = create_wireframe_box(center, R_mat, extent=[0.45, 0.45, 0.48], color=[0.0, 1.0, 0.0])
            self.vis.add_geometry(bbox, reset_bounding_box=False)

        # 8. Trajectory
        if self.show_trajectory and len(self.ekf.trajectory_log) > 0:
            traj_xy = np.array([p[:2] for p in self.ekf.trajectory_log])
            traj_z = np.array(self.ekf.z_log)
            if len(traj_z) < len(traj_xy):
                if len(traj_z) > 0:
                    traj_z = np.pad(traj_z, (0, len(traj_xy) - len(traj_z)), constant_values=traj_z[-1])
                else:
                    traj_z = np.zeros(len(traj_xy))
            traj_pts = np.column_stack([traj_xy, traj_z])
            pcd_tr = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(traj_pts))
            pcd_tr.paint_uniform_color([1.0, 0.0, 0.0])
            self.vis.add_geometry(pcd_tr, reset_bounding_box=False)

        self.vis.poll_events()
        self.vis.update_renderer()

        # --- Camera setup (Z-up, top-down) ---
        ctr = self.vis.get_view_control()
        if is_tracking:
            look_center = [state.pose_se2[0], state.pose_se2[1], 0.0]
        elif len(pts) > 0:
            look_center = [np.mean(pts[:, 0]), np.mean(pts[:, 1]), 0.0]
        else:
            look_center = [0.0, 0.0, 0.0]

        # Set camera to look from +Z (above) towards -Z (down)
        # Z is up, so front = [0, 0, -1] looks downwards
        # Up vector should be Y (or X) to keep the horizon level
        ctr.set_lookat(look_center)
        ctr.set_front([0.0, 0.0, -1.0])   # looking down
        ctr.set_up([0.0, -1.0, 0.0])      # Y is up on screen (inverted to match Open3D convention)
        ctr.set_zoom(0.6)

    def next_frame(self, vis):
        if self.current_idx < len(self.frames) - 1:
            self.current_idx += 1
            self.render_frame()

    def prev_frame(self, vis):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.render_frame()

    def toggle_raw(self, vis):        self.show_raw = not self.show_raw; self.render_frame()
    def toggle_ground(self, vis):     self.show_ground = not self.show_ground; self.render_frame()
    def toggle_walls(self, vis):      self.show_walls = not self.show_walls; self.render_frame()
    def toggle_rejected(self, vis):   self.show_rejected = not self.show_rejected; self.render_frame()
    def toggle_candidates(self, vis): self.show_candidates = not self.show_candidates; self.render_frame()
    def toggle_target(self, vis):     self.show_target = not self.show_target; self.render_frame()
    def toggle_bbox(self, vis):       self.show_bbox = not self.show_bbox; self.render_frame()
    def toggle_trajectory(self, vis): self.show_trajectory = not self.show_trajectory; self.render_frame()

    def run(self):
        self.vis.run()
        self.vis.destroy_window()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=str, required=True, help="Path to bag directory or .mcap file")
    parser.add_argument("--config", type=str, default="config/default_params.yaml")
    args = parser.parse_args()

    bag_path = Path(args.bag)
    if bag_path.is_dir():
        mcaps = list(bag_path.glob("*.mcap"))
        if not mcaps:
            print(f"[ERROR] No .mcap files in {bag_path}")
            return
        bag_path = mcaps[0]

    try:
        app = PipelineVisualizer(str(bag_path), args.config)
        app.run()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()