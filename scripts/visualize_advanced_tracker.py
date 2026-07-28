"""
visualize_advanced_tracker.py - Unclipped Overhead Global View with World Boundary Constraints.
"""

import sys
import glob
import json
import time
from pathlib import Path
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append(str(Path(__file__).parent.parent / "src"))
from turtlebot_tracker.dataloader import MCAPLiDARLoader
from turtlebot_tracker.preprocessor import PointCloudPreprocessor
from turtlebot_tracker.clustering import ClusterGaussianFitter
from turtlebot_tracker.optimal_transport import OptimalTransportMatcher
from turtlebot_tracker.pose_estimator import RigidPoseEstimator
from turtlebot_tracker.particle_filter import SDEParticleFilter

def load_canonical_model() -> list:
    canon_file = Path("config/canonical_turtlebot2.json")
    if canon_file.exists():
        with open(canon_file, "r") as f:
            return json.load(f)['canonical_gaussians']
    return [
        {'mu': [0.0, 0.0, -0.18], 'scales': [0.18, 0.18, 0.02], 'weight': 0.35},
        {'mu': [0.0, 0.0,  0.00], 'scales': [0.16, 0.16, 0.02], 'weight': 0.35},
        {'mu': [0.0, 0.0,  0.20], 'scales': [0.15, 0.15, 0.03], 'weight': 0.30}
    ]

def enforce_planar_se2_rotation(R_mat: np.ndarray) -> np.ndarray:
    yaw = np.arctan2(R_mat[1, 0], R_mat[0, 0])
    return R.from_euler('z', yaw).as_matrix()

def create_wireframe_box(center: np.ndarray, R_mat: np.ndarray, extent=[0.45, 0.45, 0.48], color=[0.0, 1.0, 0.0]) -> o3d.geometry.LineSet:
    dx, dy, dz = extent[0] / 2.0, extent[1] / 2.0, extent[2] / 2.0
    corners_local = np.array([
        [-dx, -dy, -dz], [ dx, -dy, -dz], [ dx,  dy, -dz], [-dx,  dy, -dz],
        [-dx, -dy,  dz], [ dx, -dy,  dz], [ dx,  dy,  dz], [-dx,  dy,  dz]
    ])
    corners_world = (R_mat @ corners_local.T).T + center
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return line_set

def create_large_world_boundary() -> o3d.geometry.LineSet:
    """Creates a large transparent world bounding box (8m x 8m x 4m) to prevent Open3D clipping planes."""
    min_bound = np.array([-2.0, -4.0, -1.0])
    max_bound = np.array([ 6.0,  4.0,  3.0])
    box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
    line_set.colors = o3d.utility.Vector3dVector([[0.05, 0.05, 0.05] for _ in range(12)]) # Nearly invisible dark gray
    return line_set

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_idx", type=int, default=5, help="Index of bag file (0: static_1m, 5: mov_01)")
    args = parser.parse_args()

    bag_files = sorted(glob.glob("data/bags/*/*.mcap"))
    if not bag_files:
        print("[ERROR] No .mcap files found in data/bags/*/")
        return

    bag_path = bag_files[min(args.bag_idx, len(bag_files) - 1)]
    print(f"\n[UNCLIPPED OVERHEAD VISUALIZER] Sequence: {Path(bag_path).parent.name}")

    loader = MCAPLiDARLoader(bag_path)
    preprocessor = PointCloudPreprocessor(voxel_size=0.03)
    fitter = ClusterGaussianFitter(eps=0.22, min_points=20)
    ot_matcher = OptimalTransportMatcher()
    pf = SDEParticleFilter(num_particles=120)
    canonical_model = load_canonical_model()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"HSL26 Overhead Unclipped View - {Path(bag_path).parent.name}", width=1280, height=720)

    # 1. Add Large World Boundary Box FIRST to force Open3D's near/far frustum to stay wide open
    world_bounds = create_large_world_boundary()
    vis.add_geometry(world_bounds)

    # Pre-allocated dynamic geometries
    pcd_ground = o3d.geometry.PointCloud()
    pcd_obstacles = o3d.geometry.PointCloud()
    pcd_target = o3d.geometry.PointCloud()
    pcd_trajectory = o3d.geometry.PointCloud()
    target_box_lineset = o3d.geometry.LineSet()

    vis.add_geometry(pcd_ground)
    vis.add_geometry(pcd_obstacles)
    vis.add_geometry(pcd_target)
    vis.add_geometry(pcd_trajectory)
    vis.add_geometry(target_box_lineset)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6))

    trajectory_history = []
    is_initialized = False
    last_timestamp = None
    camera_fixed = False

    cluster_colors = [
        [0.0, 0.85, 0.95], [0.95, 0.0, 0.95], [0.2, 0.95, 0.2],
        [1.0, 0.5, 0.0],  [0.4, 0.4, 1.0],  [0.9, 0.9, 0.0]
    ]

    for frame_idx, (ts, pts) in enumerate(loader.stream_point_clouds()):
        dt = 0.1 if last_timestamp is None else max(0.01, ts - last_timestamp)
        last_timestamp = ts

        obstacles, ground = preprocessor.process(pts)
        if len(obstacles) < 15:
            continue

        candidate_clusters = fitter.extract_clusters_and_fit_gaussians(obstacles)

        # Ground in dark subtle charcoal
        if len(ground) > 0:
            pcd_ground.points = o3d.utility.Vector3dVector(ground)
            pcd_ground.paint_uniform_color([0.10, 0.10, 0.12])
            vis.update_geometry(pcd_ground)

        if candidate_clusters:
            all_obs_pts = []
            all_obs_colors = []

            best_cost = float('inf')
            best_cluster = None
            best_P = None

            for c_idx, cluster in enumerate(candidate_clusters):
                cost, P_mat = ot_matcher.match_models(canonical_model, cluster['gaussians'])
                
                c_pts = cluster['cluster_pts']
                c_col = np.tile(cluster_colors[c_idx % len(cluster_colors)], (len(c_pts), 1))
                all_obs_pts.append(c_pts)
                all_obs_colors.append(c_col)

                if cost < best_cost:
                    best_cost = cost
                    best_cluster = cluster
                    best_P = P_mat

            if all_obs_pts:
                pcd_obstacles.points = o3d.utility.Vector3dVector(np.vstack(all_obs_pts))
                pcd_obstacles.colors = o3d.utility.Vector3dVector(np.vstack(all_obs_colors))
                vis.update_geometry(pcd_obstacles)

            if best_cluster is not None and best_cost < 0.15:
                R_raw, t_est = RigidPoseEstimator.estimate_pose(canonical_model, best_cluster['gaussians'], best_P)
                R_planar = enforce_planar_se2_rotation(R_raw)

                if not is_initialized:
                    pf.initialize(t_est, R_planar)
                    is_initialized = True
                else:
                    pf.predict(dt)
                    pf.update(t_est, R_planar, best_cost)

                smooth_t, smooth_R = pf.get_estimated_state()

                # Highlight Turtlebot2 in Bright Neon Yellow
                pcd_target.points = o3d.utility.Vector3dVector(best_cluster['cluster_pts'])
                pcd_target.paint_uniform_color([1.0, 0.95, 0.0])
                vis.update_geometry(pcd_target)

                # Red Trajectory Line
                trajectory_history.append(smooth_t.copy())
                pcd_trajectory.points = o3d.utility.Vector3dVector(np.array(trajectory_history))
                pcd_trajectory.paint_uniform_color([1.0, 0.0, 0.0])
                vis.update_geometry(pcd_trajectory)

                # Update Target Green Wireframe Box
                updated_box = create_wireframe_box(smooth_t, smooth_R, color=[0.0, 1.0, 0.0])
                target_box_lineset.points = updated_box.points
                target_box_lineset.lines = updated_box.lines
                target_box_lineset.colors = updated_box.colors
                vis.update_geometry(target_box_lineset)

        vis.poll_events()
        vis.update_renderer()

        # CONFIGURE OVERHEAD UNCLIPPED CAMERA ONCE
        if not camera_fixed:
            for _ in range(5):
                vis.poll_events()
                vis.update_renderer()

            ctr = vis.get_view_control()
            # 1. Target center of room
            ctr.set_lookat([1.5, 0.5, -0.2])
            # 2. Camera vector looking down from ABOVE (+Z positive!)
            ctr.set_front([-0.1, 0.0, -0.98])
            # 3. +X points UP on screen
            ctr.set_up([1.0, 0.0, 0.0])
            # 4. Comfortable wide zoom
            ctr.set_zoom(0.35)
            camera_fixed = True

        if frame_idx % 25 == 0 and is_initialized:
            print(f"Frame {frame_idx:04d} | Target Pose: X={smooth_t[0]:.2f}m, Y={smooth_t[1]:.2f}m | Cost: {best_cost:.4f}")

    vis.destroy_window()
    print("[SUCCESS] Overhead tracking visualization completed.")

if __name__ == "__main__":
    main()