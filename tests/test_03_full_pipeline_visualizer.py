"""
test_03_full_pipeline_visualizer.py - Interactive Real-time 3D Visualizer for Turtlebot2 Detection & Tracking.
"""

import sys
import glob
import time
from pathlib import Path
import open3d as o3d
import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "src"))
from turtlebot_tracker.dataloader import MCAPLiDARLoader
from turtlebot_tracker.preprocessor import PointCloudPreprocessor
from turtlebot_tracker.clustering import ClusterGaussianFitter
from turtlebot_tracker.optimal_transport import OptimalTransportMatcher
from turtlebot_tracker.pose_estimator import RigidPoseEstimator
from turtlebot_tracker.particle_filter import SDEParticleFilter

def load_canonical_model() -> list:
    """Canonical structural Gaussian components for Turtlebot2."""
    return [
        {'mu': np.array([0.0, 0.0, 0.05]), 'scales': np.array([0.18, 0.18, 0.02]), 'weight': 0.3},
        {'mu': np.array([0.0, 0.0, 0.22]), 'scales': np.array([0.16, 0.16, 0.02]), 'weight': 0.3},
        {'mu': np.array([0.0, 0.0, 0.42]), 'scales': np.array([0.15, 0.15, 0.03]), 'weight': 0.3},
        {'mu': np.array([0.0, 0.0, 0.25]), 'scales': np.array([0.05, 0.05, 0.20]), 'weight': 0.1}
    ]

def create_bounding_box_mesh(t: np.ndarray, R_mat: np.ndarray, size=[0.45, 0.45, 0.5]) -> o3d.geometry.OrientedBoundingBox:
    """Creates an oriented bounding box for estimated pose."""
    obb = o3d.geometry.OrientedBoundingBox(center=t, R=R_mat, extent=size)
    obb.color = (0.0, 1.0, 0.0)  # Green box
    return obb

def main():
    bag_files = sorted(glob.glob("data/bags/*/*.mcap"))
    if not bag_files:
        print("[ERROR] No .mcap files found in data/bags/*/")
        return

    bag_path = bag_files[0]
    print(f"[TEST 03] Full Tracking Visualization on: {Path(bag_path).name}")

    loader = MCAPLiDARLoader(bag_path)
    preprocessor = PointCloudPreprocessor(voxel_size=0.03)
    fitter = ClusterGaussianFitter(eps=0.22, min_points=15)
    ot_matcher = OptimalTransportMatcher(gamma_prune=0.3, gamma_spawn=0.3)
    pf = SDEParticleFilter(num_particles=120)
    canonical_model = load_canonical_model()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="HSL26 - Turtlebot2 Real-time 3D Tracking", width=1280, height=720)

    pcd_obs = o3d.geometry.PointCloud()
    pcd_ground = o3d.geometry.PointCloud()
    pcd_target = o3d.geometry.PointCloud()
    trajectory_pcd = o3d.geometry.PointCloud()

    vis.add_geometry(pcd_obs)
    vis.add_geometry(pcd_ground)
    vis.add_geometry(pcd_target)
    vis.add_geometry(trajectory_pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    trajectory_history = []
    is_initialized = False
    last_timestamp = None
    current_obb = None

    for frame_idx, (ts, pts) in enumerate(loader.stream_point_clouds()):
        dt = 0.1 if last_timestamp is None else max(0.01, ts - last_timestamp)
        last_timestamp = ts

        obstacles, ground = preprocessor.process(pts)
        candidate_clusters = fitter.extract_clusters_and_fit_gaussians(obstacles)

        # Update Background Clouds
        pcd_ground.points = o3d.utility.Vector3dVector(ground)
        pcd_ground.paint_uniform_color([0.2, 0.2, 0.3])

        pcd_obs.points = o3d.utility.Vector3dVector(obstacles)
        pcd_obs.paint_uniform_color([0.5, 0.5, 0.5])

        if candidate_clusters:
            best_cost = float('inf')
            best_cluster = None
            best_P = None

            for cluster in candidate_clusters:
                cost, P_mat = ot_matcher.match_models(canonical_model, cluster['gaussians'])
                if cost < best_cost:
                    best_cost = cost
                    best_cluster = cluster
                    best_P = P_mat

            if best_cluster is not None:
                R_est, t_est = RigidPoseEstimator.estimate_pose(canonical_model, best_cluster['gaussians'], best_P)

                if not is_initialized:
                    pf.initialize(t_est, R_est)
                    is_initialized = True
                else:
                    pf.predict(dt)
                    pf.update(t_est, R_est, best_cost)

                smooth_t, smooth_R = pf.get_estimated_state()

                # Highlight Target Cluster (Yellow)
                pcd_target.points = o3d.utility.Vector3dVector(best_cluster['cluster_pts'])
                pcd_target.paint_uniform_color([1.0, 0.8, 0.0])

                # Update Trajectory History (Red)
                trajectory_history.append(smooth_t.copy())
                trajectory_pcd.points = o3d.utility.Vector3dVector(np.array(trajectory_history))
                trajectory_pcd.paint_uniform_color([1.0, 0.0, 0.0])

                # Update Bounding Box
                if current_obb is not None:
                    vis.remove_geometry(current_obb, reset_bounding_box=False)
                current_obb = create_bounding_box_mesh(smooth_t, smooth_R)
                vis.add_geometry(current_obb, reset_bounding_box=False)

        vis.update_geometry(pcd_obs)
        vis.update_geometry(pcd_ground)
        vis.update_geometry(pcd_target)
        vis.update_geometry(trajectory_pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.03)

    vis.destroy_window()
    print("[SUCCESS] Test 03 completed cleanly.")

if __name__ == "__main__":
    main()