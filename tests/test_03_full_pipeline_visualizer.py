"""
test_03_full_pipeline_visualizer.py - Robust Real-Time 3D Tracking Visualizer
"""

import sys
import glob
import time
import json
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
    canon_file = Path("config/canonical_turtlebot2.json")
    if canon_file.exists():
        with open(canon_file, "r") as f:
            data = json.load(f)
            return data['canonical_gaussians']
    
    # Default local zero-centered fallback
    return [
        {'mu': [0.0, 0.0, -0.18], 'scales': [0.18, 0.18, 0.02], 'weight': 0.35},
        {'mu': [0.0, 0.0,  0.00], 'scales': [0.16, 0.16, 0.02], 'weight': 0.35},
        {'mu': [0.0, 0.0,  0.20], 'scales': [0.15, 0.15, 0.03], 'weight': 0.30}
    ]

def main():
    bag_files = sorted(glob.glob("data/bags/*/*.mcap"))
    if not bag_files:
        print("[ERROR] No .mcap files found in data/bags/*/")
        return

    loader = MCAPLiDARLoader(bag_files[0])
    preprocessor = PointCloudPreprocessor(voxel_size=0.03)
    fitter = ClusterGaussianFitter(eps=0.22, min_points=20)
    ot_matcher = OptimalTransportMatcher()
    pf = SDEParticleFilter(num_particles=100)
    canonical_model = load_canonical_model()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="HSL26 - Turtlebot2 Robust Real-time Tracking", width=1280, height=720)

    pcd_obs = o3d.geometry.PointCloud()
    pcd_ground = o3d.geometry.PointCloud()
    pcd_target = o3d.geometry.PointCloud()
    trajectory_pcd = o3d.geometry.PointCloud()

    vis.add_geometry(pcd_obs)
    vis.add_geometry(pcd_ground)
    vis.add_geometry(pcd_target)
    vis.add_geometry(trajectory_pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4))

    trajectory_history = []
    is_initialized = False
    last_timestamp = None
    current_obb = None

    for frame_idx, (ts, pts) in enumerate(loader.stream_point_clouds()):
        dt = 0.1 if last_timestamp is None else max(0.01, ts - last_timestamp)
        last_timestamp = ts

        obstacles, ground = preprocessor.process(pts)
        if len(obstacles) < 15:
            continue

        candidate_clusters = fitter.extract_clusters_and_fit_gaussians(obstacles)

        pcd_ground.points = o3d.utility.Vector3dVector(ground if len(ground) > 0 else np.zeros((1, 3)))
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

            if best_cluster is not None and best_cost < 0.8:
                R_est, t_est = RigidPoseEstimator.estimate_pose(canonical_model, best_cluster['gaussians'], best_P)

                if not is_initialized:
                    pf.initialize(t_est, R_est)
                    is_initialized = True
                else:
                    pf.predict(dt)
                    pf.update(t_est, R_est, best_cost)

                smooth_t, smooth_R = pf.get_estimated_state()

                pcd_target.points = o3d.utility.Vector3dVector(best_cluster['cluster_pts'])
                pcd_target.paint_uniform_color([1.0, 0.8, 0.0])

                trajectory_history.append(smooth_t.copy())
                trajectory_pcd.points = o3d.utility.Vector3dVector(np.array(trajectory_history))
                trajectory_pcd.paint_uniform_color([1.0, 0.0, 0.0])

                if current_obb is not None:
                    vis.remove_geometry(current_obb, reset_bounding_box=False)

                current_obb = o3d.geometry.OrientedBoundingBox(center=smooth_t, R=smooth_R, extent=np.array([0.45, 0.45, 0.45]))
                current_obb.color = (0.0, 1.0, 0.0)
                vis.add_geometry(current_obb, reset_bounding_box=False)

        vis.update_geometry(pcd_obs)
        vis.update_geometry(pcd_ground)
        vis.update_geometry(pcd_target)
        vis.update_geometry(trajectory_pcd)
        vis.poll_events()
        vis.update_renderer()

        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx:04d} | Active Tracking OK | Target Pos: [{smooth_t[0]:.2f}, {smooth_t[1]:.2f}, {smooth_t[2]:.2f}]")

    vis.destroy_window()
    print("[SUCCESS] Pipeline execution finished cleanly.")

if __name__ == "__main__":
    main()