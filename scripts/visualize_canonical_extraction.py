"""
visualize_canonical_extraction.py - Visualizes extracted canonical Gaussians with true 3D orientations.
"""

import sys
import json
import glob
from pathlib import Path
import open3d as o3d
import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "src"))
from turtlebot_tracker.dataloader import MCAPLiDARLoader
from turtlebot_tracker.preprocessor import PointCloudPreprocessor
from turtlebot_tracker.clustering import ClusterGaussianFitter

def create_oriented_ellipsoid_mesh(mu: np.ndarray, scales: np.ndarray, R_mat: np.ndarray, color=[1.0, 0.0, 0.0]) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    # 1. Scale sphere
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * scales)
    # 2. Apply 3D eigenvector orientation matrix
    mesh.rotate(R_mat, center=(0, 0, 0))
    # 3. Translate to local centroid
    mesh.translate(mu)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

def main():
    canon_file = Path("config/canonical_turtlebot2.json")
    if not canon_file.exists():
        print("[ERROR] Run scripts/run_offline_canonical.py first.")
        return

    with open(canon_file, "r") as f:
        canon_data = json.load(f)['canonical_gaussians']

    static_bags = sorted(glob.glob("data/bags/*static*/*.mcap"))
    loader = MCAPLiDARLoader(static_bags[0])
    preprocessor = PointCloudPreprocessor(voxel_size=0.02)
    fitter = ClusterGaussianFitter(eps=0.20, min_points=30)

    accumulated = []
    for idx, (ts, pts) in enumerate(loader.stream_point_clouds()):
        if idx >= 10:
            break
        obs, _ = preprocessor.process(pts)
        if len(obs) > 0:
            accumulated.append(obs)

    full_cloud = np.vstack(accumulated)
    candidates = fitter.extract_clusters_and_fit_gaussians(full_cloud, num_gaussians=4)
    best_cand = max(candidates, key=lambda c: c['num_points'])
    cluster_center = best_cand['centroid']

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Oriented Canonical Model - Turtlebot2 3D Splats", width=1280, height=720)

    # Point cloud of Turtlebot2 (White)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(best_cand['cluster_pts'])
    pcd.paint_uniform_color([0.85, 0.85, 0.85])
    vis.add_geometry(pcd)

    colors = [[1, 0, 0], [0, 0.8, 0], [0, 0.5, 1], [1, 0.8, 0]]
    for idx, g in enumerate(canon_data):
        abs_mu = np.array(g['mu']) + cluster_center
        scales = np.array(g['scales'])
        R_mat = np.array(g.get('rotation', np.eye(3)))
        
        mesh = create_oriented_ellipsoid_mesh(abs_mu, scales, R_mat, color=colors[idx % len(colors)])
        vis.add_geometry(mesh)

    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=cluster_center))
    
    # Configure Top-Down Camera View
    ctr = vis.get_view_control()
    ctr.set_front([-0.5, -0.5, 0.70])
    ctr.set_lookat(cluster_center)
    ctr.set_up([0.0, 0.0, 1.0])
    
    print("[INSPECTION] Showing accurately oriented 3D Gaussian Ellipsoids over Turtlebot2.")
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()