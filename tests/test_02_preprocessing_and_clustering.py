"""
test_02_preprocessing_and_clustering.py - Visualizes RANSAC ground plane and DBSCAN clusters with Gaussian ellipsoids.
"""

import sys
import glob
from pathlib import Path
import open3d as o3d
import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "src"))
from turtlebot_tracker.dataloader import MCAPLiDARLoader
from turtlebot_tracker.preprocessor import PointCloudPreprocessor
from turtlebot_tracker.clustering import ClusterGaussianFitter

def create_ellipsoid_mesh(mu: np.ndarray, cov: np.ndarray, color=[1.0, 0.0, 0.0]) -> o3d.geometry.TriangleMesh:
    """Creates a 3D ellipsoid mesh representing a Gaussian covariance matrix."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-5)
    scales = np.sqrt(eigvals)

    # Scale vertices
    vertices = np.asarray(mesh.vertices) * scales
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    # Rotate and translate
    mesh.rotate(eigvecs, center=(0, 0, 0))
    mesh.translate(mu)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

def main():
    bag_files = sorted(glob.glob("data/bags/*/*.mcap"))
    if not bag_files:
        print("[ERROR] No .mcap files found in data/bags/*/")
        return

    loader = MCAPLiDARLoader(bag_files[0])
    preprocessor = PointCloudPreprocessor(voxel_size=0.03, distance_threshold=0.08)
    fitter = ClusterGaussianFitter(eps=0.22, min_points=15)

    print("[TEST 2] Processing Ground Removal & DBSCAN Clustering...")

    for frame_idx, (ts, pts) in enumerate(loader.stream_point_clouds()):
        obstacles, ground = preprocessor.process(pts)
        candidate_clusters = fitter.extract_clusters_and_fit_gaussians(obstacles)

        print(f"Frame {frame_idx:04d} | Obstacle Pts: {len(obstacles)} | Ground Pts: {len(ground)} | Clusters: {len(candidate_clusters)}")

        if len(candidate_clusters) > 0 and frame_idx > 5:
            # Render a detailed breakdown of frame
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Test 02 - Segmentation & Gaussian Fitting", width=1280, height=720)

            # Ground Cloud (Light Blue)
            pcd_ground = o3d.geometry.PointCloud()
            pcd_ground.points = o3d.utility.Vector3dVector(ground)
            pcd_ground.paint_uniform_color([0.2, 0.6, 1.0])
            vis.add_geometry(pcd_ground)

            # Colors for clusters
            colors = [[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0.5, 0]]

            for idx, cluster in enumerate(candidate_clusters):
                c_color = colors[idx % len(colors)]
                pcd_c = o3d.geometry.PointCloud()
                pcd_c.points = o3d.utility.Vector3dVector(cluster['cluster_pts'])
                pcd_c.paint_uniform_color(c_color)
                vis.add_geometry(pcd_c)

                # Add Gaussian Ellipsoids
                for g in cluster['gaussians']:
                    ellipsoid = create_ellipsoid_mesh(g['mu'], g['cov'], color=c_color)
                    vis.add_geometry(ellipsoid)

            vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4))
            print("Visualizing Frame. Close Open3D window to finish test.")
            vis.run()
            vis.destroy_window()
            break

    print("[SUCCESS] Test 02 completed cleanly.")

if __name__ == "__main__":
    main()