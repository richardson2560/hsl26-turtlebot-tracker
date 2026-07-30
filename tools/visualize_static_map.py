"""
tools/visualize_static_map.py - Visualize static map prior (ground + splats).
Optionally overlay background points.
"""

import argparse
import json
import numpy as np
import open3d as o3d
from pathlib import Path

def create_ellipsoid(mu, scales, R, color, alpha=0.3):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere.vertices = o3d.utility.Vector3dVector(np.asarray(sphere.vertices) * scales)
    sphere.rotate(R, center=(0, 0, 0))
    sphere.translate(mu)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    return sphere

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=str, default="config/static_map_prior.json")
    parser.add_argument("--background_npz", type=str, default="data/outputs/background_candidate.npz",
                        help="Background points NPZ (optional)")
    parser.add_argument("--show_robot", action="store_true",
                        help="Overlay robot canonical points (if available)")
    args = parser.parse_args()

    # Load prior
    with open(args.prior, 'r') as f:
        data = json.load(f)

    z_ground = data["z_ground"]
    splats = data.get("splats", [])
    print(f"Ground Z: {z_ground:.3f} m")
    print(f"Static splats: {len(splats)}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Static Map", width=1280, height=720)

    # Ground grid
    grid_size = 6.0
    grid_pts = np.array([
        [-grid_size, -grid_size, z_ground],
        [ grid_size, -grid_size, z_ground],
        [ grid_size,  grid_size, z_ground],
        [-grid_size,  grid_size, z_ground]
    ])
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(grid_pts)
    grid.lines = o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0]])
    grid.paint_uniform_color([0.5, 0.5, 0.5])
    vis.add_geometry(grid)

    # Coordinate frame
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(coord)

    # Load background points (if provided)
    bg_path = Path(args.background_npz)
    if bg_path.exists():
        data_bg = np.load(bg_path)
        bg_pts = data_bg["points"]
        pcd_bg = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bg_pts))
        pcd_bg.paint_uniform_color([0.3, 0.3, 0.35])  # Dark gray
        vis.add_geometry(pcd_bg)
        print(f"[INFO] Loaded {len(bg_pts):,} background points.")
    else:
        print("[INFO] No background NPZ found; only splats and grid shown.")

    # Render splats
    colors = [
        [0.6, 0.6, 0.8], [0.7, 0.5, 0.7], [0.5, 0.7, 0.7],
        [0.6, 0.4, 0.6], [0.4, 0.6, 0.8], [0.8, 0.6, 0.4]
    ]
    for i, splat in enumerate(splats):
        mu = np.array(splat["mu"])
        scales = np.array(splat["scales"])
        R = np.array(splat.get("rotation", np.eye(3)))
        color = colors[i % len(colors)]
        ellipsoid = create_ellipsoid(mu, scales, R, color, alpha=0.3)
        vis.add_geometry(ellipsoid)
        bbox = o3d.geometry.LineSet.create_from_oriented_bounding_box(
            o3d.geometry.OrientedBoundingBox(center=mu, R=R, extent=scales * 2.0)
        )
        bbox.paint_uniform_color(color)
        vis.add_geometry(bbox)

    # Optionally overlay robot points
    if args.show_robot:
        robot_path = Path("config/canonical_points.json")
        if robot_path.exists():
            with open(robot_path, 'r') as f:
                pts_data = json.load(f)
            pts = np.array(pts_data["canonical_points"])
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            pcd.paint_uniform_color([0.2, 0.8, 0.2])  # Green
            vis.add_geometry(pcd)
            print("[INFO] Robot reference points overlaid.")

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()