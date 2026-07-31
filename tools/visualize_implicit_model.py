#!/usr/bin/env python3
"""
visualize_implicit_model.py - Visualize GPIS-W model: primitives and signed distance field.
"""

import argparse
import sys
import numpy as np
import open3d as o3d
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from turtlebot_tracker.core.implicit_surface import load_model


def create_arrow(origin, direction, length=0.04, color=[1, 0, 0]):
    """Create a 3D arrow mesh."""
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.003, cone_radius=0.006,
        cylinder_height=length * 0.7, cone_height=length * 0.3
    )
    z = np.array([0, 0, 1])
    v = np.cross(z, direction)
    s = np.linalg.norm(v)
    c = np.dot(z, direction)
    if s < 1e-6:
        R = np.eye(3) if c > 0 else np.diag([1, -1, -1])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))
    arrow.rotate(R, center=(0, 0, 0))
    arrow.translate(origin)
    arrow.paint_uniform_color(color)
    return arrow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="config/implicit_model.json")
    parser.add_argument("--grid_res", type=float, default=0.05,
                        help="Grid resolution for field visualization")
    parser.add_argument("--show_surface", action="store_true",
                        help="Extract and show zero-level surface (requires scikit-image)")
    args = parser.parse_args()

    model = load_model(args.model)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="GPIS-W Model", width=1280, height=720)
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.05, 0.05, 0.05])
    render_opt.point_size = 3.0

    # 1. Primitives (spheres + arrows)
    for p, n, h in zip(model.P, model.N, model.H):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=h * 0.4)
        sphere.translate(p)
        sphere.paint_uniform_color([0.3, 0.5, 0.8])
        vis.add_geometry(sphere)
        arrow = create_arrow(p, n, length=h * 0.6, color=[1, 0.6, 0])
        vis.add_geometry(arrow)

    # 2. Sample field on a coarse grid and color points by sign(f)
    # Compute bounding box
    min_pt = model.P.min(axis=0) - 0.15
    max_pt = model.P.max(axis=0) + 0.15
    grid_x = np.arange(min_pt[0], max_pt[0], args.grid_res)
    grid_y = np.arange(min_pt[1], max_pt[1], args.grid_res)
    grid_z = np.arange(min_pt[2], max_pt[2], args.grid_res)
    X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    pts_grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    print(f"[INFO] Evaluating field on {len(pts_grid)} grid points...")
    f_vals, _, _ = model.evaluate(pts_grid, compute_var=False)

    # Color points: red for inside (f<0), blue for outside (f>0), white near surface
    colors = np.zeros((len(pts_grid), 3))
    f_abs = np.abs(f_vals)
    # Near surface: white (|f| < 0.01)
    near_surface = f_abs < 0.01
    colors[near_surface] = [1.0, 1.0, 1.0]
    # Inside: red
    inside = f_vals < -0.01
    colors[inside] = [1.0, 0.2, 0.2]
    # Outside: blue
    outside = f_vals > 0.01
    colors[outside] = [0.2, 0.2, 1.0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_grid)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

    # 3. Coordinate frame
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))

    print("[INFO] Visualization ready. Close window to exit.")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()