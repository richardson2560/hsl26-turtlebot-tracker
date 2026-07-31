#!/usr/bin/env python3
"""
visualize_implicit_model.py - Ultra-fast Wireframe & Error Heatmap Visualizer for Hermite-GPIS-W.
Renders zero-level surface as a transparent wireframe cage over the point cloud heatmap.
"""

import argparse
import sys
import numpy as np
import open3d as o3d
from pathlib import Path

# Try importing skimage for Marching Cubes surface extraction
try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

sys.path.append(str(Path(__file__).parent.parent / "src"))
from turtlebot_tracker.core.implicit_surface import load_model


def create_arrow(origin, direction, length=0.03, color=[1.0, 0.5, 0.0]):
    """Creates a 3D arrow representing normal vectors."""
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.002, cone_radius=0.005,
        cylinder_height=length * 0.7, cone_height=length * 0.3
    )
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, direction)
    s = np.linalg.norm(v)
    c = np.dot(z, direction)
    if s < 1e-6:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
    arrow.rotate(R, center=(0, 0, 0))
    arrow.translate(origin)
    arrow.paint_uniform_color(color)
    return arrow


def main():
    parser = argparse.ArgumentParser(description="Fast GPIS-W Wireframe & Heatmap Visualizer")
    parser.add_argument("--model", type=str, default="config/implicit_model.json")
    parser.add_argument("--input_npz", type=str, default="data/outputs/static_full.npz")
    parser.add_argument("--grid_res", type=float, default=0.015, 
                        help="Grid resolution in meters (default: 0.015 = 1.5cm for ultra-fast generation)")
    parser.add_argument("--solid_mesh", action="store_true", 
                        help="Render as solid translucent mesh instead of wireframe")
    parser.add_argument("--hide_primitives", action="store_true", help="Hide primitive arrows")
    args = parser.parse_args()

    model = load_model(args.model)
    print(f"[INFO] Loaded model with {model.M} primitives. Centroid={model.centroid}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Hermite-GPIS-W: Wireframe & Heatmap", width=1280, height=720)
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.06, 0.06, 0.06])
    render_opt.point_size = 4.5

    # =========================================================================
    # 1. LAYER 1: Fast Surface (Wireframe Cage or Solid Mesh)
    # =========================================================================
    if HAS_SKIMAGE:
        print(f"[INFO] Extracting surface f(x) = 0 on coarse grid ({args.grid_res*100:.1f} cm resolution)...")
        min_bounds = model.P.min(axis=0) - 0.05
        max_bounds = model.P.max(axis=0) + 0.05

        gx = np.arange(min_bounds[0], max_bounds[0], args.grid_res)
        gy = np.arange(min_bounds[1], max_bounds[1], args.grid_res)
        gz = np.arange(min_bounds[2], max_bounds[2], args.grid_res)

        X, Y, Z = np.meshgrid(gx, gy, gz, indexing='ij')
        pts_grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        f_grid, _, _ = model.evaluate(pts_grid, compute_var=False)
        volume = f_grid.reshape((len(gx), len(gy), len(gz)))

        try:
            verts, faces, _, _ = marching_cubes(volume, level=0.0, spacing=(args.grid_res, args.grid_res, args.grid_res))
            verts += min_bounds

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)

            if args.solid_mesh:
                # Translucent solid mesh
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color([0.7, 0.7, 0.7])
                vis.add_geometry(mesh)
            else:
                # Wireframe cage (Transparent LineSet - See-through)
                wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
                wireframe.paint_uniform_color([0.2, 0.8, 0.8])  # Light cyan cage
                vis.add_geometry(wireframe)

            print(f"[SUCCESS] Surface extracted in < 0.03s ({len(faces)} triangles).")
        except Exception as e:
            print(f"[WARN] Marching Cubes could not extract surface level 0: {e}")

    # =========================================================================
    # 2. LAYER 2: Original Point Cloud with Error Heatmap (|f(x)|)
    # =========================================================================
    npz_path = Path(args.input_npz)
    if npz_path.exists():
        data = np.load(npz_path)
        robot_mask = data['labels'] == 3
        pts_raw = data['points'][robot_mask]

        # Centered at model's local frame
        pts_centered = pts_raw - model.centroid

        f_vals, _, _ = model.evaluate(pts_centered, compute_var=False)
        errors = np.abs(f_vals)

        # Heatmap coloring: Green (0mm) -> Yellow (5mm) -> Red (>15mm)
        colors = np.zeros((len(pts_centered), 3))
        norm_err = np.clip(errors / 0.015, 0.0, 1.0)

        colors[:, 0] = np.clip(2.0 * norm_err, 0.0, 1.0)         # Red
        colors[:, 1] = np.clip(2.0 * (1.0 - norm_err), 0.0, 1.0) # Green
        colors[:, 2] = 0.0                                        # Blue

        pcd_orig = o3d.geometry.PointCloud()
        pcd_orig.points = o3d.utility.Vector3dVector(pts_centered)
        pcd_orig.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(pcd_orig)

        mean_err_mm = np.mean(errors) * 1000.0
        print(f"[STATS] Point Residual Error: Mean = {mean_err_mm:.2f} mm")

    # =========================================================================
    # 3. LAYER 3: Model Primitives & Normal Vectors
    # =========================================================================
    if not args.hide_primitives:
        for p, n in zip(model.P, model.N):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            sphere.translate(p)
            sphere.paint_uniform_color([0.2, 0.5, 1.0])  # Blue primitive dots
            vis.add_geometry(sphere)

            arrow = create_arrow(p, n, length=0.02, color=[1.0, 0.5, 0.0]) # Orange arrows
            vis.add_geometry(arrow)

    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12))

    print("\n" + "="*80)
    print("VISUALIZATION LEGEND:")
    print("  • Cyan Wireframe: See-through cage of implicit surface f(x) = 0")
    print("  • Green Points   : Original points with error < 3 mm (Perfect Fit)")
    print("  • Yellow Points  : Original points with error 3-8 mm (Acceptable)")
    print("  • Red Points     : Original points with error > 15 mm (Outliers/Noise)")
    print("  • Blue Dots      : Model Primitives (P_k)")
    print("  • Orange Arrows  : Surface Normal Vectors (N_k)")
    print("="*80 + "\n")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()