"""
tools/visualize_canonical_model.py - 3D Organic Crystal Model & Radiometric c0_SH Inspector
Renders section-colored points, transparent GMM ellipsoids, and prints exact c0_SH physical values.
Usage: python tools/visualize_canonical_model.py
"""

import argparse
import json
import open3d as o3d
import numpy as np
from pathlib import Path

def create_transparent_ellipsoid(mu: np.ndarray, scales: np.ndarray, R_mat: np.ndarray, color: list) -> o3d.geometry.TriangleMesh:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere.vertices = o3d.utility.Vector3dVector(np.asarray(sphere.vertices) * scales * 1.5)
    sphere.rotate(R_mat, center=(0, 0, 0))
    sphere.translate(mu)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    return sphere

def main():
    parser = argparse.ArgumentParser(description="Visualize 3D Canonical Gaussian Model")
    parser.add_argument("--splats_json", type=str, default="config/canonical_turtlebot2.json", help="Splats JSON")
    parser.add_argument("--points_json", type=str, default="config/canonical_points.json", help="Points JSON (optional)")
    args = parser.parse_args()

    splat_path = Path(args.splats_json)
    points_path = Path(args.points_json)

    if not splat_path.exists():
        print(f"[ERROR] Splats JSON not found: {splat_path}")
        print("Run 'python tools/build_canonical_model.py' first.")
        return

    with open(splat_path, "r") as f:
        splat_data = json.load(f)
    gaussians = splat_data.get("canonical_gaussians", [])

    # Load points if available
    points_data = None
    if points_path.exists():
        with open(points_path, "r") as f:
            points_data = json.load(f)

    domain_colors = [
        [0.0, 1.0, 0.0], [1.0, 0.95, 0.0], [0.0, 0.8, 1.0],
        [1.0, 0.4, 0.0], [0.8, 0.0, 1.0], [1.0, 0.2, 0.2],
        [0.2, 0.8, 0.8], [0.8, 0.4, 0.8]
    ]

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="MOCD-Lite/SE(2) - Organic Crystal Model", width=1280, height=720)

    render_opt = vis.get_render_option()
    render_opt.point_size = 4.0

    # Render points if available
    if points_data:
        pts = np.array(points_data["canonical_points"])
        labels = np.array(points_data["canonical_point_labels"])
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        colors = np.array([domain_colors[lbl % len(domain_colors)] for lbl in labels])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(pcd)
        print(f"[INFO] Rendered {len(pts):,} points from {points_path}")
    else:
        print("[WARNING] Points JSON not found. Only ellipsoids will be shown.")
        print("[INFO] Run 'python tools/build_canonical_model.py' to generate points.")

    # Render splats
    print(f"\n--- ORGANIC CRYSTAL DOMAINS ({len(gaussians)}) ---")
    for idx, g in enumerate(gaussians):
        mu = np.array(g["mu"])
        scales = np.array(g["scales"])
        R_mat = np.array(g.get("rotation", np.eye(3)))
        c0_sh = g.get("sh_c0", 28.2)
        color = domain_colors[idx % len(domain_colors)]

        ellipsoid = create_transparent_ellipsoid(mu, scales, R_mat, color)
        vis.add_geometry(ellipsoid)

        bbox = o3d.geometry.LineSet.create_from_oriented_bounding_box(
            o3d.geometry.OrientedBoundingBox(center=mu, R=R_mat, extent=scales * 2.0)
        )
        bbox.paint_uniform_color(color)
        vis.add_geometry(bbox)

        print(f"  #{idx}: weight={g['weight']*100:.1f}% | c0_SH={c0_sh:.2f} | "
              f"pos=({mu[0]:.3f}, {mu[1]:.3f}, {mu[2]:.3f}) | scales=({scales[0]:.3f}, {scales[1]:.3f}, {scales[2]:.3f})")

    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25))
    print("\n[INFO] Drag to rotate, scroll to zoom. Close window to exit.")
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()