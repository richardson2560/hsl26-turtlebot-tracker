"""
tools/visualize_canonical_model.py - 3D Organic Crystal Model & OBB Shells.
Renders section-colored points, transparent GMM ellipsoids, OBB wireframes,
and optionally VMF direction arrows.
Compatible with Open3D versions that do not support per-mesh materials.
"""

import argparse
import json
import open3d as o3d
import numpy as np
from pathlib import Path


def create_ellipsoid(mu, scales, R_mat, color):
    """Create a solid-colored ellipsoid (no transparency) for compatibility."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    # Scale vertices: use 1.5x to be slightly larger than the bounding box
    sphere.vertices = o3d.utility.Vector3dVector(np.asarray(sphere.vertices) * scales * 1.5)
    sphere.rotate(R_mat, center=(0, 0, 0))
    sphere.translate(mu)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    return sphere


def create_arrow(origin, direction, length=0.1, color=[1, 1, 0]):
    """Create a 3D arrow for VMF direction."""
    if np.linalg.norm(direction) < 1e-6:
        return None
    direction = direction / np.linalg.norm(direction)
    # Compute rotation to align Z-axis with direction
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, direction)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, direction)
    if s < 1e-6:
        R = np.eye(3) if c > 0 else np.diag([1, -1, -1])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.005, cone_radius=0.015,
        cylinder_height=length * 0.7, cone_height=length * 0.3
    )
    arrow.rotate(R, center=(0, 0, 0))
    arrow.translate(origin)
    arrow.paint_uniform_color(color)
    return arrow


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D Canonical Gaussian Model with OBB shells")
    parser.add_argument("--model_json", type=str, default="config/canonical_turtlebot2.json",
                        help="Path to canonical model JSON")
    parser.add_argument("--points_json", type=str, default="config/canonical_points.json",
                        help="Path to canonical points JSON (optional)")
    parser.add_argument("--show_vmf", action="store_true",
                        help="Show VMF direction arrows")
    args = parser.parse_args()

    model_path = Path(args.model_json)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return

    with open(model_path, 'r') as f:
        model = json.load(f)

    # Support both flat and tree formats
    gaussians = model.get("canonical_gaussians", model.get("gaussians", []))
    shells = model.get("shells", [])
    metadata = model.get("metadata", {})

    if not gaussians:
        print("[ERROR] No Gaussian components found in the model.")
        return

    # Color palette for components
    domain_colors = [
        [0.0, 1.0, 0.0], [1.0, 0.95, 0.0], [0.0, 0.8, 1.0],
        [1.0, 0.4, 0.0], [0.8, 0.0, 1.0], [1.0, 0.2, 0.2],
        [0.2, 0.8, 0.8], [0.8, 0.4, 0.8]
    ]

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="MOCD-Lite: Organic Crystal Model", width=1280, height=720)
    render_opt = vis.get_render_option()
    render_opt.point_size = 4.0
    render_opt.background_color = np.array([0.05, 0.05, 0.05])

    # ---- Load and render canonical points (if available) ----
    points_path = Path(args.points_json)
    if points_path.exists():
        with open(points_path, 'r') as f:
            pts_data = json.load(f)
        pts = np.array(pts_data["canonical_points"])
        labels = np.array(pts_data["canonical_point_labels"])
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        colors = np.array([domain_colors[lbl % len(domain_colors)] for lbl in labels])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(pcd)
        print(f"[INFO] Rendered {len(pts):,} points from {points_path}")
    else:
        print("[WARNING] Points JSON not found. Only ellipsoids and shells will be shown.")

    # ---- Render splats (ellipsoids) ----
    print(f"\n--- ORGANIC CRYSTAL DOMAINS ({len(gaussians)}) ---")
    for idx, g in enumerate(gaussians):
        mu = np.array(g["mu"])
        scales = np.array(g["scales"])
        R_mat = np.array(g.get("rotation", np.eye(3)))
        c0_sh = g.get("sh_c0", 28.2)
        color = domain_colors[idx % len(domain_colors)]

        # Ellipsoid (solid color, no transparency)
        ellipsoid = create_ellipsoid(mu, scales, R_mat, color)
        vis.add_geometry(ellipsoid)

        # Bounding box wireframe (oriented bounding box)
        obb = o3d.geometry.OrientedBoundingBox(center=mu, R=R_mat, extent=scales * 2.0)
        bbox = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
        bbox.paint_uniform_color(color)
        vis.add_geometry(bbox)

        # VMF direction arrow (optional)
        if args.show_vmf:
            mu_dir = np.array(g.get("mu_dir", [0, 0, 1]))
            kappa = g.get("kappa", 1.0)
            if np.linalg.norm(mu_dir) > 0.1:
                arrow = create_arrow(mu, mu_dir, length=0.12, color=[1, 1, 0])
                if arrow is not None:
                    vis.add_geometry(arrow)

        print(f"  #{idx}: weight={g['weight']*100:.1f}% | c0_SH={c0_sh:.2f} | "
              f"pos=({mu[0]:.3f}, {mu[1]:.3f}, {mu[2]:.3f}) | scales=({scales[0]:.3f}, {scales[1]:.3f}, {scales[2]:.3f}) | "
              f"kappa={g.get('kappa', 0):.2f}")

    # ---- Render OBB shells (robot parts) ----
    if shells:
        print(f"\n--- ROBOT SHELLS ({len(shells)}) ---")
        for idx, s in enumerate(shells):
            center = np.array(s["center"])
            axes = np.array(s["axes"])
            extents = np.array(s["extents"])
            obb = o3d.geometry.OrientedBoundingBox(center, axes, extents * 2.0)
            wire = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
            wire.paint_uniform_color([0.2, 0.6, 0.8])  # light blue
            vis.add_geometry(wire)

    # Coordinate frame
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    vis.add_geometry(coord)

    print("\n[INFO] Drag to rotate, scroll to zoom. Close window to exit.")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()