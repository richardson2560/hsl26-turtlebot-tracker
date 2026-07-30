"""
visualize_static_map.py - Unified Auditor.
OBB Shells are Red Wireframes. Splats are Green Transparent Ellipsoids.
"""

import json
import numpy as np
import open3d as o3d
from pathlib import Path

def main():
    path = Path("config/static_map_prior.json")
    if not path.exists():
        print("[ERROR] JSON not found. Run build_static_map.py first.")
        return
        
    with open(path, 'r') as f: data = json.load(f)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Expert World Auditor - OBB & Splats", width=1280, height=720)

    # 1. Render Debug Points (Audit Colors)
    if "debug_points" in data:
        pts = np.array(data["debug_points"])
        lbls = np.array(data["debug_labels"])
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        colors = np.random.uniform(0.4, 1.0, size=(int(lbls.max()) + 1, 3))
        pcd.colors = o3d.utility.Vector3dVector(colors[lbls])
        vis.add_geometry(pcd)

    # 2. Render OBB Shells (Red Wireframes)
    for shell in data.get("shells", []):
        center = np.array(shell['center'])
        R = np.array(shell['axes'])
        extent = np.array(shell['extents']) * 2.0
        box = o3d.geometry.OrientedBoundingBox(center, R, extent)
        box.color = [1, 0.2, 0.2]
        vis.add_geometry(o3d.geometry.LineSet.create_from_oriented_bounding_box(box))
        
        # Local Axes
        l_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        l_frame.rotate(R, center=(0,0,0)).translate(center)
        vis.add_geometry(l_frame)

    # 3. Render Static Splats (Green Transparent-style Ellipsoids)
    for splat in data.get("static_splats", []):
        mu, v, w = np.array(splat['mu']), np.array(splat['scales']), np.array(splat['rotation'])
        ellip = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
        ellip.vertices = o3d.utility.Vector3dVector(np.asarray(ellip.vertices) * v * 2.0)
        ellip.rotate(w, center=(0,0,0)).translate(mu)
        
        ellip.paint_uniform_color([0.1, 0.5, 0.2])
        vis.add_geometry(ellip)
        
        # Wireframe overlay for transparency effect
        wire = o3d.geometry.LineSet.create_from_triangle_mesh(ellip)
        wire.paint_uniform_color([0.2, 1.0, 0.4])
        vis.add_geometry(wire)

    # 4. Context
    gz = data['ground']['z']
    grid = o3d.geometry.LineSet.create_from_oriented_bounding_box(
        o3d.geometry.OrientedBoundingBox([0,0,gz], np.eye(3), [12, 12, 0.001]))
    vis.add_geometry(grid)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    vis.get_render_option().background_color = np.array([0.02, 0.02, 0.02])
    vis.get_render_option().point_size = 1.5
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()