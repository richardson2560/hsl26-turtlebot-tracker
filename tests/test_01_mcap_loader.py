"""
test_01_mcap_loader.py - Tests MCAP loading and renders raw Livox MID-360 point clouds.
"""

import sys
import glob
from pathlib import Path
import open3d as o3d
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from turtlebot_tracker.dataloader import MCAPLiDARLoader

def main():
    bag_files = sorted(glob.glob("data/bags/*/*.mcap"))
    if not bag_files:
        print("[ERROR] No .mcap files found in data/bags/*/. Place dataset folders under data/bags/.")
        return

    test_bag = bag_files[0]
    print(f"[TEST 1] Loading Point Clouds from: {test_bag}")

    loader = MCAPLiDARLoader(test_bag)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Test 01 - Raw Livox MID-360 Point Cloud", width=1280, height=720)

    pcd_o3d = o3d.geometry.PointCloud()
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(coord_frame)

    first_frame = True

    for frame_idx, (ts, pts) in enumerate(loader.stream_point_clouds()):
        print(f"Frame {frame_idx:04d} | Timestamp: {ts:.3f} s | Point Count: {len(pts)}")
        
        pcd_o3d.points = o3d.utility.Vector3dVector(pts)
        pcd_o3d.paint_uniform_color([0.7, 0.7, 0.7])  # Gray points

        if first_frame:
            vis.add_geometry(pcd_o3d)
            first_frame = False
        else:
            vis.update_geometry(pcd_o3d)

        vis.poll_events()
        vis.update_renderer()

    print("[SUCCESS] Test 01 completed cleanly.")
    vis.destroy_window()

if __name__ == "__main__":
    main()