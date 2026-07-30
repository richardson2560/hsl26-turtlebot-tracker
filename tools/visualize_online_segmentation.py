"""
visualize_online_segmentation.py - Online Tracker with Known Object Gating.
Colors:
- Blue: Ground
- Gray: Walls
- Red: Known Static Candidates (from Prior Splats)
- Green: Valid New Candidates (Robot)
"""

import argparse
import json
import sys
import time
import numpy as np
import open3d as o3d
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from turtlebot_tracker.io.mcap_loader import MCAPLiDARLoader
from turtlebot_tracker.core.online_segmenter import OnlineSegmenter

def find_mcap_file(input_path: str) -> str:
    p = Path(input_path)
    if p.is_file(): return str(p)
    if p.is_dir():
        mcap_files = list(p.glob("*.mcap"))
        if mcap_files: return str(mcap_files[0])
    raise FileNotFoundError(f"No MCAP found at {input_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=str, required=True)
    parser.add_argument("--prior", type=str, default="config/static_map_prior.json")
    parser.add_argument("--metadata", type=str, default="data/outputs/static_metadata.json")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    # 1. Load Data
    try:
        mcap_path = find_mcap_file(args.bag)
        with open(args.prior, 'r') as f: prior = json.load(f)
        with open(args.metadata, 'r') as f: metadata = json.load(f)
    except Exception as e:
        print(f"[ERROR] {e}"); return

    segmenter = OnlineSegmenter(prior, metadata)
    loader = MCAPLiDARLoader(mcap_path)

    # 2. Visualizer Setup
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Online Stage 1: Gating & Segmentation", width=1280, height=720)
    
    # Static Background: Gray Wireframes
    for shell in prior.get('shells', []):
        box = o3d.geometry.OrientedBoundingBox(shell['center'], shell['axes'], np.array(shell['extents'])*2)
        wire = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
        wire.paint_uniform_color([0.3, 0.3, 0.3])
        vis.add_geometry(wire)

    # Static Objects (Prior Splats): Dim Red Wireframes
    for splat in prior.get('static_splats', []):
        mu, scales, R = np.array(splat['mu']), np.array(splat['scales']), np.array(splat['rotation'])
        box = o3d.geometry.OrientedBoundingBox(mu, R, scales * 2.5)
        wire = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
        wire.paint_uniform_color([0.5, 0.1, 0.1])
        vis.add_geometry(wire)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    vis.get_render_option().background_color = np.array([0.02, 0.02, 0.02])
    vis.get_render_option().point_size = 2.5

    print(f"[INFO] Streaming... Green = Valid Candidate, Red = Known Static")
    
    frame_idx = 0
    try:
        for ts, pts, intensity in loader.stream_point_clouds():
            t_start = time.perf_counter()

            pts_world, labels, valid_clusters = segmenter.classify_and_cluster(pts)

            # Coloring
            colors = np.zeros((len(pts_world), 3))
            colors[labels == -1] = [0.1, 0.1, 0.1] # Blind spot
            colors[labels == 0]  = [0.1, 0.1, 0.4] # Ground (Blue)
            colors[labels == 1]  = [0.2, 0.2, 0.2] # Wall (Gray)
            colors[labels == 2]  = [1.0, 0.2, 0.2] # Matched Static Object (Red)
            colors[labels == 3]  = [1.0, 1.0, 0.0] # Unclustered residual (Yellow)
            
            # Green for survivng clusters
            for cluster in valid_clusters:
                colors[cluster['indices']] = [0.0, 1.0, 0.0]

            pcd.points = o3d.utility.Vector3dVector(pts_world)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(pcd)
            
            if not vis.poll_events(): break
            vis.update_renderer()

            t_end = time.perf_counter()
            frame_idx += 1
            time.sleep(max(0, (1.0/args.fps) - (t_end - t_start)))
            
            if frame_idx % 10 == 0:
                print(f"\rFrame {frame_idx:04d} | Valid Candidates: {len(valid_clusters)}", end="")

    except KeyboardInterrupt: pass
    vis.destroy_window()

if __name__ == "__main__":
    main()