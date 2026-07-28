"""
run_offline_canonical.py - Extracts canonical Turtlebot2 Gaussians including 3D orientation matrices.
"""

import sys
import json
import glob
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "src"))
from turtlebot_tracker.dataloader import MCAPLiDARLoader
from turtlebot_tracker.preprocessor import PointCloudPreprocessor
from turtlebot_tracker.clustering import ClusterGaussianFitter

def main():
    static_bags = sorted(glob.glob("data/bags/*static*/*.mcap"))
    if not static_bags:
        static_bags = sorted(glob.glob("data/bags/*/*.mcap"))

    print(f"[OFFLINE] Extracting 3D canonical model from: {static_bags[0]}")
    loader = MCAPLiDARLoader(static_bags[0])
    preprocessor = PointCloudPreprocessor(voxel_size=0.02)
    fitter = ClusterGaussianFitter(eps=0.20, min_points=30)

    accumulated_points = []
    for idx, (ts, pts) in enumerate(loader.stream_point_clouds()):
        if idx >= 10:
            break
        obstacles, _ = preprocessor.process(pts)
        if len(obstacles) > 0:
            accumulated_points.append(obstacles)

    full_cloud = np.vstack(accumulated_points)
    candidates = fitter.extract_clusters_and_fit_gaussians(full_cloud, num_gaussians=4)

    if not candidates:
        print("[ERROR] Could not isolate Turtlebot2 cluster.")
        return

    best_cand = max(candidates, key=lambda c: c['num_points'])
    cluster_center = best_cand['centroid']

    canonical_gaussians = []
    for g in best_cand['gaussians']:
        rel_mu = g['mu'] - cluster_center
        canonical_gaussians.append({
            'mu': rel_mu.tolist(),
            'scales': g['scales'].tolist(),
            'rotation': g['rotation'].tolist(),  # Save 3x3 eigenvector rotation matrix!
            'weight': float(g['weight'])
        })

    out_path = Path("config/canonical_turtlebot2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump({'canonical_gaussians': canonical_gaussians}, f, indent=4)

    print(f"[SUCCESS] Saved 3D oriented canonical model to {out_path}")

if __name__ == "__main__":
    main()