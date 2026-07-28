"""
debug_and_benchmark.py - Diagnostic Tool to Benchmark Cluster Filtering, Shape Matching, and EMD Solver.
"""

import sys
import time
import glob
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "src"))
from turtlebot_tracker.dataloader import MCAPLiDARLoader
from turtlebot_tracker.preprocessor import PointCloudPreprocessor
from turtlebot_tracker.clustering import ClusterGaussianFitter
from turtlebot_tracker.optimal_transport import OptimalTransportMatcher

def main():
    bag_files = sorted(glob.glob("data/bags/*/*.mcap"))
    if not bag_files:
        print("[ERROR] No .mcap files found in data/bags/")
        return

    loader = MCAPLiDARLoader(bag_files[0])
    preprocessor = PointCloudPreprocessor(voxel_size=0.03)
    fitter = ClusterGaussianFitter(eps=0.22, min_points=20)
    ot_matcher = OptimalTransportMatcher()

    canonical_model = [
        {'mu': np.array([0.0, 0.0, -0.15]), 'scales': np.array([0.18, 0.18, 0.02]), 'weight': 0.3},
        {'mu': np.array([0.0, 0.0, 0.02]),  'scales': np.array([0.16, 0.16, 0.02]), 'weight': 0.3},
        {'mu': np.array([0.0, 0.0, 0.22]),  'scales': np.array([0.15, 0.15, 0.03]), 'weight': 0.3},
    ]

    print("\n=======================================================")
    print(" PIPELINE DIAGNOSTIC & BENCHMARK METRICS")
    print("=======================================================\n")

    for frame_idx, (ts, pts) in enumerate(loader.stream_point_clouds()):
        if frame_idx >= 5:
            break

        t0 = time.perf_counter()
        obstacles, ground = preprocessor.process(pts)
        t_prep = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        candidates = fitter.extract_clusters_and_fit_gaussians(obstacles)
        t_fit = (time.perf_counter() - t1) * 1000

        print(f"--- Frame {frame_idx:02d} ---")
        print(f"  [1] Preprocessing: {t_prep:.2f} ms | Obstacles: {len(obstacles)} pts")
        print(f"  [2] Valid Candidates Filtered: {len(candidates)} clusters (Fit time: {t_fit:.2f} ms)")

        for c_idx, cand in enumerate(candidates):
            t2 = time.perf_counter()
            cost, P_mat = ot_matcher.match_models(canonical_model, cand['gaussians'])
            t_ot = (time.perf_counter() - t2) * 1000
            
            ext = cand['cluster_pts'].max(axis=0) - cand['cluster_pts'].min(axis=0)
            print(f"      -> Candidate {c_idx}: Pts={cand['num_points']} | Extents=[{ext[0]:.2f}, {ext[1]:.2f}, {ext[2]:.2f}] m | OT Cost={cost:.4f} | OT Time={t_ot:.3f} ms")

        print("")

if __name__ == "__main__":
    main()