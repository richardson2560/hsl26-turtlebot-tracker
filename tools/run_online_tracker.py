"""
tools/run_online_tracker.py - Main pipeline runner for all 7 bags.
"""

import sys
import time
import glob
import yaml
import csv
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from turtlebot_tracker.io.mcap_loader import MCAPLiDARLoader
from turtlebot_tracker.core.preprocessor import LiDARPreprocessor
from turtlebot_tracker.core.segmentation import RangeImageSegmenter
from turtlebot_tracker.core.candidate_filter import CandidateFilter
from turtlebot_tracker.core.registration import DirectGMMRegistrator
from turtlebot_tracker.core.tracking import SE2ManifoldEKF
from turtlebot_tracker.datatypes import FrameData

def process_bag(bag_path, config, output_dir):
    print(f"\nProcessing: {bag_path}")
    loader = MCAPLiDARLoader(bag_path)
    preproc = LiDARPreprocessor(config)
    segmenter = RangeImageSegmenter(config)
    filter_ = CandidateFilter(config)
    registrator = DirectGMMRegistrator(config)
    ekf = SE2ManifoldEKF(config)

    trajectory = []
    latencies = []
    nis_values = []
    is_initialized = False
    last_timestamp = None

    for frame_idx, (ts, pts, intensity) in enumerate(loader.stream_point_clouds()):
        t0 = time.perf_counter()
        dt = 0.1 if last_timestamp is None else max(0.01, ts - last_timestamp)
        last_timestamp = ts

        frame_data = preproc.process(ts, pts, intensity)
        clusters = segmenter.segment(frame_data)
        candidates = filter_.filter_candidates(clusters)
        state, best_cand = registrator.register_and_track(frame_data, candidates, ekf)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)
        nis_values.append(state.nis if hasattr(state, 'nis') else 0.0)

        if frame_idx % 50 == 0:
            print(f"Frame {frame_idx:04d} | Pose: ({state.pose_se2[0]:.2f}, {state.pose_se2[1]:.2f}) | NIS: {state.nis:.2f} | ZUPT: {state.is_zupt_active}")

        if state.pose_se2 is not None:
            trajectory.append(state.pose_se2.copy())

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bag_name = Path(bag_path).parent.name

    # Trajectory
    np.save(out_dir / f"{bag_name}_trajectory.npy", np.array(trajectory))

    # Latency & NIS logs
    with open(out_dir / f"{bag_name}_metrics.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'latency_ms', 'nis'])
        for i, (lat, nis) in enumerate(zip(latencies, nis_values)):
            writer.writerow([i, lat, nis])

    print(f"  Done. Avg latency: {np.mean(latencies):.2f} ms, 99th: {np.percentile(latencies, 99):.2f} ms")

def main():
    config_path = Path("config/default_params.yaml")
    if not config_path.exists():
        print("[ERROR] config/default_params.yaml not found.")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    bag_dirs = sorted(glob.glob("data/bags/*"))
    if not bag_dirs:
        print("[ERROR] No bag directories found in data/bags/")
        return

    output_dir = "data/outputs"
    for bag_dir in bag_dirs:
        process_bag(bag_dir, config, output_dir)

if __name__ == "__main__":
    main()