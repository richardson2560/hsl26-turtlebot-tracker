"""
run_online_tracker.py - Corrected Online Batch Pipeline Execution Script.
"""

import csv
import glob
from pathlib import Path
import sys
import time
import yaml
import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "src"))

from turtlebot_tracker.core.candidate_filter import CandidateFilter
from turtlebot_tracker.core.preprocessor import LiDARPreprocessor
from turtlebot_tracker.core.registration import DirectGMMRegistrator
from turtlebot_tracker.core.segmentation import RangeImageSegmenter
from turtlebot_tracker.core.temporal_buffer import MotionCompensatedBuffer
from turtlebot_tracker.core.tracking import SE2ManifoldEKF
from turtlebot_tracker.io.mcap_loader import MCAPLiDARLoader


def process_bag_sequence(bag_path: str, config: dict, output_dir: Path) -> None:
    bag_name = Path(bag_path).parent.name if Path(bag_path).is_file() else Path(bag_path).name
    print(f"\n=======================================================")
    print(f" PROCESSING BAG: {bag_name}")
    print(f"=======================================================")

    loader = MCAPLiDARLoader(bag_path)
    preproc = LiDARPreprocessor(config)
    segmenter = RangeImageSegmenter(config)
    candidate_filter = CandidateFilter(config)
    temporal_buffer = MotionCompensatedBuffer(config)
    registrator = DirectGMMRegistrator(config)
    ekf_tracker = SE2ManifoldEKF(config)

    trajectory = []
    metric_rows = []
    last_timestamp = None

    for frame_idx, (ts, pts, intensity) in enumerate(loader.stream_point_clouds()):
        t_start = time.perf_counter()
        dt = 0.1 if last_timestamp is None else max(0.01, ts - last_timestamp)
        last_timestamp = ts

        # 1. Preprocessing (Rodrigues Z-Up + Static Map Veto)
        frame_data = preproc.process(ts, pts, intensity)

        # 2. Segmentation O(N)
        t_seg_start = time.perf_counter()
        clusters = segmenter.segment(frame_data)
        t_seg = (time.perf_counter() - t_seg_start) * 1000.0

        # 3. Cascaded Filtering
        candidates = candidate_filter.filter_candidates(clusters)
        passed_candidates = [c for c in candidates if c.passed_filters]

        # --- FIX: Store original candidates in buffer BEFORE any modification ---
        temporal_buffer.add_frame_candidates(candidates, ts)

        # Temporal accumulation (only if needed, on a copy of the candidate)
        if passed_candidates and len(passed_candidates[0].points) < 30:
            target_v = ekf_tracker.x[3:6]
            accum_pts, accum_int = temporal_buffer.get_accumulated_points(passed_candidates[0], target_v, ts)
            # Modify the candidate directly (buffer already has a copy of the original)
            passed_candidates[0].points = accum_pts
            passed_candidates[0].intensity = accum_int

        # 4. Direct GMM MAP Registration
        t_reg_start = time.perf_counter()
        state, best_cand = registrator.register_and_track(frame_data, candidates, ekf_tracker)
        t_reg = (time.perf_counter() - t_reg_start) * 1000.0

        t_total = (time.perf_counter() - t_start) * 1000.0

        best_score = best_cand.map_score if best_cand is not None else -999.0
        n_raw = len(pts)
        n_obs = len(frame_data.obstacle_points) if frame_data.obstacle_points is not None else 0

        metric_rows.append([
            frame_idx, ts, int(state.lifecycle_state), t_total, t_seg, t_reg,
            n_raw, n_obs, len(clusters), len(passed_candidates), best_score,
            state.surprise_triggered, state.nis, state.is_zupt_active,
            state.pose_se2[0], state.pose_se2[1], state.pose_se2[2]
        ])

        if state.pose_se2 is not None:
            trajectory.append(state.pose_se2.copy())

        if frame_idx % 25 == 0:
            print(f"Frame {frame_idx:04d} | State: {state.lifecycle_state.name:15s} | "
                  f"Pose: ({state.pose_se2[0]:.2f}, {state.pose_se2[1]:.2f}) m | "
                  f"NIS: {state.nis:.2f} | Latency: {t_total:.1f} ms")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{bag_name}_trajectory.npy", np.array(trajectory))

    csv_path = output_dir / f"{bag_name}_metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame', 'timestamp', 'fsm_state', 'latency_total_ms', 'latency_seg_ms', 'latency_reg_ms',
            'num_raw_points', 'num_obs_points', 'num_candidates', 'passed_candidates', 'best_map_score',
            'surprise_triggered', 'nis', 'zupt_active', 'pose_x', 'pose_y', 'pose_yaw'
        ])
        writer.writerows(metric_rows)

    print(f"[COMPLETED] Saved output metrics for {bag_name} -> {csv_path}")


def main():
    config_path = Path("config/default_params.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    bag_dirs = sorted(glob.glob("data/bags/*"))
    output_dir = Path("data/outputs")
    for bag_dir in bag_dirs:
        process_bag_sequence(bag_dir, config, output_dir)


if __name__ == "__main__":
    main()