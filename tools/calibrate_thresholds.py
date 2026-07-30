"""
calibrate_thresholds.py - Data-Driven Automatic Parameter Auto-Tuner.

Processes calibration bags (static_1m, rot_1m) to compute empirical percentiles
for density, hull volume, solidity, Taubin 2D arc residuals, and reflectance std.
Automatically writes calibrated values to config/default_params.yaml.
"""

import argparse
from pathlib import Path
import sys
import yaml
import numpy as np
import open3d as o3d

# Add src to Python Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from turtlebot_tracker.core.candidate_filter import CandidateFilter
from turtlebot_tracker.core.preprocessor import LiDARPreprocessor
from turtlebot_tracker.io.mcap_loader import MCAPLiDARLoader


def main():
    parser = argparse.ArgumentParser(description="Auto-tune default_params.yaml from empirical data.")
    parser.add_argument("--static_bag", type=str, default="data/bags/rosbag2_2026_06_27-18_27-static_1m")
    parser.add_argument("--config", type=str, default="config/default_params.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    bag_path = Path(args.static_bag)
    if not bag_path.exists():
        print(f"[WARNING] Static bag not found at {bag_path}. Looking in data/bags/*static*...")
        found = list(Path("data/bags").glob("*static*"))
        if found:
            bag_path = found[0]
        else:
            print("[ERROR] No calibration static bag found.")
            return

    print(f"\n=======================================================")
    print(f" AUTO-TUNING HYPERPARAMETERS FROM: {bag_path.name}")
    print(f"=======================================================\n")

    loader = MCAPLiDARLoader(str(bag_path))
    preprocessor = LiDARPreprocessor(cfg)
    candidate_filter = CandidateFilter(cfg)

    n_points_list = []
    v_hull_list = []
    solidity_list = []
    rms_fit_list = []
    volumetricity_list = []
    intensity_list = []

    for idx, (ts, pts, intensity) in enumerate(loader.stream_point_clouds()):
        if idx >= 30:
            break
        frame_data = preprocessor.process(ts, pts, intensity)
        if frame_data.obstacle_points is None or len(frame_data.obstacle_points) < 15:
            continue

        # Extract candidates
        from turtlebot_tracker.core.segmentation import RangeImageSegmenter
        segmenter = RangeImageSegmenter(cfg)
        clusters = segmenter.segment(frame_data)

        for cand in clusters:
            pts_c = cand.points
            if len(pts_c) < 12:
                continue

            n_points_list.append(len(pts_c))
            
            # Hull metrics
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pts_c)
                v_h = float(hull.volume)
                sol = len(pts_c) / max(v_h, 1e-5)
                v_hull_list.append(v_h)
                solidity_list.append(sol)
            except Exception:
                pass

            # Taubin fit residual
            _, _, r_fit, rms_fit = candidate_filter._fit_taubin_circle_2d(pts_c[:, :2])
            if rms_fit < 10.0:
                rms_fit_list.append(rms_fit)

            # Volumetricity
            cov = np.cov(pts_c.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.maximum(eigvals, 1e-6)
            volum = float(eigvals[0] / np.sum(eigvals))
            volumetricity_list.append(volum)

            # Intensity std
            r_i = np.linalg.norm(pts_c, axis=1)
            r_clamped = np.minimum(r_i, 2.5)
            cos_eta = np.maximum(np.abs(pts_c[:, 2]) / np.maximum(r_i, 1e-6), 0.1)
            i_corr = np.clip(cand.intensity * (r_clamped**2) / cos_eta, 0.0, 255.0)
            intensity_list.extend(i_corr.tolist())

    if not n_points_list:
        print("[ERROR] Insufficient candidate clusters extracted for calibration.")
        return

    # Calculate empirical non-parametric statistics
    n_pts_arr = np.array(n_points_list)
    v_hull_arr = np.array(v_hull_list)
    solidity_arr = np.array(solidity_list)
    rms_fit_arr = np.array(rms_fit_list)
    volum_arr = np.array(volumetricity_list)
    intensity_arr = np.array(intensity_list)

    n_0_tuned = float(np.median(n_pts_arr))
    iqr_n = np.percentile(n_pts_arr, 75) - np.percentile(n_pts_arr, 25)
    kappa_n_tuned = float(max(1.0, iqr_n / 1.35))

    rho_piso_tuned = float(np.percentile(solidity_arr, 5))
    rho_base_tuned = float(np.percentile(solidity_arr, 95) - rho_piso_tuned)

    v_nominal_tuned = float(np.percentile(v_hull_arr, 10))
    taubin_rms_max_tuned = float(np.percentile(rms_fit_arr, 95))
    volumetricity_min_tuned = float(max(0.01, np.percentile(volum_arr, 5)))
    sh_intensity_std_tuned = float(np.std(intensity_arr))

    print("--- CALIBRATED HYPERPARAMETERS (DATA-DRIVEN) ---")
    print(f"  candidate_filter.n_0              : {n_0_tuned:.2f}")
    print(f"  candidate_filter.kappa_n          : {kappa_n_tuned:.2f}")
    print(f"  candidate_filter.rho_piso         : {rho_piso_tuned:.2f}")
    print(f"  candidate_filter.rho_base         : {rho_base_tuned:.2f}")
    print(f"  candidate_filter.v_nominal        : {v_nominal_tuned:.4f} m^3")
    print(f"  candidate_filter.taubin_rms_max   : {taubin_rms_max_tuned:.4f} m")
    print(f"  candidate_filter.volumetricity_min: {volumetricity_min_tuned:.4f}")
    print(f"  registration.sh_intensity_std     : {sh_intensity_std_tuned:.2f}")

    # Update YAML configuration
    cfg['candidate_filter']['v_nominal'] = round(v_nominal_tuned, 5)
    cfg['candidate_filter']['rho_base'] = round(rho_base_tuned, 2)
    cfg['candidate_filter']['rho_piso'] = round(rho_piso_tuned, 2)
    cfg['candidate_filter']['taubin_rms_max'] = round(taubin_rms_max_tuned, 4)
    cfg['candidate_filter']['volumetricity_min'] = round(volumetricity_min_tuned, 4)
    cfg['registration']['sh_intensity_std'] = round(sh_intensity_std_tuned, 2)

    with open(config_path, 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\n[SUCCESS] Updated {config_path} with auto-tuned parameters.")


if __name__ == "__main__":
    main()