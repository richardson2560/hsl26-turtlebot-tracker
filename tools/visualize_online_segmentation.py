"""
visualize_online_segmentation.py - Master Pipeline Viewer with GPIS‑W or GMM registrator.
"""

import argparse
import json
import sys
import time
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.stats import chi2

sys.path.append(str(Path(__file__).parent.parent / "src"))
from turtlebot_tracker.io.mcap_loader import MCAPLiDARLoader
from turtlebot_tracker.core.online_segmenter import OnlineSegmenter
from turtlebot_tracker.core.registration import DirectGMMRegistrator
try:
    from turtlebot_tracker.core.registration import GPISRegistrator
except ImportError:
    GPISRegistrator = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=str, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--registrator", choices=["gmm", "gpis"], default="gpis",
                        help="Registrator type: gmm (Gaussian mixture) or gpis (GPIS‑W)")
    parser.add_argument("--model", type=str, default="config/implicit_model.json",
                        help="GPIS model JSON (only used with --registrator gpis)")
    parser.add_argument("--gmm_model", type=str, default="config/canonical_turtlebot2.json",
                        help="GMM model JSON (only used with --registrator gmm)")
    args = parser.parse_args()

    # Load components
    with open("config/static_map_prior.json", 'r') as f:
        prior = json.load(f)
    with open("data/outputs/static_metadata.json", 'r') as f:
        meta = json.load(f)

    # --- Instantiate registrator ---
    if args.registrator == "gpis":
        if GPISRegistrator is None:
            print("[ERROR] GPISRegistrator not available. Install or use --registrator gmm.")
            return
        registrator = GPISRegistrator({"registration": {"score_threshold": 2.0}}, args.model)
        use_gpis = True
        print(f"[INFO] Using GPIS registrator with model {args.model}")
    else:
        with open(args.gmm_model, 'r') as f:
            model = json.load(f)
        robot_volume = model.get("metadata", {}).get("robot_volume", 0.05)
        registrator = DirectGMMRegistrator(
            {"registration": {"sh_intensity_std": 1000.0, "container_volume": 5.0}},
            args.gmm_model
        )
        use_gpis = False
        outlier_w = registrator.outlier_weight
        lrt_threshold = chi2.ppf(0.999, df=3)
        print(f"[INFO] Using GMM registrator with model {args.gmm_model}")

    segmenter = OnlineSegmenter(prior, meta)

    p = Path(args.bag)
    mcap = str(list(p.glob("*.mcap"))[0]) if p.is_dir() else str(p)
    loader = MCAPLiDARLoader(mcap)

    # Open3D setup
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"MOCD-Lite: {args.registrator.upper()}", width=1280, height=720)

    # Background shells
    for s in prior.get('shells', []):
        extents = np.array(s['extents']) * 2.0
        if np.any(extents <= 0):
            continue
        box = o3d.geometry.OrientedBoundingBox(s['center'], s['axes'], extents)
        wire = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
        wire.paint_uniform_color([0.3, 0.3, 0.3])
        vis.add_geometry(wire)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.06)
    marker.paint_uniform_color([1.0, 0.0, 0.0])
    vis.add_geometry(marker)

    # Header
    if use_gpis:
        print("\n" + "="*120)
        print(f"{'FRAME':^6} | {'SCORE':^10} | {'LL_avg':^8} | {'YAW':^7} | {'POS (X,Y)':^14} | {'N_PTS':^5} | {'STATUS'}")
        print("-"*120)
    else:
        print("\n" + "="*120)
        print(f"{'FRAME':^6} | {'LL_avg':^8} | {'LL_tot':^9} | {'YAW':^7} | {'POS (X,Y)':^14} | {'N_PTS':^5} | {'LRT':^8} | {'PASS'}")
        print("-"*120)

    frame_idx = 0

    for ts, pts, intensity in loader.stream_point_clouds():
        t0 = time.perf_counter()
        pts_w, labels, clusters = segmenter.classify_and_cluster(pts)

        colors = np.zeros((len(pts_w), 3))
        colors[labels == 0] = [0.1, 0.1, 0.3]   # Ground
        colors[labels == 1] = [0.2, 0.2, 0.2]   # Wall
        colors[labels == 2] = [0.7, 0.1, 0.1]   # Static object

        best_score = np.inf if use_gpis else -np.inf
        best_data = None
        candidate_info = []

        for i, c in enumerate(clusters):
            c_int = intensity[c['indices']] if intensity is not None else np.full(len(c['points']), 100.0)

            if use_gpis:
                score, R, t, ll_avg = registrator._fit_gpis_se2(c['points'], None)
                yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                candidate_info.append((i, score, ll_avg, yaw, t, len(c['points'])))
                if score < best_score:
                    best_score = score
                    best_data = (c['indices'], t, yaw, len(c['points']), score, ll_avg)
            else:
                R, t, ll_avg = registrator._fit_em_se2(c['points'], c_int, registrator.canonical_gmm)
                ll_total = ll_avg * len(c['points'])
                yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                log_l0 = len(c['points']) * np.log(outlier_w / robot_volume)
                lambda_lrt = 2.0 * (ll_total - log_l0)
                passed = lambda_lrt > lrt_threshold
                candidate_info.append((i, ll_avg, ll_total, yaw, t, len(c['points']), lambda_lrt, passed))
                if passed and ll_avg > best_score:
                    best_score = ll_avg
                    best_data = (c['indices'], t, yaw, len(c['points']), lambda_lrt, passed)

        # Coloring and print
        marker.translate([100, 100, 100], relative=False)
        for i, c in enumerate(clusters):
            colors[c['indices']] = [0.0, 0.8, 1.0]  # default cyan

        if best_data is not None:
            if use_gpis:
                idx, t, yaw, n_pts, score, ll_avg = best_data
                # ✅ UMBRAL CORREGIDO: score_threshold = 2.0
                accepted = score < 2.0
                colors[idx] = [0.0, 1.0, 0.0] if accepted else [0.8, 0.8, 0.0]
                marker.translate(t, relative=False)
                status = "TARGET" if accepted else "BEST(rej)"
                print(f"{frame_idx:6d} | {score:10.4f} | {ll_avg:8.2f} | {yaw:6.1f}° | ({t[0]:4.1f},{t[1]:4.1f}) | {n_pts:5d} | {status}")
            else:
                idx, t, yaw, n_pts, lrt, passed = best_data
                colors[idx] = [0.0, 1.0, 0.0] if passed else [0.8, 0.8, 0.0]
                marker.translate(t, relative=False)
                status = "TARGET" if passed else "BEST(rej)"
                print(f"{frame_idx:6d} | {best_score:8.2f} | {best_score*n_pts:9.1f} | {yaw:6.1f}° | ({t[0]:4.1f},{t[1]:4.1f}) | {n_pts:5d} | {lrt:8.1f} | {status}")
        else:
            if candidate_info:
                if use_gpis:
                    candidate_info.sort(key=lambda x: x[1])  # lowest score first
                    _, best_score, best_ll, best_yaw, best_t, best_n = candidate_info[0]
                    print(f"{frame_idx:6d} | {best_score:10.4f} | {best_ll:8.2f} | {best_yaw:6.1f}° | ({best_t[0]:4.1f},{best_t[1]:4.1f}) | {best_n:5d} | NONE")
                else:
                    candidate_info.sort(key=lambda x: x[1], reverse=True)
                    _, best_ll, ll_tot, best_yaw, best_t, best_n, _, _ = candidate_info[0]
                    print(f"{frame_idx:6d} | {best_ll:8.2f} | {ll_tot:9.1f} | {best_yaw:6.1f}° | ({best_t[0]:4.1f},{best_t[1]:4.1f}) | {best_n:5d} | {'---':^8} | NONE")

        pcd.points = o3d.utility.Vector3dVector(pts_w)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(pcd)
        vis.update_geometry(marker)
        vis.poll_events()
        vis.update_renderer()

        frame_idx += 1
        time.sleep(max(0, (1.0 / args.fps) - (time.perf_counter() - t0)))

    vis.destroy_window()


if __name__ == "__main__":
    main()