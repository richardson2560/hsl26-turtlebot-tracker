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
try:
    from turtlebot_tracker.core.registration import GPISRegistrator
except ImportError:
    GPISRegistrator = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=str, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--model", type=str, default="config/implicit_model.json",
                        help="GPIS model JSON (only used with --registrator gpis)")
    args = parser.parse_args()

    # Load components
    with open("config/static_map_prior.json", 'r') as f:
        prior = json.load(f)
    with open("data/outputs/static_metadata.json", 'r') as f:
        meta = json.load(f)

    # --- Instantiate registrator ---
    if GPISRegistrator is None:
        print("[ERROR] GPISRegistrator not available. Install or use --registrator gmm.")
        return
    registrator = GPISRegistrator({"registration": {"score_threshold": 2.0}}, args.model)
    use_gpis = True
    print(f"[INFO] Using GPIS registrator with model {args.model}")

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

            score, R, t, ll_avg = registrator._fit_gpis_se2(c['points'], None)
            yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            candidate_info.append((i, score, ll_avg, yaw, t, len(c['points'])))
            if score < best_score:
                best_score = score
                best_data = (c['indices'], t, yaw, len(c['points']), score, ll_avg)

        # Coloring and print
        marker.translate([100, 100, 100], relative=False)
        for i, c in enumerate(clusters):
            colors[c['indices']] = [0.0, 0.8, 1.0]  # default cyan

        if best_data is not None:
            idx, t, yaw, n_pts, score, ll_avg = best_data
            # ✅ UMBRAL CORREGIDO: score_threshold = 2.0
            accepted = score < 2.0
            colors[idx] = [0.0, 1.0, 0.0] if accepted else [0.8, 0.8, 0.0]
            marker.translate(t, relative=False)
            status = "TARGET" if accepted else "BEST(rej)"
            print(f"{frame_idx:6d} | {score:10.4f} | {ll_avg:8.2f} | {yaw:6.1f}° | ({t[0]:4.1f},{t[1]:4.1f}) | {n_pts:5d} | {status}")
        else:
            if candidate_info:
                candidate_info.sort(key=lambda x: x[1])  # lowest score first
                _, best_score, best_ll, best_yaw, best_t, best_n = candidate_info[0]
                print(f"{frame_idx:6d} | {best_score:10.4f} | {best_ll:8.2f} | {best_yaw:6.1f}° | ({best_t[0]:4.1f},{best_t[1]:4.1f}) | {best_n:5d} | NONE")
                
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