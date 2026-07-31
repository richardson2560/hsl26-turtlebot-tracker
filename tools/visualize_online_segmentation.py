"""
visualize_online_segmentation.py - Master Online Tracker Stage 1.
Probabilistic Candidate Evaluation with LL-Score reporting.
"""

import argparse
import json
import sys
import time
import traceback
import numpy as np
import open3d as o3d
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from turtlebot_tracker.io.mcap_loader import MCAPLiDARLoader
from turtlebot_tracker.core.online_segmenter import OnlineSegmenter
from turtlebot_tracker.core.registration import DirectGMMRegistrator

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
    parser.add_argument("--robot_model", type=str, default="config/canonical_turtlebot2.json")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    # 1. Initialization
    try:
        mcap_path = find_mcap_file(args.bag)
        with open(args.prior, 'r') as f: prior = json.load(f)
        with open(args.metadata, 'r') as f: metadata = json.load(f)
        
        # El registrador carga el ADN del robot
        registrator = DirectGMMRegistrator({"registration": {}}, args.robot_model)
        segmenter = OnlineSegmenter(prior, metadata)
        loader = MCAPLiDARLoader(mcap_path)
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        traceback.print_exc(); return

    # 2. Open3D Setup
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="MOCD-Lite: Multi-Candidate LL Evaluation", width=1440, height=810)
    
    # Static geometry
    for shell in prior.get('shells', []):
        box = o3d.geometry.OrientedBoundingBox(shell['center'], shell['axes'], np.array(shell['extents'])*2)
        wire = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
        wire.paint_uniform_color([0.3, 0.3, 0.3]) 
        vis.add_geometry(wire)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    
    # Pool de esferas para centros de masa
    centroid_pool = [o3d.geometry.TriangleMesh.create_sphere(radius=0.05) for _ in range(10)]
    for m in centroid_pool:
        m.paint_uniform_color([1.0, 1.0, 1.0])
        vis.add_geometry(m)

    vis.get_render_option().background_color = np.array([0.02, 0.02, 0.02])
    vis.get_render_option().point_size = 3.5

    print("\n" + "="*100)
    print(f"{'ID':^4} | {'PTS':^5} | {'LL-SCORE':^10} | {'POSE (X, Y, Z)':^25} | {'YAW':^8} | {'STATUS'}")
    print("-" * 100)

    frame_idx = 0
    try:
        for ts, pts, intensity in loader.stream_point_clouds():
            t_start = time.perf_counter()
            pts_world, labels, clusters = segmenter.classify_and_cluster(pts)

            # Ocultar marcadores
            for m in centroid_pool: m.translate([100, 100, 100], relative=False)

            best_score = -np.inf
            robot_idx = -1
            match_results = []

            # --- EVALUACIÓN DE CANDIDATOS ---
            for i, cluster in enumerate(clusters):
                # Usamos el motor EM para alinear el ADN y obtener el score
                dummy_intensity = np.full(len(cluster['points']), 100.0)
                R_est, t_est, ll = registrator._fit_em_se2(
                    cluster['points'], dummy_intensity, registrator.canonical_gmm
                )
                
                yaw_deg = np.degrees(np.arctan2(R_est[1, 0], R_est[0, 0]))
                
                # Criterio de Selección: El LL más alto que supere el umbral de ruido
                status = "CANDIDATE"
                if ll > -9.8: # Umbral de aceptación
                    if ll > best_score:
                        best_score = ll
                        robot_idx = i
                        status = "TARGET"
                
                match_results.append({
                    "score": ll, "yaw": yaw_deg, "pos": t_est, "status": status, 
                    "n_pts": len(cluster['points'])
                })

            # --- RENDER Y LOGS ---
            colors = np.zeros((len(pts_world), 3))
            colors[labels == 0] = [0.1, 0.1, 0.3] # Suelo
            colors[labels == 1] = [0.2, 0.2, 0.2] # Pared
            colors[labels == 2] = [0.8, 0.1, 0.1] # Estático conocido
            colors[labels == 3] = [0.5, 0.5, 0.1] # Ruido

            for i, cluster in enumerate(clusters):
                res = match_results[i]
                marker = centroid_pool[i % 10]
                marker.translate(res['pos'], relative=False)
                
                if i == robot_idx:
                    # EL ROBOT (Verde + Marcador Rojo)
                    colors[cluster['indices']] = [0.0, 1.0, 0.0]
                    marker.paint_uniform_color([1.0, 0.0, 0.0]) 
                    print(f"\r[{i}] ROBOT! LL:{res['score']:.2f} | Pos:({res['pos'][0]:.2f}, {res['pos'][1]:.2f}) | Yaw:{res['yaw']:.1f}°   ", end="")
                else:
                    # OTRO MOVIBLE (Cian + Marcador Blanco)
                    colors[cluster['indices']] = [0.0, 0.8, 1.0]
                    marker.paint_uniform_color([1.0, 1.0, 1.0])

            pcd.points = o3d.utility.Vector3dVector(pts_world)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(pcd)
            for m in centroid_pool: vis.update_geometry(m)
            
            if not vis.poll_events(): break
            vis.update_renderer()

            frame_idx += 1
            time.sleep(max(0, (1.0/args.fps) - (time.perf_counter() - t_start)))

    except Exception as e:
        print(f"\n[ERROR] {e}"); traceback.print_exc()

    vis.destroy_window()

if __name__ == "__main__":
    main()