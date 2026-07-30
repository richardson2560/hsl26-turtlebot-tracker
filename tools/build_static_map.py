"""
build_static_map.py - Expert Hierarchical World Modeler.
Segments structure into Solid Bounded Boxes and candidates into Splats.
"""

import argparse
import json
import sys
import yaml
import numpy as np
import open3d as o3d
from pathlib import Path

# Repository imports
sys.path.append(str(Path(__file__).parent.parent / "src"))
from turtlebot_tracker.core.mvi_clustering import MVIHierarchicalClustering, numpy_to_native
from turtlebot_tracker.core.geometry import fit_upright_obb, boxes_intersect

class StructuralDecomposer:
    def __init__(self, voxel_size, solidity_threshold, min_pts=50, min_height=0.15):
        self.voxel_size = voxel_size
        self.solidity_threshold = solidity_threshold
        self.min_pts = min_pts
        self.min_height = min_height
        self.shells = []

    def clean_cluster(self, pts):
        """Removes noise before fitting boxes."""
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        return pts[ind]

    def recursive_split(self, pts):
        """Top-Down Split: Breaks boxes at corners/junctions if they contain empty space."""
        pts = self.clean_cluster(pts)
        if len(pts) < self.min_pts:
            return

        obb = fit_upright_obb(pts, self.voxel_size)
        if obb is None: return

        # VETO: Discard floor debris/noise
        if obb['height'] < self.min_height:
            return

        # Check if box is 'Hollow' (likely a V, L, or Z shape)
        if obb['solidity'] < self.solidity_threshold and len(pts) > self.min_pts * 2:
            axis = obb['max_axis']
            coords_local = (pts - obb['center']) @ obb['axes']
            split_val = np.median(coords_local[:, axis])
            
            mask = coords_local[:, axis] > split_val
            pts_l, pts_r = pts[~mask], pts[mask]
            
            if len(pts_l) >= self.min_pts and len(pts_r) >= self.min_pts:
                self.recursive_split(pts_l)
                self.recursive_split(pts_r)
            else:
                self.shells.append(obb)
        else:
            self.shells.append(obb)

    def merge_shells(self):
        """Bottom-Up Merge: Reunites fragments of long straight walls."""
        merged_any = True
        while merged_any:
            merged_any = False
            new_list = []
            skip = set()
            for i in range(len(self.shells)):
                if i in skip: continue
                merged = False
                for j in range(i + 1, len(self.shells)):
                    if j in skip: continue
                    if boxes_intersect(self.shells[i], self.shells[j]):
                        combined = np.vstack([self.shells[i]['pts'], self.shells[j]['pts']])
                        trial = fit_upright_obb(combined, self.voxel_size)
                        if trial['solidity'] >= self.solidity_threshold:
                            new_list.append(trial)
                            skip.add(j)
                            merged = True
                            merged_any = True
                            break
                if not merged:
                    new_list.append(self.shells[i])
            self.shells = new_list

def main():
    parser = argparse.ArgumentParser(description="Expert Static Map Builder")
    parser.add_argument("--input_npz", type=str, default="data/outputs/static_full.npz")
    parser.add_argument("--output_json", type=str, default="config/static_map_prior.json")
    parser.add_argument("--solidity", type=float, default=0.35, help="Stricter = more splits at corners")
    parser.add_argument("--voxel", type=float, default=0.04)
    args = parser.parse_args()

    # Load Data
    data = np.load(args.input_npz)
    points, labels = data['points'], data['labels']
    with open("data/outputs/static_metadata.json", 'r') as f: meta = json.load(f)

    # 1. STRUCTURE (OBB Shells)
    struct_pts = points[labels == 1]
    pts_down = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(struct_pts)).voxel_down_sample(args.voxel).points)

    print(f"[INFO] Processing {len(pts_down)} structural points...")
    mvi_islands = MVIHierarchicalClustering(pts_down, k_neighbors=12, volume_threshold=2.2).cluster(max_components=30)
    
    decomposer = StructuralDecomposer(args.voxel, args.solidity)
    for idxs in mvi_islands.values():
        decomposer.recursive_split(pts_down[idxs])
    
    print(f"[INFO] Split into {len(decomposer.shells)} fragments. Starting merge...")
    decomposer.merge_shells()

    # 2. STATIC CANDIDATES (Splats)
    cand_pts = points[labels == 2]
    cand_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cand_pts)).voxel_down_sample(0.02)
    cand_pts_down = np.asarray(cand_pcd.points)
    
    static_splats = []
    if len(cand_pts_down) > 40:
        cand_mvi = MVIHierarchicalClustering(cand_pts_down, k_neighbors=10, volume_threshold=1.5).cluster(max_components=12)
        for idxs in cand_mvi.values():
            pts = cand_pts_down[idxs]
            if len(pts) < 30: continue
            mu = np.mean(pts, axis=0)
            if np.max(pts[:, 2]) - np.min(pts[:, 2]) < 0.1: continue # Reject flat ground artifacts

            cov = np.cov(pts.T) + np.eye(3) * 1e-6
            vals, vecs = np.linalg.eigh(cov)
            static_splats.append({
                "mu": mu.tolist(), "cov": cov.tolist(),
                "scales": np.sqrt(np.maximum(vals, 1e-7)).tolist(), "rotation": vecs.tolist()
            })

    # 3. EXPORT
    shells_to_save = []
    debug_pts, debug_lbls = [], []
    for i, shell in enumerate(decomposer.shells):
        shells_to_save.append({
            "center": shell['center'].tolist(),
            "axes": shell['axes'].tolist(),
            "extents": shell['extents'].tolist(),
            "solidity": shell['solidity']
        })
        debug_pts.extend(shell['pts'].tolist())
        debug_lbls.extend([i] * len(shell['pts']))

    prior = {
        "alignment": meta['R_align'],
        "ground": {"z": meta['z_ground'], "thickness": meta['ground_thickness']},
        "shells": shells_to_save,
        "static_splats": static_splats,
        "debug_points": debug_pts,
        "debug_labels": debug_lbls
    }

    with open(args.output_json, "w") as f:
        json.dump(prior, f, indent=4, default=numpy_to_native)
    
    print(f"[SUCCESS] Environment Built: {len(shells_to_save)} Shells, {len(static_splats)} Splats.")

if __name__ == "__main__":
    main()