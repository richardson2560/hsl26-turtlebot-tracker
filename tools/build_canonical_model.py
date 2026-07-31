"""
build_canonical_model.py - Expert Robot Modeler with Structural Decomposition.
Integrates RBS (gap split), G-ICP (clamp all eigenvalues), VMF (absolute direction),
range noise calibration, orphan absorption, and unified radiometric correction.
"""

import argparse
import json
import sys
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
import glob

sys.path.append(str(Path(__file__).parent.parent / "src"))
from turtlebot_tracker.core.geometry import fit_upright_obb, boxes_intersect, find_gap_split
from turtlebot_tracker.core.mvi_clustering import MVIHierarchicalClustering, numpy_to_native
from turtlebot_tracker.core.hierarchical_gmm import HierarchicalGMM


# ---------- VMF parameter estimation (CORRECTED: uses absolute points) ----------
def compute_vmf_params(pts_abs, pts_centered):
    """
    pts_abs: points in absolute sensor frame (not centered)
    pts_centered: points centered at robot centroid (for shape, not used for direction)
    Returns mu_dir (direction from surface to sensor) and kappa.
    """
    if len(pts_abs) < 10:
        return [0.0, 0.0, 1.0], 0.1
    # View vectors: from each point toward sensor (origin in sensor frame)
    view_vecs = -pts_abs / (np.linalg.norm(pts_abs, axis=1, keepdims=True) + 1e-6)
    mean_v = np.mean(view_vecs, axis=0)
    r_bar = np.linalg.norm(mean_v)
    if r_bar < 1e-3:
        return [0.0, 0.0, 1.0], 0.1
    mu_dir = mean_v / r_bar
    kappa = (r_bar * (3 - r_bar**2)) / (1 - r_bar**2 + 1e-6)
    return mu_dir.tolist(), float(np.clip(kappa, 0.1, 50.0))


# ---------- G-ICP regularization (clamp ALL eigenvalues) ----------
def apply_gicp_regularization(cov, sigma_floor_sq=2.5e-5):
    vals, vecs = np.linalg.eigh(cov)
    n_degenerate = int(np.sum(vals < sigma_floor_sq))
    vals_reg = np.maximum(vals, sigma_floor_sq)
    return vecs @ np.diag(vals_reg) @ vecs.T, n_degenerate


# ---------- Range Noise Calibration (using multiple bags) ----------
def calibrate_range_noise(npz_paths, n_bins=15):
    """
    Calibrates radial noise sigma(r) from ground plane residuals.
    npz_paths: list of NPZ files (static_1m, rot_1m, rot_2m, rot_3m, etc.)
    """
    all_residuals = []
    all_ranges = []
    
    for npz_path in npz_paths:
        if not Path(npz_path).exists():
            continue
        data = np.load(npz_path)
        points = data['points']
        labels = data['labels']
        ground_mask = labels == 0
        if not np.any(ground_mask):
            continue
        ground_pts = points[ground_mask]
        if len(ground_pts) < 50:
            continue
        
        # Fit plane robustly
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ground_pts))
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=100)
        if len(inliers) < 50:
            continue
        normal = np.array(plane_model[:3])
        normal = normal / (np.linalg.norm(normal) + 1e-6)
        d = plane_model[3] / (np.linalg.norm(plane_model[:3]) + 1e-6)
        
        residuals = np.abs(ground_pts @ normal + d)
        ranges = np.linalg.norm(ground_pts, axis=1)
        
        all_residuals.append(residuals)
        all_ranges.append(ranges)
    
    if not all_residuals:
        print("[WARN] No ground points found for calibration. Using default noise floor.")
        return None
    
    residuals = np.concatenate(all_residuals)
    ranges = np.concatenate(all_ranges)
    
    # Bin by range
    bin_edges = np.percentile(ranges, np.linspace(0, 100, n_bins + 1))
    bin_centers = []
    bin_sigmas = []
    
    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i+1]
        mask = (ranges >= lo) & (ranges < hi)
        if np.sum(mask) > 30:
            bin_centers.append(0.5*(lo+hi))
            bin_sigmas.append(np.std(residuals[mask]))
    
    if len(bin_centers) < 3:
        print("[WARN] Not enough bins for calibration. Using default.")
        return None
    
    r_mid = np.array(bin_centers)
    sigma_vals = np.array(bin_sigmas)
    sigma_vals = np.maximum(sigma_vals, 0.001)
    f_sigma = interp1d(r_mid, sigma_vals, kind='linear',
                       fill_value=(sigma_vals[0], sigma_vals[-1]), bounds_error=False)
    
    calib_data = {
        "r_mid": r_mid.tolist(),
        "sigma": sigma_vals.tolist(),
        "min_r": float(r_mid[0]),
        "max_r": float(r_mid[-1])
    }
    calib_path = Path("config/range_noise_calibration.json")
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    with open(calib_path, 'w') as f:
        json.dump(calib_data, f, indent=4, default=numpy_to_native)
    print(f"[SAVED] Range noise calibration -> {calib_path}")
    return f_sigma


# ---------- Structural decomposer (with orphan handling) ----------
class StructuralDecomposer:
    def __init__(self, voxel_size, solidity_threshold, min_pts=60, min_height=0.0):
        self.voxel_size = voxel_size
        self.solidity_threshold = solidity_threshold
        self.min_pts = min_pts
        self.min_height = min_height
        self.shells = []
        self.orphans = []

    def clean_cluster(self, pts):
        if len(pts) < 10:
            return pts
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        return pts[ind] if len(ind) > 0 else pts

    def recursive_split(self, pts):
        pts = self.clean_cluster(pts)
        if len(pts) < self.min_pts:
            self.orphans.append(pts)
            return

        obb = fit_upright_obb(pts, self.voxel_size)
        if obb is None:
            self.orphans.append(pts)
            return

        if self.min_height > 0 and obb['height'] < self.min_height:
            self.orphans.append(pts)
            return

        if obb['solidity'] < self.solidity_threshold and len(pts) > self.min_pts * 2:
            axis = obb['max_axis']
            coords_local = (pts - obb['center']) @ obb['axes']
            split_val = find_gap_split(coords_local[:, axis], bin_width=0.01, min_gap_width=0.015)
            if split_val is not None:
                mask = coords_local[:, axis] > split_val
                pts_l, pts_r = pts[~mask], pts[mask]
                if len(pts_l) >= self.min_pts and len(pts_r) >= self.min_pts:
                    self.recursive_split(pts_l)
                    self.recursive_split(pts_r)
                    return
        self.shells.append(obb)

    def merge_shells(self, margin=0.05):
        merged_any = True
        while merged_any:
            merged_any = False
            new_list = []
            skip = set()
            for i in range(len(self.shells)):
                if i in skip:
                    continue
                merged = False
                for j in range(i + 1, len(self.shells)):
                    if j in skip:
                        continue
                    if boxes_intersect(self.shells[i], self.shells[j], margin=margin):
                        combined = np.vstack([self.shells[i]['pts'], self.shells[j]['pts']])
                        trial = fit_upright_obb(combined, self.voxel_size)
                        if trial is not None and trial['solidity'] >= self.solidity_threshold:
                            new_list.append(trial)
                            skip.add(j)
                            merged = True
                            merged_any = True
                            break
                if not merged:
                    new_list.append(self.shells[i])
            self.shells = new_list

    def finalize_orphans(self, margin=0.1):
        if not self.orphans or not self.shells:
            return len(self.orphans)
        absorbed_count = 0
        still_orphan = []
        for orphan_pts in self.orphans:
            if len(orphan_pts) == 0:
                continue
            o_center = np.mean(orphan_pts, axis=0)
            nearest = min(self.shells, key=lambda s: np.linalg.norm(s['center'] - o_center))
            dist = np.linalg.norm(nearest['center'] - o_center)
            max_extent = np.max(nearest['extents'])
            if dist < margin + max_extent:
                nearest['pts'] = np.vstack([nearest['pts'], orphan_pts])
                absorbed_count += len(orphan_pts)
            else:
                still_orphan.append(orphan_pts)
        self.orphans = still_orphan
        return absorbed_count


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", type=str, default="data/outputs/static_full.npz")
    parser.add_argument("--output_json", type=str, default="config/canonical_turtlebot2.json")
    parser.add_argument("--voxel", type=float, default=0.008)
    parser.add_argument("--solidity_thresh", type=float, default=0.45)
    parser.add_argument("--min_pts", type=int, default=60)
    parser.add_argument("--min_height", type=float, default=0.0)
    parser.add_argument("--mvi_volume", type=float, default=1.8)
    parser.add_argument("--max_components", type=int, default=10)
    parser.add_argument("--merge_margin", type=float, default=0.01)
    parser.add_argument("--calibrate_noise", action="store_true",
                        help="Run range noise calibration from ground planes of all NPZs")
    parser.add_argument("--extra_npz_pattern", type=str, default="data/outputs/*_full.npz",
                        help="Glob pattern for additional NPZs for noise calibration")
    args = parser.parse_args()

    # ---- Noise calibration ----
    sigma_lookup_func = None
    calib_path = Path("config/range_noise_calibration.json")
    if args.calibrate_noise or not calib_path.exists():
        print("[INFO] Calibrating range noise from ground planes...")
        npz_files = [args.input_npz]
        if args.extra_npz_pattern:
            npz_files.extend(glob.glob(args.extra_npz_pattern))
        npz_files = list(set(npz_files))
        print(f"[INFO] Using {len(npz_files)} NPZ files for calibration: {npz_files}")
        sigma_func = calibrate_range_noise(npz_files)
        if sigma_func:
            sigma_lookup_func = sigma_func
        else:
            sigma_lookup_func = lambda r: 0.005
    else:
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
        r_mid = np.array(calib_data['r_mid'])
        sigma_vals = np.array(calib_data['sigma'])
        sigma_lookup_func = interp1d(r_mid, sigma_vals, kind='linear',
                                     fill_value=(sigma_vals[0], sigma_vals[-1]), bounds_error=False)
        print(f"[INFO] Loaded range noise calibration from {calib_path}")

    # ---- Load robot data ----
    data = np.load(args.input_npz)
    robot_mask = data['labels'] == 3
    if not np.any(robot_mask):
        print("[ERROR] No robot points (label 3). Run select_canonical_candidate.py first.")
        return

    pts_raw = data['points'][robot_mask]
    int_raw = data['intensity'][robot_mask]

    # Centroid (robot center in sensor frame)
    centroid = np.mean(pts_raw, axis=0)
    pts_rel = pts_raw - centroid  # centered at robot origin
    print(f"[INFO] Robot points: {len(pts_rel):,}, centroid: {centroid}")

    # ---- Voxel downsampling ----
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_rel))
    pcd_down, _, trace = pcd.voxel_down_sample_and_trace(
        args.voxel, pcd.get_min_bound(), pcd.get_max_bound()
    )
    pts_down = np.asarray(pcd_down.points)  # centered
    int_down = np.array([np.mean(int_raw[t]) for t in trace if len(t) > 0])
    if len(int_down) != len(pts_down):
        int_down = np.full(len(pts_down), 100.0, dtype=np.float32)
    print(f"[INFO] Downsampled to {len(pts_down):,} points")

    # ---- MVI initial clustering ----
    clusterer = MVIHierarchicalClustering(
        pts_down,
        k_neighbors=12,
        volume_threshold=args.mvi_volume
    )
    islands = clusterer.cluster(max_components=args.max_components)
    print(f"[INFO] MVI produced {len(islands)} initial islands")

    # ---- Structural decomposition ----
    decomposer = StructuralDecomposer(
        voxel_size=args.voxel,
        solidity_threshold=args.solidity_thresh,
        min_pts=args.min_pts,
        min_height=args.min_height
    )

    for idxs in islands.values():
        cluster_pts = pts_down[idxs]
        if len(cluster_pts) >= args.min_pts:
            decomposer.recursive_split(cluster_pts)
        else:
            decomposer.orphans.append(cluster_pts)

    print(f"[INFO] After recursive split: {len(decomposer.shells)} shells")
    decomposer.merge_shells(margin=args.merge_margin)
    print(f"[INFO] After merge: {len(decomposer.shells)} shells")

    # ---- Absorb orphans ----
    absorbed = decomposer.finalize_orphans(margin=0.1)
    total_orphan_pts = sum(len(o) for o in decomposer.orphans) + absorbed
    print(f"[INFO] Absorbed {absorbed} orphan points. Remaining orphans: {len(decomposer.orphans)} clusters.")
    if len(decomposer.orphans) > 0:
        print(f"[WARN] {total_orphan_pts} points ({100*total_orphan_pts/len(pts_down):.1f}%) remain orphaned.")

    # ---- Generate splats ----
    tree = KDTree(pts_down)  # for intensity mapping
    gaussians = []
    shells_to_save = []
    total_pts = len(pts_down)

    for shell in decomposer.shells:
        c_pts_centered = shell['pts']   # centered at robot origin
        if len(c_pts_centered) < args.min_pts // 2:
            continue

        # ---- Absolute positions (for VMF direction) ----
        c_pts_abs = c_pts_centered + centroid

        # ---- OBB for visualization ----
        shells_to_save.append({
            "center": shell['center'].tolist(),
            "axes": shell['axes'].tolist(),
            "extents": shell['extents'].tolist(),
            "solidity": shell['solidity']
        })

        # ---- Gaussian splat ----
        mu_centered = np.mean(c_pts_centered, axis=0)
        mean_range = float(np.linalg.norm(mu_centered + centroid))  # absolute range

        sigma_floor = sigma_lookup_func(mean_range)
        sigma_floor_sq = sigma_floor ** 2

        cov_raw = np.cov(c_pts_centered.T) + np.eye(3) * 1e-7
        cov_reg, n_degenerate = apply_gicp_regularization(cov_raw, sigma_floor_sq)
        vals, vecs = np.linalg.eigh(cov_reg)
        scales = np.sqrt(np.maximum(vals, 1e-7))

        # ---- Radiometric (UNIFIED with online correction) ----
        # Compute range and incidence angle same as in registration.py
        r = np.linalg.norm(c_pts_centered, axis=1)
        r_clamped_sq = np.minimum(r**2, 6.25)          # clamp to 2.5m
        cos_eta = np.maximum(np.abs(c_pts_centered[:, 2]) / (r + 1e-6), 0.1)
        I_corr = np.clip(int_down[tree.query(c_pts_centered, k=1)[1]] * r_clamped_sq / cos_eta, 0.0, 255.0)
        sh_c0 = float(0.28209479177 * np.mean(I_corr))

        # ---- VMF (using absolute points for direction) ----
        mu_dir, kappa = compute_vmf_params(c_pts_abs, c_pts_centered)

        # ---- Classification ----
        shape_type = "bulk"
        if n_degenerate == 1: shape_type = "plate"
        elif n_degenerate == 2: shape_type = "rod"
        elif n_degenerate >= 3: shape_type = "noise"

        # ---- SKIP noise splats ----
        if shape_type == "noise":
            decomposer.orphans.append(c_pts_centered)
            continue

        gaussians.append({
            "mu": mu_centered.tolist(),
            "cov": cov_reg.tolist(),
            "scales": scales.tolist(),
            "rotation": vecs.tolist(),
            "weight": float(len(c_pts_centered) / total_pts),
            "sh_c0": sh_c0,
            "mu_dir": mu_dir,
            "kappa": kappa,
            "mean_range": mean_range,
            "shape_type": shape_type,
            "n_degenerate": n_degenerate
        })

    if not gaussians:
        print("[ERROR] No valid components generated.")
        return

    print(f"[INFO] Generated {len(gaussians)} splats and {len(shells_to_save)} shells.")
    if decomposer.orphans:
        orphan_pts = sum(len(o) for o in decomposer.orphans)
        print(f"[INFO] {orphan_pts} points discarded as noise or unassigned.")

    if len(pts_down) > 100:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts_down)
        robot_volume = float(hull.volume)
    else:
        # Fallback: sumar volúmenes de los splats (aproximación)
        robot_volume = 0.0
        for g in gaussians:
            cov = np.array(g['cov'])
            robot_volume += np.sqrt(np.linalg.det(cov)) * (4/3)*np.pi  # elipsoide
        robot_volume = max(robot_volume, 0.01)

    # ---- Save model ----
    model_data = {
        "canonical_gaussians": gaussians,
        "shells": shells_to_save,
        "metadata": {
            "centroid_offset": centroid.tolist(),
            "num_points": int(total_pts),
            "robot_volume": robot_volume,
            "algorithm": "MVI_RBS_Merge_GICP_VMF_v5",
            "range_noise_calibrated": calib_path.exists()
        }
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(model_data, f, indent=4, default=numpy_to_native)
    print(f"[SAVED] Canonical model -> {out_path}")

    # ---- Runnalls tree ----
    hg = HierarchicalGMM(gaussians)
    tree_path = Path("config/canonical_tree.json")
    hg.save(tree_path)
    print(f"[SAVED] Runnalls tree -> {tree_path}")

    print("\n[SUCCESS] Robot model built with unified radiometric correction.")


if __name__ == "__main__":
    main()