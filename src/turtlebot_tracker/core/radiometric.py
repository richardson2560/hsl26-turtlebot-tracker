"""
radiometric.py - Radiometric correction and c0_SH coefficient computation.
"""

import numpy as np
import open3d as o3d


def compute_radiometric_sh_c0(points: np.ndarray, intensity: np.ndarray) -> float:
    """Computes range-corrected c0_SH coefficient for reflectance."""
    if len(points) == 0:
        return 28.2
    r_sq = np.sum(points**2, axis=1)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=15))
    normals = np.asarray(pcd.normals)
    if len(normals) == len(points):
        ray_vectors = -points / (np.sqrt(r_sq)[:, None] + 1e-6)
        cos_eta = np.abs(np.sum(ray_vectors * normals, axis=1))
        cos_eta = np.maximum(cos_eta, 0.1)
    else:
        cos_eta = np.ones(len(points)) * 0.5

    r_clamped_sq = np.minimum(r_sq, 6.25)  # 2.5m max clamp
    corrected = np.clip(intensity * (r_clamped_sq / cos_eta), 0.0, 255.0)
    return float(0.28209479177 * np.mean(corrected))