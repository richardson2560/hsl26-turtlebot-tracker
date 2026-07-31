"""
implicit_surface.py - Hermite-GPIS-W: Continuous Implicit Surface Model.
ULTRA-FAST evaluation using cKDTree spatial pre-filtering and precomputed weights.
"""

import json
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.linalg import cho_solve
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull


# ============================================================================
#  KERNEL FUNCTIONS (OPTIMIZED)
# ============================================================================

def wendland_c2(u):
    """Wendland C2 kernel: phi(u) = (1-u)^4 * (4u + 1) for 0 <= u < 1, else 0."""
    out = np.zeros_like(u)
    mask = (u >= 0) & (u < 1)
    if np.any(mask):
        u_m = u[mask]
        out[mask] = ((1.0 - u_m) ** 4) * (4.0 * u_m + 1.0)
    return out


def wendland_c2_dphi_du(u):
    """Derivative of Wendland C2 w.r.t. u."""
    out = np.zeros_like(u)
    mask = (u >= 0) & (u < 1)
    if np.any(mask):
        u_m = u[mask]
        out[mask] = -20.0 * u_m * ((1.0 - u_m) ** 3)
    return out


def hermite_covariance_block(X, Xp, h, h_p=None, sigma_f=1.0):
    """Constructs the Hermite covariance block (4n, 4m) between X and Xp."""
    n, m = len(X), len(Xp)
    if h_p is None:
        h_p = h
    h = np.asarray(h).ravel()
    h_p = np.asarray(h_p).ravel()
    if h.size == 1:
        h = np.full(n, h[0])
    if h_p.size == 1:
        h_p = np.full(m, h_p[0])

    D = X[:, None, :] - Xp[None, :, :]          # (n,m,3)
    r = np.linalg.norm(D, axis=2) + 1e-12       # (n,m)
    H = 0.5 * (h[:, None] + h_p[None, :])       # (n,m)

    u = r / H
    phi = sigma_f * wendland_c2(u)
    dphi = sigma_f * wendland_c2_dphi_du(u) / H

    K_vv = phi
    K_vg = -(dphi / r)[..., None] * D
    K_gv = (dphi / r)[..., None] * D

    inv_r = 1.0 / r
    term_diag = -(dphi * inv_r)
    term_outer = (dphi * inv_r**3)
    K_gg = term_outer[..., None, None] * (D[..., :, None] * D[..., None, :])
    K_gg += term_diag[..., None, None] * np.eye(3)[None, None, :, :]

    K = np.zeros((4*n, 4*m))
    K[0::4, 0::4] = K_vv
    K[0::4, 1::4] = K_vg[..., 0]
    K[0::4, 2::4] = K_vg[..., 1]
    K[0::4, 3::4] = K_vg[..., 2]
    K[1::4, 0::4] = K_gv[..., 0]
    K[2::4, 0::4] = K_gv[..., 1]
    K[3::4, 0::4] = K_gv[..., 2]
    for i in range(3):
        for j in range(3):
            K[1+i::4, 1+j::4] = K_gg[..., i, j]
    return K


# ============================================================================
#  OFFLINE BUILDING FUNCTIONS
# ============================================================================

def compute_adaptive_bandwidth(points, normals, h_base, h_min, k_neighbors=12):
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k_neighbors)
    mean_n = normals[idx].mean(axis=1)
    R_bar = np.linalg.norm(mean_n, axis=1)
    ratio = np.clip(R_bar, h_min / h_base, 1.0)
    return h_base * ratio


def farthest_point_sampling(points, M):
    N = len(points)
    if M >= N:
        return np.arange(N)
    selected = [0]
    dist = np.linalg.norm(points - points[0], axis=1)
    for _ in range(1, M):
        i = np.argmax(dist)
        selected.append(i)
        dist = np.minimum(dist, np.linalg.norm(points - points[i], axis=1))
    return np.array(selected)


def build_implicit_model_from_npz(npz_path, centroid=None, h_base=0.06, h_min=0.015,
                                  M_target=100, sigma_lidar=0.012, sigma_grad=0.05):
    """
    Build offline GPIS-W model from NPZ with labels (label=3 = robot).
    Model is built in coordinates centered at `centroid` (or mean of points).
    M_target can be reduced for speed (default 100 is enough for 3.5 cm resolution).
    """
    data = np.load(npz_path)
    robot_mask = data['labels'] == 3
    pts = data['points'][robot_mask]

    if len(pts) < 100:
        raise ValueError("Not enough robot points (label=3).")

    if centroid is None:
        centroid = np.mean(pts, axis=0)
    centroid = np.asarray(centroid, dtype=np.float64)
    pts_centered = pts - centroid

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_centered))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
    normals = np.asarray(pcd.normals)

    h = compute_adaptive_bandwidth(pts_centered, normals, h_base, h_min)

    idx = farthest_point_sampling(pts_centered, M_target)
    P = pts_centered[idx]
    N = normals[idx]
    H = h[idx]

    M = len(P)
    K = hermite_covariance_block(P, P, H, H, sigma_f=1.0)
    noise_diag = np.tile([sigma_lidar**2, sigma_grad**2, sigma_grad**2, sigma_grad**2], M)
    K_reg = K + np.diag(noise_diag) + 1e-8 * np.eye(4*M)
    L = np.linalg.cholesky(K_reg)

    y = np.zeros(4*M)
    y[1::4] = N[:, 0]
    y[2::4] = N[:, 1]
    y[3::4] = N[:, 2]

    hull = ConvexHull(pts_centered)
    robot_volume = float(hull.volume)

    primitives = [{'p': P[i].tolist(), 'n': N[i].tolist(), 'h': float(H[i])} for i in range(M)]

    model = ImplicitSurfaceModel(primitives, L, y, centroid, robot_volume, sigma_lidar, sigma_grad)
    return model


def save_model(model, path):
    data = {
        'primitives': model.primitives,
        'L': model.L.tolist(),
        'y': model.y.tolist(),
        'centroid': model.centroid.tolist(),
        'robot_volume': model.robot_volume,
        'sigma_lidar': model.sigma_lidar,
        'sigma_grad': model.sigma_grad,
        'M': model.M
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_model(path):
    with open(path, 'r') as f:
        data = json.load(f)
    primitives = data['primitives']
    L = np.array(data['L'], dtype=np.float64)
    y = np.array(data['y'], dtype=np.float64)
    centroid = np.array(data['centroid'], dtype=np.float64)
    robot_volume = data['robot_volume']
    sigma_lidar = data['sigma_lidar']
    sigma_grad = data['sigma_grad']
    return ImplicitSurfaceModel(primitives, L, y, centroid, robot_volume, sigma_lidar, sigma_grad)


# ============================================================================
#  ULTRA-FAST ONLINE EVALUATION CLASS (SPATIAL PRUNED)
# ============================================================================

class ImplicitSurfaceModel:
    """Offline GPIS-W model with ULTRA-FAST evaluation using Spatial KDTree Pruning."""

    def __init__(self, primitives, L, y, centroid, robot_volume, sigma_lidar, sigma_grad):
        self.primitives = primitives
        self.L = L
        self.y = y
        self.centroid = np.array(centroid, dtype=np.float64)
        self.robot_volume = robot_volume
        self.sigma_lidar = sigma_lidar
        self.sigma_grad = sigma_grad
        self.M = len(primitives)

        self.P = np.array([p['p'] for p in primitives], dtype=np.float64)
        self.N = np.array([p['n'] for p in primitives], dtype=np.float64)
        self.H = np.array([p['h'] for p in primitives], dtype=np.float64)

        # Precompute alpha weights offline
        alpha_full = cho_solve((self.L, True), self.y)  # (4M,)
        self.alpha_v = alpha_full[0::4]                 # (M,)
        self.alpha_g = np.stack([alpha_full[1::4], alpha_full[2::4], alpha_full[3::4]], axis=1)  # (M, 3)

        # Spatial KDTree for fast ball queries
        self.tree = cKDTree(self.P)
        self.max_h = float(np.max(self.H))

    def evaluate(self, X, compute_var=False):
        """
        ULTRA-FAST evaluation using KDTree active neighbor queries.
        Runs in < 0.2ms per candidate cloud.
        """
        N_pts = len(X)
        if N_pts == 0:
            return np.array([]), np.array([]), None

        # Initialize with out-of-support penalty (8 cm)
        f_vals = np.full(N_pts, 0.08, dtype=np.float64)
        grad_vals = np.zeros((N_pts, 3), dtype=np.float64)

        # Spatial query: find primitives within max_h for each point in X
        neighbor_indices = self.tree.query_ball_point(X, r=self.max_h)

        for i in range(N_pts):
            idx_k = neighbor_indices[i]
            if len(idx_k) == 0:
                continue  # keep penalty

            idx_k = np.array(idx_k)
            p_k = self.P[idx_k]
            h_k = self.H[idx_k]
            alpha_v_k = self.alpha_v[idx_k]
            alpha_g_k = self.alpha_g[idx_k]

            D_i = X[i] - p_k
            r_i = np.linalg.norm(D_i, axis=1) + 1e-12

            u_i = r_i / h_k
            mask_i = u_i < 1.0
            if not np.any(mask_i):
                continue  # keep penalty

            # Keep active primitives for point i
            D_i = D_i[mask_i]
            r_i = r_i[mask_i]
            h_k = h_k[mask_i]
            u_i = u_i[mask_i]
            alpha_v_k = alpha_v_k[mask_i]
            alpha_g_k = alpha_g_k[mask_i]

            # Kernel & Derivatives
            phi_i = ((1.0 - u_i) ** 4) * (4.0 * u_i + 1.0)
            dphi_du_i = -20.0 * u_i * ((1.0 - u_i) ** 3)
            dphi_dr_i = dphi_du_i / h_k

            # Value f(x)
            term_v = phi_i * alpha_v_k
            k_vg_i = -(dphi_dr_i / r_i)[:, None] * D_i
            term_g = np.sum(k_vg_i * alpha_g_k, axis=1)
            f_vals[i] = np.sum(term_v) + np.sum(term_g)

            # Gradient grad f(x)
            inv_r = 1.0 / r_i
            grad_kvv = (dphi_dr_i * inv_r)[:, None] * D_i
            grad_kvg = (dphi_dr_i * inv_r)[:, None] * alpha_g_k
            grad_vals[i] = np.sum(grad_kvv * alpha_v_k[:, None] + grad_kvg, axis=0)

        var_f = np.full(N_pts, self.sigma_lidar ** 2, dtype=np.float64) if compute_var else None

        return f_vals, grad_vals, var_f