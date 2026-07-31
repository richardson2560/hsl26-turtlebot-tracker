"""
geometry.py - Expert Geometric Kernels for Structural Decomposition.
Provides PCA-based Upright OBB fitting, Solidity metrics, and intersection logic.
"""

import numpy as np

def fit_upright_obb(pts, voxel_size: float):
    """
    Fits an Oriented Bounding Box (OBB) constrained to the Z-axis (Upright).
    
    Args:
        pts: (N, 3) array of points.
        voxel_size: Size of the voxel used for volume estimation.
        
    Returns:
        Dictionary with center, axes, extents, solidity, height, and point count.
    """
    if len(pts) < 10:
        return None

    mu = np.mean(pts, axis=0)
    
    # Orientation: We only care about XY variance to find the wall heading.
    # This prevents noise on the top/bottom of the wall from tilting the box.
    pts_xy = pts[:, :2] - mu[:2]
    cov_xy = np.cov(pts_xy.T)
    vals, vecs = np.linalg.eigh(cov_xy)
    
    # Construct 3D rotation matrix: XY from PCA, Z forced to [0,0,1]
    R_3d = np.eye(3)
    R_3d[0:2, 0:2] = vecs 
    
    # Project points to local coordinate system to find actual bounds
    coords = (pts - mu) @ R_3d
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    
    # Calculate extents (half-lengths) and the true geometric center
    extent = (maxs - mins) / 2.0
    center = mu + R_3d @ ((maxs + mins) / 2.0)
    
    # Solidity: Ratio of occupied volume (points) vs container volume (box)
    box_vol = (maxs[0]-mins[0]) * (maxs[1]-mins[1]) * (maxs[2]-mins[2])
    occupied_vol = len(pts) * (voxel_size ** 3)
    solidity = occupied_vol / (box_vol + 1e-8)
    
    # Height: Vertical extent for noise rejection
    height = maxs[2] - mins[2]
    
    return {
        "center": center, 
        "axes": R_3d, 
        "extents": extent,
        "solidity": float(solidity),
        "height": float(height),
        "pts": pts, 
        "max_axis": int(np.argmax(extent))
    }

def boxes_intersect(box1, box2, margin=0.10):
    """
    Checks if two OBBs are close enough to be considered for merging.
    Uses a centroid-distance heuristic based on box extents.
    """
    dist = np.linalg.norm(box1['center'] - box2['center'])
    # Heuristic: if distance between centers < sum of max dimensions + margin
    threshold = np.max(box1['extents']) + np.max(box2['extents']) + margin
    return dist < threshold

def find_gap_split(coords_1d, bin_width=0.008, min_gap_width=0.015, search_pctile=(0.15, 0.85)):
    """
    Encuentra el mayor hueco real de densidad en el rango central.
    Reemplaza el corte por mediana con un corte basado en el valle físico de la nube.
    Retorna el centro del hueco si existe y supera min_gap_width, sino retorna None.
    """
    if len(coords_1d) < 10:
        return None
    lo, hi = np.percentile(coords_1d, [search_pctile[0]*100, search_pctile[1]*100])
    # Si el rango útil es muy pequeño, no hay hueco significativo
    if hi - lo < bin_width * 3:
        return None
    
    bins = np.arange(lo, hi + bin_width, bin_width)
    hist, edges = np.histogram(coords_1d, bins=bins)
    
    max_gap_width = 0.0
    best_center = None
    
    i = 0
    while i < len(hist):
        if hist[i] == 0:
            j = i
            while j < len(hist) and hist[j] == 0:
                j += 1
            gap_start = edges[i]
            gap_end = edges[j]
            gap_width = gap_end - gap_start
            if gap_width >= min_gap_width and gap_width > max_gap_width:
                max_gap_width = gap_width
                best_center = 0.5 * (gap_start + gap_end)
            i = j
        else:
            i += 1
    return best_center