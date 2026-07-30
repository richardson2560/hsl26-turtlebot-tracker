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