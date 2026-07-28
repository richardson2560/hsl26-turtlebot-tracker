"""
test_unit_math.py - Mathematical Unit Verification for Bures-Wasserstein, UOT, and Procrustes SVD.
"""

import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append(str(Path(__file__).parent.parent / "src"))
from turtlebot_tracker.optimal_transport import OptimalTransportMatcher
from turtlebot_tracker.pose_estimator import RigidPoseEstimator

def test_bures_wasserstein_zero_distance():
    g1 = {'mu': np.array([1.0, 2.0, 3.0]), 'scales': np.array([0.1, 0.2, 0.3]), 'weight': 1.0}
    ot = OptimalTransportMatcher()
    dist = ot.compute_bures_wasserstein_distance(g1, g1)
    assert np.isclose(dist, 0.0), f"Expected 0.0, got {dist}"
    print("[PASS] Bures-Wasserstein Zero Distance Test")

def test_procrustes_svd_exact_reconstruction():
    canonical = [
        {'mu': np.array([0.0, 0.0, 0.0]), 'weight': 0.5},
        {'mu': np.array([1.0, 0.0, 0.0]), 'weight': 0.5}
    ]
    
    # Apply known Rotation (90 deg around Z) and Translation
    R_true = R.from_euler('z', 90, degrees=True).as_matrix()
    t_true = np.array([2.0, -1.0, 0.5])

    observed = [
        {'mu': R_true @ canonical[0]['mu'] + t_true, 'weight': 0.5},
        {'mu': R_true @ canonical[1]['mu'] + t_true, 'weight': 0.5}
    ]

    P_mat = np.array([[0.5, 0.0], [0.0, 0.5]])  # Exact correspondence

    R_est, t_est = RigidPoseEstimator.estimate_pose(canonical, observed, P_mat)

    assert np.allclose(R_est, R_true, atol=1e-5), f"Rotation mismatch: {R_est} vs {R_true}"
    assert np.allclose(t_est, t_true, atol=1e-5), f"Translation mismatch: {t_est} vs {t_true}"
    print("[PASS] Weighted Procrustes SVD Exact Reconstruction Test")

if __name__ == "__main__":
    test_bures_wasserstein_zero_distance()
    test_procrustes_svd_exact_reconstruction()
    print("\n✅ All Mathematical Unit Tests Passed Successfully!")