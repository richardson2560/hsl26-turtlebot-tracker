"""
tests/test_ekf_consistency.py - NIS statistical consistency test for SE(2) Manifold EKF
"""

import pytest
import numpy as np
import yaml
from scipy.stats import chi2
from turtlebot_tracker.core.tracking import SE2ManifoldEKF

def test_ekf_nis_consistency():
    with open("config/default_params.yaml", "r") as f:
        config = yaml.safe_load(f)

    ekf = SE2ManifoldEKF(config)
    
    # Extract measurement noise standard deviation matching configured R matrix
    r_std = np.sqrt(np.diag(ekf.R))

    nis_values = []
    np.random.seed(42)

    for _ in range(100):
        ekf.predict(dt=0.1)
        # Add Gaussian measurement noise matching configured R covariance
        z_meas = ekf.get_state().pose_se2 + np.random.normal(0, r_std)
        nis = ekf.update(z_meas)
        nis_values.append(nis)

    # Check that NIS values fall within 95% chi-square bounds (df = 3)
    dof = 3
    lower_bound = chi2.ppf(0.025, df=dof)
    upper_bound = chi2.ppf(0.975, df=dof)

    in_bounds = np.sum((np.array(nis_values) >= lower_bound) & (np.array(nis_values) <= upper_bound))
    ratio = in_bounds / len(nis_values)

    assert ratio >= 0.85, f"EKF NIS ratio {ratio:.2f} is below acceptable threshold (expected >= 0.85)"