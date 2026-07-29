"""
tests/test_unit_math.py - Unit tests for mathematical invariants (Rodrigues, SE2, ConvexHull)
"""

import numpy as np
from turtlebot_tracker.core.preprocessor import LiDARPreprocessor

def test_rodrigues_z_up_alignment():
    # Tilted ground normal vector
    tilted_normal = np.array([0.3, 0.0, 0.9539])
    tilted_normal = tilted_normal / np.linalg.norm(tilted_normal)

    # Compute rotation using Rodrigues alignment
    v = np.cross(tilted_normal, np.array([0.0, 0.0, 1.0]))
    s = np.linalg.norm(v)
    c = np.dot(tilted_normal, np.array([0.0, 0.0, 1.0]))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R_align = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s ** 2))

    aligned_normal = R_align @ tilted_normal
    np.testing.assert_allclose(aligned_normal, [0.0, 0.0, 1.0], atol=1e-5)