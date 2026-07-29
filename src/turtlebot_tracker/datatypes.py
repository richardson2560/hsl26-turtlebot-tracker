from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class FrameData:
    timestamp: float
    raw_points: np.ndarray
    intensity: np.ndarray
    ground_points: Optional[np.ndarray] = None
    obstacle_points: Optional[np.ndarray] = None
    R_align: np.ndarray = field(default_factory=lambda: np.eye(3))

@dataclass
class ClusterCandidate:
    id: int
    points: np.ndarray
    intensity: np.ndarray
    centroid: np.ndarray
    v_hull: float = 0.0
    solidity: float = 0.0
    rho_2p: float = 0.0
    is_corner: bool = False
    volumetricity: float = 0.0
    extent_likelihood: float = 0.0
    passed_filters: bool = False

@dataclass
class TrackingState:
    pose_se2: np.ndarray
    velocity_se2: np.ndarray
    covariance: np.ndarray
    is_zupt_active: bool = False
    surprise_triggered: bool = False
    bearing_compass_kappa: float = 1.0
    bearing_compass_mu: float = 0.0          
    trajectory_history: List[np.ndarray] = field(default_factory=list)
    nis: float = 0.0