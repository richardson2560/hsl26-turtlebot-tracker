"""
datatypes.py - Core data structures for turtlebot_tracker.
"""

from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import List, Optional, Tuple, Union
import numpy as np


class SemanticLabel(IntEnum):
    GROUND = 0
    STRUCTURE_WALL = 1
    CANDIDATE_FREE = 2
    TARGET = 3
    UNKNOWN = -1


class LifecycleState(Enum):
    SEARCHING_MAP = "SEARCHING_MAP"
    ACTIVE_TRACKING = "ACTIVE_TRACKING"
    COASTING_LOST = "COASTING_LOST"


@dataclass
class StaticMapPrimitives:
    """Static background primitives (ground + walls)."""
    normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    distances: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    is_initialized: bool = False


@dataclass
class ClusterCandidate:
    """A candidate cluster from segmentation."""
    id: int
    points: np.ndarray
    intensity: np.ndarray
    centroid: np.ndarray
    semantic_label: int = SemanticLabel.CANDIDATE_FREE
    passed_filters: bool = False
    v_hull: float = 0.0
    solidity: float = 0.0
    is_corner: bool = False
    is_arc_valid: bool = False
    rho_2p: float = 0.0
    volumetricity: float = 0.0
    extent_likelihood: float = -999.0
    map_score: float = -999.0
    kinematic_penalty: float = 0.0
    effective_score: float = 0.0
    accepted: bool = False


@dataclass
class FrameData:
    """Processed frame data."""
    timestamp: float
    raw_points: np.ndarray
    intensity: np.ndarray
    ground_points: Optional[np.ndarray] = None
    obstacle_points: Optional[np.ndarray] = None
    R_align: Optional[np.ndarray] = None
    z_ground: float = 0.0
    semantic_labels: Optional[np.ndarray] = None
    clusters: List[ClusterCandidate] = field(default_factory=list)


@dataclass
class TrackingState:
    """State of the EKF tracker."""
    pose_se2: np.ndarray              # [x, y, yaw]
    velocity_se2: np.ndarray          # [vx, vy, omega]
    covariance: np.ndarray            # 6x6 covariance matrix
    z: float = 0.0                    # Absolute height
    lifecycle_state: LifecycleState = LifecycleState.SEARCHING_MAP
    is_zupt_active: bool = False
    surprise_triggered: bool = False
    coasting_time: float = 0.0        # deprecated, kept for compatibility
    coasting_frames: int = 0          # nuevo: contador de frames en coasting
    bearing_compass_kappa: float = 3.0
    bearing_compass_mu: float = 0.0
    trajectory_history: List[np.ndarray] = field(default_factory=list)
    z_log: List[float] = field(default_factory=list)
    nis: float = 0.0
    reliability_score: float = 0.0