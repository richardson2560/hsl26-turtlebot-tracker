"""
datatypes.py - Strongly Typed Dataclasses and Enum Definitions for turtlebot_tracker.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional
import numpy as np


class LifecycleState(IntEnum):
    """Bayesian Lifecycle FSM States."""
    UNINITIALIZED = 0
    SEARCHING_MAP = 1
    ACTIVE_TRACKING = 2
    COASTING_LOST = 3


class SemanticLabel(IntEnum):
    """4-Class Point Cloud Semantic Segmentation Labels."""
    GROUND = 0
    STRUCTURE_WALL = 1
    CANDIDATE_FREE = 2
    ROBOT_TARGET = 3


@dataclass
class StaticMapPrimitives:
    """Parametric background map M_bg containing planar primitives."""
    normals: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    distances: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    is_initialized: bool = False


@dataclass
class FrameData:
    timestamp: float
    raw_points: np.ndarray
    intensity: np.ndarray
    ground_points: Optional[np.ndarray] = None
    obstacle_points: Optional[np.ndarray] = None
    R_align: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    z_ground: float = 0.0
    semantic_labels: Optional[np.ndarray] = None


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
    is_arc_valid: bool = False
    volumetricity: float = 0.0
    extent_likelihood: float = 0.0
    passed_filters: bool = False
    map_score: float = -999.0


@dataclass
class TrackingState:
    pose_se2: np.ndarray              # [x, y, psi]
    velocity_se2: np.ndarray          # [vx, vy, omega]
    covariance: np.ndarray            # P matrix 6x6
    z: float = 0.00                   # Absolute height (z_ground + h_base)
    lifecycle_state: LifecycleState = LifecycleState.SEARCHING_MAP
    is_zupt_active: bool = False
    surprise_triggered: bool = False
    coasting_time: float = 0.0
    bearing_compass_kappa: float = 1.0
    bearing_compass_mu: float = 0.0
    trajectory_history: List[np.ndarray] = field(default_factory=list)
    z_log: List[float] = field(default_factory=list)
    nis: float = 0.0