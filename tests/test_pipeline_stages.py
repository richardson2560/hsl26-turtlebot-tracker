"""
tests/test_pipeline_stages.py - Integration test for preprocessor, segmentation, and candidate filter
"""

import pytest
import numpy as np
import yaml
from turtlebot_tracker.core.preprocessor import LiDARPreprocessor
from turtlebot_tracker.core.segmentation import RangeImageSegmenter
from turtlebot_tracker.core.candidate_filter import CandidateFilter

@pytest.fixture
def config():
    with open("config/default_params.yaml", "r") as f:
        return yaml.safe_load(f)

def test_pipeline_stages_integration(config):
    np.random.seed(42)
    # Synthetic point cloud containing ground plane and a box-like obstacle
    ground = np.random.uniform([-2, -2, -0.1], [2, 2, 0.0], (500, 3))
    box = np.random.uniform([0.5, 0.5, 0.1], [0.9, 0.9, 0.5], (100, 3))
    raw_points = np.vstack([ground, box])
    intensity = np.ones(len(raw_points)) * 100.0

    preprocessor = LiDARPreprocessor(config)
    frame_data = preprocessor.process(timestamp=0.0, raw_points=raw_points, intensity=intensity)

    assert frame_data.ground_points is not None
    assert frame_data.obstacle_points is not None

    segmenter = RangeImageSegmenter(config)
    clusters = segmenter.segment(frame_data)
    
    filter_stage = CandidateFilter(config)
    candidates = filter_stage.filter_candidates(clusters)

    assert isinstance(candidates, list)