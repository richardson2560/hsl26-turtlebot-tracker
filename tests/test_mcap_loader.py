"""
tests/test_mcap_loader.py - Unit test for MCAP loader folder structure resolution
"""

import pytest
from pathlib import Path
from turtlebot_tracker.io.mcap_loader import MCAPLiDARLoader

def test_mcap_loader_folder_resolution():
    bags_dir = Path("data/bags")
    if not bags_dir.exists():
        pytest.skip("data/bags directory not found")

    bag_folders = [d for d in bags_dir.iterdir() if d.is_dir()]
    if not bag_folders:
        pytest.skip("No bag folders found inside data/bags")

    loader = MCAPLiDARLoader(str(bag_folders[0]))
    frames = list(loader.stream_point_clouds())
    
    assert len(frames) > 0, "Loader failed to stream frames from bag"
    ts, pts, intensity = frames[0]
    
    assert ts > 0.0, "Invalid timestamp"
    assert pts.ndim == 2 and pts.shape[1] == 3, "Points must be 3D (Nx3)"
    assert len(intensity) == len(pts), "Intensity array length must match points"