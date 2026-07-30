"""
mcap_loader.py - High-Performance ROS2 MCAP PointCloud2 Loader.

Uses rosbags AnyReader for zero-dependency parsing of Livox MID-360 point clouds,
extracting 3D coordinates (XYZ) and raw infrared intensity with memory-view slicing.
"""

from pathlib import Path
from typing import Generator, Optional, Tuple, Union
import numpy as np

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


class MCAPLiDARLoader:
    """Streams point clouds from ROS2 MCAP bags with fast binary unpacking."""

    def __init__(self, target_path: Union[str, Path], topic_name: str = "/livox/lidar"):
        path = Path(target_path)
        if path.is_dir():
            mcap_files = sorted(list(path.glob("*.mcap")))
            if not mcap_files:
                raise FileNotFoundError(f"No .mcap files found in directory: {path}")
            self.mcap_path = mcap_files[0]
        else:
            self.mcap_path = path

        if not self.mcap_path.exists():
            raise FileNotFoundError(f"MCAP file not found: {self.mcap_path}")

        self.topic_name = topic_name
        self.typestore = get_typestore(Stores.ROS2_HUMBLE)

    def stream_point_clouds(self) -> Generator[Tuple[float, np.ndarray, np.ndarray], None, None]:
        """
        Streams (timestamp_sec, xyz_points, intensity_array) tuples.

        Yields:
            timestamp_sec: Message timestamp in seconds (float).
            pts: Nx3 float32 array of cartesian XYZ coordinates.
            intensity: N float32 array of raw infrared reflectance values.
        """
        with AnyReader([self.mcap_path], default_typestore=self.typestore) as reader:
            connections = [c for c in reader.connections if c.topic == self.topic_name]
            if not connections:
                connections = [c for c in reader.connections if "PointCloud2" in c.msgtype]
            if not connections:
                raise ValueError(f"No PointCloud2 streams found in {self.mcap_path}")

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                timestamp_sec = timestamp / 1e9
                pts, intensity = self._unpack_point_cloud2(msg)
                if pts is not None and len(pts) > 0:
                    yield timestamp_sec, pts, intensity

    @staticmethod
    def _unpack_point_cloud2(msg) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Unpacks PointCloud2 message buffer using zero-copy binary views."""
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            point_step = getattr(msg, "point_step", 32)
            num_points = (
                msg.width * msg.height
                if hasattr(msg, "width") and msg.width > 0
                else len(data) // point_step
            )

            offsets = {"x": 0, "y": 4, "z": 8, "intensity": 12}
            if hasattr(msg, "fields"):
                for f in msg.fields:
                    if hasattr(f, "name") and f.name in offsets:
                        offsets[f.name] = f.offset

            reshaped = data[: num_points * point_step].reshape(num_points, point_step)

            x = reshaped[:, offsets["x"] : offsets["x"] + 4].view(np.float32).flatten()
            y = reshaped[:, offsets["y"] : offsets["y"] + 4].view(np.float32).flatten()
            z = reshaped[:, offsets["z"] : offsets["z"] + 4].view(np.float32).flatten()

            if offsets["intensity"] + 4 <= point_step:
                intensity = reshaped[:, offsets["intensity"] : offsets["intensity"] + 4].view(np.float32).flatten()
            else:
                intensity = np.ones_like(x, dtype=np.float32) * 100.0

            points = np.column_stack((x, y, z))
            valid = np.isfinite(points).all(axis=1)

            return points[valid], intensity[valid]
        except Exception:
            return None, None