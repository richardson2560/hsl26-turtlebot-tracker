from pathlib import Path
from typing import Generator, Tuple, Optional
import numpy as np

# Recommended high-level API for rosbags >= 0.9.15
USE_ANYREADER = False
try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    USE_ANYREADER = True
except ImportError:
    pass

# Fallback imports for older rosbags versions
try:
    from rosbags.rosbag2 import Reader
except ImportError:
    Reader = None

deserialize_cdr = None
try:
    from rosbags.serde import deserialize_cdr
except ImportError:
    try:
        from rosbags.serde.cdr import deserialize_cdr
    except ImportError:
        pass


class MCAPLiDARLoader:
    """Reads PointCloud2 messages from ROS2 MCAP bag files without ROS environment dependencies."""
    
    def __init__(self, bag_path: str, topic_name: str = "/livox/lidar"):
        self.bag_path = Path(bag_path)
        self.topic_name = topic_name
        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag file not found at: {self.bag_path}")

    def stream_point_clouds(self) -> Generator[Tuple[float, np.ndarray], None, None]:
        """Streams (timestamp_sec, point_cloud_xyz) tuples from the MCAP file."""
        if USE_ANYREADER:
            yield from self._stream_with_anyreader()
        else:
            yield from self._stream_with_reader()

    def _stream_with_anyreader(self) -> Generator[Tuple[float, np.ndarray], None, None]:
        """Reads MCAP using AnyReader (rosbags >= 0.9.15)."""
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        with AnyReader([self.bag_path], default_typestore=typestore) as reader:
            connections = [c for c in reader.connections if c.topic == self.topic_name]
            if not connections:
                connections = [c for c in reader.connections if 'PointCloud2' in c.msgtype]
            
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                timestamp_sec = timestamp / 1e9
                pcd_np = self._unpack_point_cloud2(msg)
                if pcd_np is not None and len(pcd_np) > 0:
                    yield timestamp_sec, pcd_np

    def _stream_with_reader(self) -> Generator[Tuple[float, np.ndarray], None, None]:
        """Fallback for older rosbags releases."""
        if Reader is None:
            raise ImportError("rosbags is not installed properly in your Conda environment.")
        
        with Reader(self.bag_path) as reader:
            connections = [c for c in reader.connections if c.topic == self.topic_name]
            if not connections:
                connections = [c for c in reader.connections if 'PointCloud2' in c.msgtype]

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = deserialize_cdr(rawdata, connection.msgtype) if deserialize_cdr else rawdata
                timestamp_sec = timestamp / 1e9
                pcd_np = self._unpack_point_cloud2(msg)
                if pcd_np is not None and len(pcd_np) > 0:
                    yield timestamp_sec, pcd_np

    @staticmethod
    def _unpack_point_cloud2(msg) -> Optional[np.ndarray]:
        """Unpacks Nx3 float32 XYZ coordinates dynamically from PointCloud2 object or raw bytes."""
        try:
            # Handle raw byte buffers if message was not deserialized
            if isinstance(msg, (bytes, bytearray, memoryview, np.ndarray)):
                data_bytes = np.frombuffer(msg, dtype=np.uint8)
                num_floats = len(data_bytes) // 4
                floats = np.frombuffer(data_bytes[:num_floats*4], dtype=np.float32)
                if len(floats) >= 3:
                    step_floats = 8 if len(floats) % 8 == 0 else 4
                    num_pts = len(floats) // step_floats
                    pts = floats[:num_pts * step_floats].reshape(num_pts, step_floats)[:, :3]
                    valid = np.isfinite(pts).all(axis=1) & (np.abs(pts).sum(axis=1) > 0)
                    return pts[valid]
                return None

            # Handle standard rosbags deserialized PointCloud2 object
            if not hasattr(msg, 'data') or len(msg.data) == 0:
                return None

            offsets = {'x': 0, 'y': 4, 'z': 8}
            if hasattr(msg, 'fields'):
                for f in msg.fields:
                    if hasattr(f, 'name') and f.name in offsets:
                        offsets[f.name] = f.offset

            point_step = getattr(msg, 'point_step', 32)
            width = getattr(msg, 'width', len(msg.data) // point_step)
            height = getattr(msg, 'height', 1)
            num_points = width * height

            data_bytes = np.frombuffer(msg.data, dtype=np.uint8)
            if len(data_bytes) < num_points * point_step:
                num_points = len(data_bytes) // point_step

            if num_points == 0:
                return None

            reshaped = data_bytes[:num_points * point_step].reshape(num_points, point_step)

            x = reshaped[:, offsets['x']:offsets['x']+4].view(np.float32).flatten()
            y = reshaped[:, offsets['y']:offsets['y']+4].view(np.float32).flatten()
            z = reshaped[:, offsets['z']:offsets['z']+4].view(np.float32).flatten()

            points = np.column_stack((x, y, z))
            valid_mask = np.isfinite(points).all(axis=1)
            return points[valid_mask]
        except Exception as e:
            return None