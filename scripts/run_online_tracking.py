import sys
import time
from pathlib import Path
import numpy as np

# Add src to Python Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from turtlebot_tracker.dataloader import MCAPLiDARLoader
from turtlebot_tracker.preprocessor import PointCloudPreprocessor
from turtlebot_tracker.clustering import ClusterGaussianFitter
from turtlebot_tracker.optimal_transport import OptimalTransportMatcher
from turtlebot_tracker.pose_estimator import RigidPoseEstimator
from turtlebot_tracker.particle_filter import SDEParticleFilter

def load_canonical_model() -> list:
    """Defines structural canonical 3D Gaussian components for Turtlebot2."""
    # Synthetic representation of 3 structural plates and central pillar
    return [
        {'mu': np.array([0.0, 0.0, 0.05]), 'scales': np.array([0.18, 0.18, 0.02]), 'weight': 0.3}, # Base Kobuki
        {'mu': np.array([0.0, 0.0, 0.22]), 'scales': np.array([0.16, 0.16, 0.02]), 'weight': 0.3}, # Middle Plate
        {'mu': np.array([0.0, 0.0, 0.42]), 'scales': np.array([0.15, 0.15, 0.03]), 'weight': 0.3}, # Top Computer
        {'mu': np.array([0.0, 0.0, 0.25]), 'scales': np.array([0.05, 0.05, 0.20]), 'weight': 0.1}  # Vertical Support
    ]

def process_bag_sequence(bag_path: str):
    print(f"\n==========================================")
    print(f" Processing Bag: {Path(bag_path).name}")
    print(f"==========================================")

    loader = MCAPLiDARLoader(bag_path)
    preprocessor = PointCloudPreprocessor(voxel_size=0.03)
    fitter = ClusterGaussianFitter(eps=0.22, min_points=15)
    ot_matcher = OptimalTransportMatcher(gamma_prune=0.3, gamma_spawn=0.3)
    pf = SDEParticleFilter(num_particles=100)

    canonical_model = load_canonical_model()
    is_initialized = False
    last_timestamp = None
    
    frame_count = 0
    total_latency_ms = []

    for timestamp, points in loader.stream_point_clouds():
        t_start = time.perf_counter()
        frame_count += 1
        
        dt = 0.1 if last_timestamp is None else max(0.01, timestamp - last_timestamp)
        last_timestamp = timestamp

        # 1. Preprocess
        obstacles, ground = preprocessor.process(points)
        if len(obstacles) < 10:
            continue

        # 2. Extract Candidate Clusters & Fit Gaussians
        candidate_clusters = fitter.extract_clusters_and_fit_gaussians(obstacles)
        if not candidate_clusters:
            continue

        # 3. Match against Canonical Model via UOT
        best_cost = float('inf')
        best_cluster = None
        best_P = None

        for cluster in candidate_clusters:
            cost, P_mat = ot_matcher.match_models(canonical_model, cluster['gaussians'])
            if cost < best_cost:
                best_cost = cost
                best_cluster = cluster
                best_P = P_mat

        if best_cluster is None:
            continue

        # 4. Extract 6-DOF Pose via SVD
        R_est, t_est = RigidPoseEstimator.estimate_pose(canonical_model, best_cluster['gaussians'], best_P)

        # 5. Filter Update (SDE Particle Filter)
        if not is_initialized:
            pf.initialize(t_est, R_est)
            is_initialized = True
        else:
            pf.predict(dt)
            pf.update(t_est, R_est, best_cost)

        smooth_t, smooth_R = pf.get_estimated_state()
        
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0
        total_latency_ms.append(latency_ms)

        if frame_count % 10 == 0:
            print(f"Frame {frame_count:04d} | Pos: [{smooth_t[0]:.2f}, {smooth_t[1]:.2f}, {smooth_t[2]:.2f}] m | OT Cost: {best_cost:.3f} | Latency: {latency_ms:.2f} ms")

    if total_latency_ms:
        print(f"\n---> Sequence Complete.")
        print(f"     Average Latency per Frame: {np.mean(total_latency_ms):.2f} ms ({1000.0/np.mean(total_latency_ms):.1f} FPS)")
        print(f"     99th Percentile Latency:   {np.percentile(total_latency_ms, 99):.2f} ms")

if __name__ == "__main__":
    import glob
    bag_files = sorted(glob.glob("data/bags/*/*.mcap"))
    if not bag_files:
        print("No .mcap files found in data/bags/*/. Place dataset files in data/bags/.")
    else:
        for bag in bag_files:
            process_bag_sequence(bag)