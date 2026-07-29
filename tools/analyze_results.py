"""
tools/analyze_results.py - Generate full statistics and plots from the output data.
Reads .csv and .npy files from data/outputs/ and produces:
  - Console table with key metrics.
  - Plots of trajectory, NIS, and latency for all bags.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2

OUTPUT_DIR = Path("data/outputs")
REPORT_DIR = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# NIS threshold for 3 DOF (95% confidence)
NIS_THRESHOLD = chi2.ppf(0.95, df=3)  # ≈ 7.81

def analyze_bag(bag_name: str, traj_path: Path, metrics_path: Path):
    """Process a single bag and return a dict of statistics."""
    stats = {}
    stats['name'] = bag_name

    # Load trajectory
    if traj_path.exists():
        traj = np.load(traj_path)
        stats['num_frames'] = len(traj)
        if len(traj) > 1:
            diffs = np.diff(traj[:, :2], axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            stats['total_distance'] = float(np.sum(distances))
        else:
            stats['total_distance'] = 0.0
    else:
        stats['num_frames'] = 0
        stats['total_distance'] = 0.0

    # Load metrics (CSV)
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            reader = csv.DictReader(f)
            latency_ms = []
            nis_vals = []
            for row in reader:
                latency_ms.append(float(row['latency_ms']))
                nis_vals.append(float(row['nis']))
            
            latency_ms = np.array(latency_ms)
            nis_vals = np.array(nis_vals)

            stats['frames_logged'] = len(latency_ms)
            stats['latency_mean'] = np.mean(latency_ms)
            stats['latency_95'] = np.percentile(latency_ms, 95)
            stats['latency_99'] = np.percentile(latency_ms, 99)
            stats['latency_max'] = np.max(latency_ms)

            stats['nis_mean'] = np.mean(nis_vals)
            stats['nis_95'] = np.percentile(nis_vals, 95)
            stats['nis_max'] = np.max(nis_vals)
            
            consistent = nis_vals < NIS_THRESHOLD
            stats['nis_consistency_ratio'] = np.mean(consistent)
            
            realtime = latency_ms < 100.0
            stats['realtime_ratio'] = np.mean(realtime)
    else:
        stats['frames_logged'] = 0
        stats['latency_mean'] = 0.0
        stats['latency_95'] = 0.0
        stats['latency_99'] = 0.0
        stats['latency_max'] = 0.0
        stats['nis_mean'] = 0.0
        stats['nis_95'] = 0.0
        stats['nis_max'] = 0.0
        stats['nis_consistency_ratio'] = 0.0
        stats['realtime_ratio'] = 0.0

    return stats

def plot_trajectories(all_stats):
    """Plot trajectories for all bags."""
    n_bags = len(all_stats)
    if n_bags == 0:
        return
    cols = 3
    rows = int(np.ceil(n_bags / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows), squeeze=False)
    axes = axes.ravel()  # Flatten to 1D array

    for i, stats in enumerate(all_stats):
        ax = axes[i]
        bag_name = stats['name']
        traj_path = OUTPUT_DIR / f"{bag_name}_trajectory.npy"
        if traj_path.exists():
            traj = np.load(traj_path)
            if len(traj) > 0:
                ax.plot(traj[:,0], traj[:,1], 'b-', linewidth=1.5, label='Trajectory')
                ax.plot(traj[0,0], traj[0,1], 'go', markersize=6, label='Start')
                ax.plot(traj[-1,0], traj[-1,1], 'ro', markersize=6, label='End')
        ax.set_title(f"{bag_name}\nDist: {stats['total_distance']:.2f} m", fontsize=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "trajectories_all.png", dpi=150)
    plt.close()

def plot_nis(all_stats):
    """Plot NIS over frames for all bags."""
    n_bags = len(all_stats)
    if n_bags == 0:
        return
    cols = 3
    rows = int(np.ceil(n_bags / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows), squeeze=False)
    axes = axes.ravel()

    for i, stats in enumerate(all_stats):
        ax = axes[i]
        bag_name = stats['name']
        metrics_path = OUTPUT_DIR / f"{bag_name}_metrics.csv"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                reader = csv.DictReader(f)
                frames = []
                nis_vals = []
                for row in reader:
                    frames.append(int(row['frame']))
                    nis_vals.append(float(row['nis']))
            if frames:
                ax.plot(frames, nis_vals, 'b-', linewidth=0.8, alpha=0.7)
                ax.axhline(y=NIS_THRESHOLD, color='r', linestyle='--', label=f'χ²(3) 95% = {NIS_THRESHOLD:.2f}')
                ax.set_title(f"{bag_name} - Consistency: {stats['nis_consistency_ratio']*100:.1f}%", fontsize=10)
                ax.set_xlabel("Frame")
                ax.set_ylabel("NIS")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                # Set y-limit to a reasonable range, but avoid if max is 0
                max_nis = stats['nis_max']
                if max_nis > 0:
                    ax.set_ylim(0, min(max_nis + 5, 100))

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "nis_all.png", dpi=150)
    plt.close()

def plot_latency(all_stats):
    """Plot latency over frames for all bags."""
    n_bags = len(all_stats)
    if n_bags == 0:
        return
    cols = 3
    rows = int(np.ceil(n_bags / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows), squeeze=False)
    axes = axes.ravel()

    for i, stats in enumerate(all_stats):
        ax = axes[i]
        bag_name = stats['name']
        metrics_path = OUTPUT_DIR / f"{bag_name}_metrics.csv"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                reader = csv.DictReader(f)
                frames = []
                latencies = []
                for row in reader:
                    frames.append(int(row['frame']))
                    latencies.append(float(row['latency_ms']))
            if frames:
                ax.plot(frames, latencies, 'b-', linewidth=0.8, alpha=0.7)
                ax.axhline(y=100, color='g', linestyle='--', label='10 Hz limit (100 ms)')
                ax.set_title(f"{bag_name} - RT frames: {stats['realtime_ratio']*100:.1f}%", fontsize=10)
                ax.set_xlabel("Frame")
                ax.set_ylabel("Latency (ms)")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "latency_all.png", dpi=150)
    plt.close()

def print_table(all_stats):
    """Print a formatted table to the console."""
    print("\n" + "=" * 120)
    print(" BAG METRICS REPORT")
    print("=" * 120)

    header = f"{'Bag':<20} | {'Frames':>7} | {'Dist(m)':>8} | {'Latency(ms)':>12} | {'NIS':>10} | {'Consist.':>8} | {'RT(≤100ms)':>10}"
    print(header)
    print("-" * 120)

    for stats in all_stats:
        print(f"{stats['name'][:20]:<20} | {stats['num_frames']:>7} | {stats['total_distance']:>8.2f} | "
              f"{stats['latency_mean']:>6.1f}±{stats['latency_95']-stats['latency_mean']:>4.1f} | "
              f"{stats['nis_mean']:>6.1f}±{stats['nis_95']-stats['nis_mean']:>4.1f} | "
              f"{stats['nis_consistency_ratio']*100:>7.1f}% | {stats['realtime_ratio']*100:>9.1f}%")

    print("=" * 120)
    print(f"* NIS consistency threshold: χ²(3) 95% = {NIS_THRESHOLD:.2f}")
    print("  - 'Consist.' = fraction of frames with NIS < threshold (well-tuned filter).")
    print("  - 'RT(≤100ms)' = fraction of frames with latency < 100 ms (real-time).")
    print(f"\n📊 Plots saved to: {REPORT_DIR}")

def main():
    all_stats = []
    csv_files = sorted(OUTPUT_DIR.glob("*_metrics.csv"))
    
    if not csv_files:
        print("[ERROR] No *_metrics.csv files found in data/outputs/")
        print("Run 'python tools/run_online_tracker.py' first.")
        return

    for csv_path in csv_files:
        bag_name = csv_path.stem.replace("_metrics", "")
        traj_path = OUTPUT_DIR / f"{bag_name}_trajectory.npy"
        stats = analyze_bag(bag_name, traj_path, csv_path)
        all_stats.append(stats)

    all_stats.sort(key=lambda x: x['name'])

    plot_trajectories(all_stats)
    plot_nis(all_stats)
    plot_latency(all_stats)
    print_table(all_stats)

if __name__ == "__main__":
    main()