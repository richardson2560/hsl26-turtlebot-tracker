"""
analyze_results.py - Diagnostic Analysis, NIS Consistency and Latency Reporter.

Reads CSV metrics and NPY trajectories from data/outputs/ and produces:
  - Terminal table report with latency percentiles, NIS consistency, and FSM breakdown.
  - Diagnostic report plots in data/outputs/reports/: trajectories, NIS vs chi^2_3 limit, latency vs 100ms.
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

OUTPUT_DIR = Path("data/outputs")
REPORT_DIR = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# NIS 95% threshold for 3 DOF
NIS_CHI2_95 = chi2.ppf(0.95, df=3)  # Exact = 7.8147


def analyze_bag_metrics(bag_name: str, csv_path: Path, traj_path: Path) -> dict:
    stats = {'name': bag_name}

    if traj_path.exists():
        traj = np.load(traj_path)
        stats['num_frames'] = len(traj)
        if len(traj) > 1:
            diffs = np.diff(traj[:, :2], axis=0)
            stats['total_distance'] = float(np.sum(np.linalg.norm(diffs, axis=1)))
        else:
            stats['total_distance'] = 0.0
    else:
        stats['num_frames'] = 0
        stats['total_distance'] = 0.0

    if csv_path.exists():
        latencies, nis_vals, fsm_states = [], [], []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                latencies.append(float(r['latency_total_ms']))
                nis_vals.append(float(r['nis']))
                fsm_states.append(int(r['fsm_state']))

        latencies = np.array(latencies)
        nis_vals = np.array(nis_vals)
        fsm_states = np.array(fsm_states)

        stats['latency_mean'] = np.mean(latencies)
        stats['latency_p95'] = np.percentile(latencies, 95)
        stats['latency_p99'] = np.percentile(latencies, 99)
        stats['realtime_ratio'] = np.mean(latencies <= 100.0) * 100.0

        stats['nis_mean'] = np.mean(nis_vals)
        stats['nis_p95'] = np.percentile(nis_vals, 95)
        stats['nis_consistency'] = np.mean(nis_vals <= NIS_CHI2_95) * 100.0

        stats['active_ratio'] = np.mean(fsm_states == 2) * 100.0
        stats['coasting_ratio'] = np.mean(fsm_states == 3) * 100.0
    else:
        stats['latency_mean'] = stats['latency_p95'] = stats['latency_p99'] = 0.0
        stats['realtime_ratio'] = stats['nis_mean'] = stats['nis_p95'] = 0.0
        stats['nis_consistency'] = stats['active_ratio'] = stats['coasting_ratio'] = 0.0

    return stats


def generate_report_plots(all_stats: list) -> None:
    n_bags = len(all_stats)
    if n_bags == 0:
        return

    cols = 3
    rows = int(np.ceil(n_bags / cols))

    # 1. Trajectories Plot
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), squeeze=False)
    axes = axes.ravel()
    for i, st in enumerate(all_stats):
        ax = axes[i]
        traj_path = OUTPUT_DIR / f"{st['name']}_trajectory.npy"
        if traj_path.exists():
            traj = np.load(traj_path)
            if len(traj) > 0:
                ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1.5, label='SE(2) Path')
                ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=6, label='Start')
                ax.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=6, label='End')
        ax.set_title(f"{st['name']}\nDist: {st['total_distance']:.2f} m", fontsize=9)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "trajectories_all.png", dpi=150)
    plt.close()

    # 2. NIS Plot
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), squeeze=False)
    axes = axes.ravel()
    for i, st in enumerate(all_stats):
        ax = axes[i]
        csv_path = OUTPUT_DIR / f"{st['name']}_metrics.csv"
        if csv_path.exists():
            frames, nis_vals = [], []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    frames.append(int(r['frame']))
                    nis_vals.append(float(r['nis']))
            if frames:
                ax.plot(frames, nis_vals, 'b-', linewidth=0.8, alpha=0.7)
                ax.axhline(y=NIS_CHI2_95, color='r', linestyle='--', label=f'χ²_3(0.95) = {NIS_CHI2_95:.2f}')
                ax.set_title(f"{st['name']} - Consistency: {st['nis_consistency']:.1f}%", fontsize=9)
                ax.set_xlabel("Frame")
                ax.set_ylabel("NIS")
                ax.set_ylim(0, max(15.0, min(np.max(nis_vals) + 2, 50.0)))
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "nis_all.png", dpi=150)
    plt.close()

    # 3. Latency Plot
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), squeeze=False)
    axes = axes.ravel()
    for i, st in enumerate(all_stats):
        ax = axes[i]
        csv_path = OUTPUT_DIR / f"{st['name']}_metrics.csv"
        if csv_path.exists():
            frames, latencies = [], []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    frames.append(int(r['frame']))
                    latencies.append(float(r['latency_total_ms']))
            if frames:
                ax.plot(frames, latencies, 'g-', linewidth=0.8, alpha=0.7)
                ax.axhline(y=100.0, color='r', linestyle='--', label='10 Hz Limit (100 ms)')
                ax.set_title(f"{st['name']} - Realtime: {st['realtime_ratio']:.1f}%", fontsize=9)
                ax.set_xlabel("Frame")
                ax.set_ylabel("Latency (ms)")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "latency_all.png", dpi=150)
    plt.close()


def print_table_report(all_stats: list) -> None:
    print("\n" + "=" * 115)
    print(" TURTLEBOT_TRACKER MOCD-LITE/SE(2) PERFORMANCE BENCHMARK REPORT")
    print("=" * 115)
    header = f"{'Bag Name':<20} | {'Frames':>6} | {'Dist(m)':>7} | {'Lat p50(ms)':>11} | {'Lat p95':>9} | {'NIS Consist.':>11} | {'RT (<=100ms)':>11} | {'Active %':>8}"
    print(header)
    print("-" * 115)

    for st in all_stats:
        print(f"{st['name'][:20]:<20} | {st['num_frames']:>6d} | {st['total_distance']:>7.2f} | "
              f"{st['latency_mean']:>11.1f} | {st['latency_p95']:>9.1f} | "
              f"{st['nis_consistency']:>10.1f}% | {st['realtime_ratio']:>10.1f}% | "
              f"{st['active_ratio']:>7.1f}%")

    print("=" * 115)
    print(f"📊 Diagnostic report plots generated in: {REPORT_DIR}\n")


def main():
    all_stats = []
    csv_files = sorted(OUTPUT_DIR.glob("*_metrics.csv"))

    if not csv_files:
        print("[ERROR] No metrics CSV files found in data/outputs/")
        print("Run 'python tools/run_online_tracker.py' first.")
        return

    for csv_p in csv_files:
        bag_name = csv_p.stem.replace("_metrics", "")
        traj_p = OUTPUT_DIR / f"{bag_name}_trajectory.npy"
        st = analyze_bag_metrics(bag_name, csv_p, traj_p)
        all_stats.append(st)

    generate_report_plots(all_stats)
    print_table_report(all_stats)


if __name__ == "__main__":
    main()