#!/usr/bin/env python3
"""
build_implicit_model.py - Build GPIS-W offline model from static NPZ,
using robot_centroid from metadata to center the model.
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from turtlebot_tracker.core.implicit_surface import build_implicit_model_from_npz, save_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", type=str, default="data/outputs/static_full.npz")
    parser.add_argument("--metadata", type=str, default="data/outputs/static_metadata.json")
    parser.add_argument("--output_json", type=str, default="config/implicit_model.json")
    parser.add_argument("--h_base", type=float, default=0.06, help="Base bandwidth (m)")
    parser.add_argument("--h_min", type=float, default=0.015, help="Minimum bandwidth at edges (m)")
    parser.add_argument("--M", type=int, default=200, help="Number of primitives")
    parser.add_argument("--sigma_lidar", type=float, default=0.012, help="Range noise (m)")
    parser.add_argument("--sigma_grad", type=float, default=0.05, help="Normal noise (rad)")
    args = parser.parse_args()

    # Load robot centroid from metadata
    with open(args.metadata, 'r') as f:
        meta = json.load(f)
    centroid = np.array(meta['robot_centroid'], dtype=np.float64)
    print(f"[INFO] Using robot centroid: {centroid}")

    model = build_implicit_model_from_npz(
        args.input_npz,
        centroid=centroid,
        h_base=args.h_base,
        h_min=args.h_min,
        M_target=args.M,
        sigma_lidar=args.sigma_lidar,
        sigma_grad=args.sigma_grad
    )

    save_model(model, args.output_json)
    print(f"[INFO] Model saved to {args.output_json} with {model.M} primitives.")
    print(f"[INFO] Robot volume: {model.robot_volume:.4f} m^3")


if __name__ == "__main__":
    main()