"""
tools/build_hierarchical_tree.py - Build hierarchical tree from canonical splats.
Usage: python tools/build_hierarchical_tree.py
"""

import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from turtlebot_tracker.core.hierarchical_gmm import HierarchicalGMM

def main():
    splat_path = Path("config/canonical_turtlebot2.json")
    tree_path = Path("config/canonical_tree.json")

    if not splat_path.exists():
        print("[ERROR] Run tools/build_canonical_model.py first.")
        return

    with open(splat_path, 'r') as f:
        data = json.load(f)

    gmm_list = data.get("canonical_gaussians", [])
    if len(gmm_list) < 2:
        print("[WARNING] Only 1 component. No tree needed.")
        return

    print(f"[INFO] Building hierarchical tree from {len(gmm_list)} components...")
    hg = HierarchicalGMM(gmm_list)
    hg.save(tree_path)
    print(f"[SAVED] Tree -> {tree_path}")

if __name__ == "__main__":
    main()