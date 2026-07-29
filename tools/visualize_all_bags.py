"""
tools/visualize_all_bags.py - Open the interactive visualizer for each bag sequentially.
Useful to visually inspect segmentation, matching, and trajectory for all files.
Press ESC in each window to close and move to the next bag.
"""

import subprocess
import sys
import glob
from pathlib import Path

def main():
    bag_dirs = sorted(glob.glob("data/bags/*"))
    if not bag_dirs:
        print("[ERROR] No bag directories found in data/bags/")
        return

    print("=" * 60)
    print(" VISUAL INSPECTION FOR ALL BAGS")
    print("=" * 60)
    print("An Open3D window will open for each bag.")
    print("Navigate with LEFT/RIGHT arrows, use keys 1-7 for layer toggles.")
    print("Press ESC in the window to close it and proceed to the next bag.\n")

    for i, bag_dir in enumerate(bag_dirs):
        bag_name = Path(bag_dir).name
        print(f"\n[{i+1}/{len(bag_dirs)}] Opening: {bag_name}")
        print("   (Waiting for load... Press ESC to skip to next)")

        cmd = [
            sys.executable,
            "tools/visualize_pipeline_stages.py",
            "--bag", str(bag_dir),
            "--config", "config/default_params.yaml"
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Exiting...")
            break
        except Exception as e:
            print(f"   [ERROR] Failed to open {bag_name}: {e}")

    print("\n[DONE] Visual inspection finished.")

if __name__ == "__main__":
    main()