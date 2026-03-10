"""
Processing Demo — Run preprocessing + RRT* on the campus map image.

Usage (from the Task root directory):
  cd processing_demo
  python run.py
"""

import subprocess
import sys
import os

# Ensure we're running from processing_demo/
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 55)
print("STEP 1: Preprocessing (preprocess.py)")
print("=" * 55)
subprocess.run([sys.executable, "preprocess.py"], check=True)

print("\n" + "=" * 55)
print("STEP 2: RRT* Path Planning")
print("=" * 55)
subprocess.run([
    sys.executable, "../rrt_star.py",
    "--map", "map_cropped.jpg",
    "--mask", "output/road_mask.png",
    "--start", "15,100",
    "--end", "380,188",
    "--output-dir", "output",
], check=True)

print("\nDone! Check processing_demo/output/ for results.")
