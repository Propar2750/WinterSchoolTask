"""
RRT* Demo — Download OSM road network + run RRT* on it.

Usage (from the Task root directory):
  cd rrt_demo
  python run.py
"""

import subprocess
import sys
import os

# Ensure we're running from rrt_demo/
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Only download if map files don't exist yet
if not os.path.exists("output/osm_visual_map.png") or not os.path.exists("output/osm_road_mask.png"):
    print("=" * 55)
    print("STEP 1: Download OSM Road Network (osm_map.py)")
    print("=" * 55)
    subprocess.run([sys.executable, "osm_map.py"], check=True)
else:
    print("=" * 55)
    print("STEP 1: OSM map already exists, skipping download.")
    print("        (Delete output/osm_visual_map.png to re-download)")
    print("=" * 55)

print("\n" + "=" * 55)
print("STEP 2: RRT* Path Planning")
print("=" * 55)
subprocess.run([
    sys.executable, "../rrt_star.py",
    "--map", "output/osm_visual_map.png",
    "--mask", "output/osm_road_mask.png",
    "--output-dir", "output",
], check=True)

print("\nDone! Check rrt_demo/output/ for results.")
