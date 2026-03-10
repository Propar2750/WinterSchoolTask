"""
OSMnx Road Network Generator for IIT Kharagpur
================================================
Downloads the road network from OpenStreetMap using OSMnx,
renders it as:
  1. A visual map image (background for RRT* visualization)
  2. A binary road mask (for RRT* pathfinding)

Usage:
  python osm_map.py

Requirements:
  pip install osmnx networkx matplotlib

Output:
  osm_visual_map.png   — Colored map with roads, suitable as RRT* background
  osm_road_mask.png    — Binary mask (white = road, black = obstacle)
"""

import osmnx as ox
import numpy as np
import os
import cv2
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# ======================== CONFIGURATION ========================

# IIT Kharagpur center point and search radius
CENTER_LAT = 22.3190
CENTER_LON = 87.3100
DIST_METERS = 1800  # radius in meters from center

# Output image dimensions (pixels)
IMG_WIDTH = 800
IMG_HEIGHT = 800

# Output filenames
OUTPUT_DIR = "output"
VISUAL_MAP_PATH = os.path.join(OUTPUT_DIR, "osm_visual_map.png")
ROAD_MASK_PATH = os.path.join(OUTPUT_DIR, "osm_road_mask.png")


# ======================== HELPERS ========================

def download_road_network():
    """Download the road network graph from OpenStreetMap."""
    print("Downloading road network from OpenStreetMap...")
    print(f"  Center: ({CENTER_LAT}, {CENTER_LON}), radius: {DIST_METERS}m")

    G = ox.graph_from_point(
        center_point=(CENTER_LAT, CENTER_LON),
        dist=DIST_METERS,
        network_type="all",
        simplify=True,
    )

    # Get basic stats
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print(f"  Downloaded: {num_nodes} nodes, {num_edges} edges")

    return G


def render_visual_map(G):
    """
    Render a colored visual map using matplotlib + OSMnx.
    Saves a clean map image suitable as RRT* visualization background.
    """
    print(f"\nRendering visual map ({IMG_WIDTH}x{IMG_HEIGHT})...")

    fig, ax = ox.plot_graph(
        G,
        figsize=(IMG_WIDTH / 100, IMG_HEIGHT / 100),
        dpi=100,
        bgcolor="white",
        node_size=0,
        edge_color="#404040",
        edge_linewidth=1.5,
        show=False,
        close=False,
    )

    # Remove axis labels and ticks
    ax.set_axis_off()
    fig.tight_layout(pad=0)

    # Save to file
    fig.savefig(
        VISUAL_MAP_PATH,
        dpi=100,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="white",
    )
    plt.close(fig)

    # Resize to exact target dimensions
    img = cv2.imread(VISUAL_MAP_PATH)
    if img is not None:
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        cv2.imwrite(VISUAL_MAP_PATH, img)
        print(f"  Saved {VISUAL_MAP_PATH} ({img.shape[1]}x{img.shape[0]})")
    else:
        print(f"  Warning: Could not reload {VISUAL_MAP_PATH}")

    return img


def render_road_mask(visual_img):
    """
    Create a binary road mask by thresholding + skeletonizing the visual map.
    Roads are dark pixels on white background.
    """
    print(f"\nCreating road mask via skeletonization...")

    gray = cv2.cvtColor(visual_img, cv2.COLOR_BGR2GRAY)
    # Roads are dark on white → invert so roads become white
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    pre_pixels = cv2.countNonZero(binary)
    print(f"  Thresholded road pixels: {pre_pixels}")

    # Skeletonize
    skeleton = skeletonize(binary > 0).astype(np.uint8) * 255
    skel_pixels = cv2.countNonZero(skeleton)
    print(f"  Skeleton pixels: {skel_pixels}")

    total = visual_img.shape[0] * visual_img.shape[1]
    print(f"  Road coverage: {100 * skel_pixels / total:.1f}%")

    cv2.imwrite(ROAD_MASK_PATH, skeleton)
    print(f"  Saved {ROAD_MASK_PATH}")

    return skeleton


def verify_mask_connectivity(mask):
    """Check how many connected components the mask has."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    print(f"\nMask connectivity:")
    print(f"  Connected components: {num_labels - 1}")
    if len(areas) > 0:
        print(f"  Largest component: {areas.max()} px")
        print(f"  Smallest component: {areas.min()} px")


# ======================== MAIN ========================

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download road network
    G = download_road_network()

    # Render visual map (for RRT* background)
    visual_img = render_visual_map(G)

    # Render binary road mask by skeletonizing the visual map
    mask = render_road_mask(visual_img)

    # Check connectivity
    verify_mask_connectivity(mask)

    # Print suggested RRT* configuration
    print("\n" + "=" * 55)
    print("SUGGESTED rrt_star.py CONFIGURATION")
    print("=" * 55)
    print(f'  MAP_PATH = "{VISUAL_MAP_PATH}"')
    print(f'  ROAD_MASK_PATH = "{ROAD_MASK_PATH}"')
    print(f"  # Image size: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"  # Pick START and END points on the road network")
    print(f"  # Suggested STEP_SIZE = 5-10 (larger map -> larger steps)")
    print(f"  # Suggested GOAL_RADIUS = 15-20")
    print(f"  # Suggested REWIRE_RADIUS = 30")

    print("\nOutput files:")
    print(f"  {VISUAL_MAP_PATH}  — Visual map (RRT* background)")
    print(f"  {ROAD_MASK_PATH}    — Binary road mask (RRT* pathfinding)")


if __name__ == "__main__":
    main()
