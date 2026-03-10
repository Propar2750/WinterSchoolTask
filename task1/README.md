# Task 1: RRT* Path Planning on IIT Kharagpur Campus Map

## Overview

This project implements the **RRT\* (Rapidly-exploring Random Tree Star)** algorithm to find an optimal path between two points on the IIT Kharagpur campus road network. It includes two demos: one that extracts roads from a real campus map image via computer vision, and another that downloads roads from OpenStreetMap. Both feed into a single unified RRT\* planner that works directly on skeleton (1px-wide) road masks.

---

## Project Structure

```
task1/
├── rrt_star.py                  # Unified RRT* planner (argparse CLI, skeleton-aware)
├── README.md                    # This file
├── approach.md                  # Detailed approach document
├── initial_attempts.md          # Log of failed preprocessing approaches
├── references.txt               # Reference material
├── map.jpg                      # Original full campus map
│
├── processing_demo/             # Demo 1: Real image → road extraction → RRT*
│   ├── preprocess.py            # Multi-stage preprocessing pipeline
│   ├── map_cropped.jpg          # Cropped campus map (416×237 px)
│   ├── run.py                   # Runs preprocess + RRT* end-to-end
│   └── output/                  # All generated outputs
│       ├── stage1_color_threshold.png
│       ├── stage2_cleaned.png
│       ├── stage2b_denoised.png
│       ├── stage3_gap_filled.png
│       ├── stage4_skeleton.png
│       ├── stage4_final_road.png
│       ├── road_mask.png        # Final skeleton mask for RRT*
│       ├── pipeline_comparison.png
│       ├── rrt_star_result.jpg  # Final path image
│       └── rrt_star_output.mp4  # Tree growth video
│
└── rrt_demo/                    # Demo 2: OpenStreetMap → skeleton → RRT*
    ├── osm_map.py               # Downloads roads via OSMnx, skeletonizes
    ├── run.py                   # Runs osm_map + RRT* end-to-end
    ├── cache/                   # OSMnx download cache
    └── output/
        ├── osm_visual_map.png   # Rendered visual map (800×800)
        ├── osm_road_mask.png    # Skeleton road mask
        ├── rrt_star_result.jpg
        └── rrt_star_output.mp4
```

---

## Requirements

```
pip install opencv-python numpy scipy scikit-image osmnx networkx matplotlib
```

- Python 3.10+
- `osmnx` is only required for the `rrt_demo`

---

## Quick Start

### Processing Demo (real campus image)

```bash
cd task1/processing_demo
python run.py
```

Or step by step:
```bash
cd task1/processing_demo
python preprocess.py                          # generates output/road_mask.png
python ../rrt_star.py --map map_cropped.jpg --mask output/road_mask.png --start 15,100 --end 380,188 --output-dir output
```

### RRT\* Demo (OpenStreetMap)

```bash
cd task1/rrt_demo
python run.py
```

Or step by step:
```bash
cd task1/rrt_demo
python osm_map.py                             # downloads + skeletonizes roads
python ../rrt_star.py --map output/osm_visual_map.png \
    --mask output/osm_road_mask.png \
    --output-dir output
```

> `run.py` in `rrt_demo/` skips the download if the map files already exist.

---

## How It Works

### 1. Preprocessing Pipeline (`processing_demo/preprocess.py`)

Converts a campus map image into a clean 1px-wide skeleton road mask:

| Stage | Name | Description |
|-------|------|-------------|
| **1** | HSV Color Threshold | Isolates blue-ish road lines using HSV bounds `(80,10,50)` to `(150,255,255)` |
| **2** | Small Component Removal | Removes connected components < 18 px (noise dots) |
| **2b** | Gaussian Denoise | 3×3 Gaussian blur + re-threshold — smooths noise while preserving thin roads |
| **3** | HoughLinesP Gap Fill | Detects line segments and bridges gaps between disconnected road fragments |
| **4** | Skeletonization | Reduces all roads to 1px-wide center lines using morphological thinning |

Each stage saves its output image for inspection. A `pipeline_comparison.png` shows all stages side by side.

**Configuration** (top of `preprocess.py`):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `HSV_LOWER` / `HSV_UPPER` | `(80,10,50)` / `(150,255,255)` | HSV range for road color detection |
| `MIN_COMPONENT_AREA` | `18` | Minimum area to keep a component |
| `HOUGH_THRESHOLD` | `12` | Min votes for a HoughLinesP line |
| `HOUGH_MIN_LINE_LENGTH` | `10` | Min line length (pixels) |
| `HOUGH_MAX_LINE_GAP` | `5` | Max gap to bridge between line segments |

### 2. OSM Map Generator (`rrt_demo/osm_map.py`)

Downloads the IIT KGP road network from OpenStreetMap via `osmnx.graph_from_point()`, renders an 800×800 visual map, then creates a skeleton mask by thresholding + skeletonizing the rendered image.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `CENTER_LAT` / `CENTER_LON` | `22.319` / `87.31` | IIT KGP map center |
| `DIST_METERS` | `1800` | Download radius (meters) |
| `IMG_WIDTH` / `IMG_HEIGHT` | `800` / `800` | Output image dimensions |

### 3. RRT\* Path Planner (`rrt_star.py`)

A unified, skeleton-aware RRT\* implementation that works on any map + mask combination.

#### Key Features

- **Skeleton-aware steering**: After computing a step toward the sampled point, the new node is snapped to the nearest road pixel via a KDTree over all road pixels. This ensures the tree grows exactly along the skeleton.
- **Dilated collision checking**: A 2-iteration dilation of the mask is used internally for collision checks, so straight-line connections between nearby skeleton pixels aren't rejected for passing through 1px gaps.
- **Informed sampling**: Only samples from road pixels (not uniform random across the image).
- **KDTree acceleration**: $O(\log n)$ nearest-neighbor queries via `scipy.spatial.KDTree`, rebuilt periodically as the tree grows.
- **Rewiring**: Continuously improves existing paths by checking if rerouting through new nodes reduces cost.
- **Goal bias**: Probabilistically steers toward the goal for faster convergence.
- **Auto endpoints**: If `--start` / `--end` are omitted, picks two maximally distant road pixels.
- **Snap-to-road**: Start and end points are snapped to the nearest road pixel if off-road.

#### Usage

```
python rrt_star.py --map MAP_IMAGE [--mask MASK | --threshold VALUE]
                   [--start X,Y] [--end X,Y] [--output-dir DIR]
                   [--step-size N] [--goal-radius N] [--max-iter N]
                   [--rewire-radius N] [--goal-bias F]
                   [--fps N] [--draw-every N]
```

#### Mask Input Modes

| Mode | Flag | Description |
|------|------|-------------|
| Pre-generated mask | `--mask path.png` | Loads a binary mask (white = road) |
| Binary threshold | `--threshold 200` | Thresholds the map image (dark roads on light background → inverted) |
| Default | *(neither)* | Falls back to `--threshold 200` |

#### Parameters

All RRT\* parameters are **auto-computed from image size** (`dim = max(width, height)`) so the same script works for any map without manual tuning. Override any value via CLI flags.

| Parameter | Auto-Computed Default | Description |
|-----------|----------------------|-------------|
| `--map` | *(required)* | Background map image for visualization |
| `--mask` | — | Pre-generated binary road mask (skeleton) |
| `--threshold` | `200` | Threshold for generating mask from map |
| `--start` | auto | Start point as `X,Y` |
| `--end` | auto | End point as `X,Y` |
| `--step-size` | `max(3, round(dim × 0.005))` | Max pixel distance per tree extension step |
| `--goal-radius` | `max(8, round(dim × 0.02))` | Distance (px) to consider goal reached |
| `--max-iter` | `max(50000, dim × 100)` | Maximum RRT\* iterations |
| `--rewire-radius` | `max(15, round(dim × 0.025))` | Radius (px) for neighborhood rewiring |
| `--goal-bias` | `0.05` | Probability of sampling the goal directly |
| `--output-dir` | `.` | Directory for output files |
| `--fps` | `60` | Video frame rate |
| `--draw-every` | `5` | Write a video frame every N iterations |

#### Output

| File | Description |
|------|-------------|
| `rrt_star_result.jpg` | Final frame — path overlay on the map (saved even if goal not reached) |
| `rrt_star_output.mp4` | Video showing the tree growing and the final path |

---

## Algorithm Details

### RRT\* (Rapidly-exploring Random Tree Star)

RRT\* is a sampling-based motion planning algorithm that builds a tree rooted at the start point, growing toward the goal while continuously optimizing path cost.

**Per-iteration loop:**

1. **Sample** a random road pixel (or goal with probability `goal_bias`)
2. **Find nearest** tree node to the sample using KDTree
3. **Steer** from nearest toward sample by `step_size` pixels, then **snap** to nearest road pixel
4. **Collision check** the edge from nearest → new node on the dilated mask
5. **Choose best parent** among nearby nodes (within `rewire_radius`) that minimizes cost
6. **Rewire** — check if rerouting any nearby nodes through the new node reduces their cost
7. If new node is within `goal_radius` of the goal, record the path (keep improving until `max_iter`)

**Complexity:** Each iteration is $O(\log n)$ for nearest-neighbor and $O(k)$ for rewiring ($k$ = nodes within `rewire_radius`).

### Skeleton-Aware Adaptations

Standard RRT\* assumes open 2D space. Our road masks are 1px-wide skeletons, which requires:

- **Snap-to-road steering**: Without snapping, a steered point will almost always land off a 1px line. By snapping to the nearest road pixel (via a pre-built KDTree over all road pixels), every tree node sits exactly on the skeleton.
- **Dilated collision mask**: The straight line between two skeleton pixels that are both on a curving road may briefly pass through a non-road pixel. A 2-iteration dilation creates a ~5px corridor around the skeleton, allowing these short connections.

---

## Design Decisions

1. **Unified `rrt_star.py`** — A single argparse-based script serves both demos with no code duplication.
2. **Skeleton masks** — Skeletonization reduces road masks to 1px-wide lines, giving a cleaner topology with no redundant parallel paths. RRT\* is adapted to work directly on these thin masks.
3. **Gap fill before skeleton** — HoughLinesP runs on the thresholded mask (before skeletonization) to bridge disconnected road fragments, then skeletonization thins the result.
4. **Gaussian blur over morphological opening** — Opening (erosion → dilation) was too aggressive for 1–3 px wide roads. Gaussian blur + re-threshold removes isolated noise while preserving thin structures (see [initial_attempts.md](initial_attempts.md)).
5. **Informed sampling** — Sampling only from road pixels (not the full image) dramatically improves convergence since the tree never wastes iterations on obstacle space.
6. **Always save result** — `rrt_star_result.jpg` is written even when the goal isn't reached, showing the tree's current state with a failure message.
