# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TRS Winter School Computer Vision project with two independent tasks:
- **Task 1:** RRT* path planning on IIT Kharagpur campus map (road extraction from images + pathfinding)
- **Task 2:** Background removal via K-Means clustering (unsupervised segmentation + compositing)

## Running the Tasks

### Task 1: Image Processing Demo
```bash
cd task1/processing_demo
python run.py
```
Or manually: `python preprocess.py` then `python ../rrt_star.py --map map_cropped.jpg --mask output/road_mask.png --start 15,100 --end 380,188 --output-dir output`

### Task 1: OpenStreetMap Demo
```bash
cd task1/rrt_demo
python run.py
```
First run downloads OSM data (requires internet).

### Task 2: Background Removal
```bash
cd task2
python kmeans_bg_removal.py
```

## Dependencies

```bash
pip install opencv-python numpy scipy scikit-image osmnx networkx matplotlib pillow
```

Python 3.10+. Virtual environment in `.venv/`. `osmnx` only needed for `task1/rrt_demo/`.

## Architecture

### Task 1: RRT* Path Planning

Two demo pipelines share a unified `task1/rrt_star.py` planner:

1. **Processing demo** (`task1/processing_demo/`): `preprocess.py` runs a 5-stage pipeline (HSV thresholding → component removal → Gaussian denoising → HoughLinesP gap filling → skeletonization) to extract a 1px-wide road mask from map images. `run.py` chains preprocessing into RRT*.

2. **OSM demo** (`task1/rrt_demo/`): `osm_map.py` downloads road networks via OSMnx and renders them as binary masks. `run.py` chains download into RRT*.

3. **rrt_star.py** accepts any binary road mask. Key adaptations for skeleton-thin roads:
   - Skeleton-aware steering: snaps samples to nearest road pixel via KDTree
   - Dilated collision mask (2-iteration) for bridging 1px gaps
   - Informed sampling from road pixels only (not uniform random)
   - Auto-computed parameters based on image dimensions (step_size, goal_radius, rewire_radius, max_iter)

### Task 2: K-Means Background Removal

Single script `task2/kmeans_bg_removal.py`: LAB/HSV color clustering → border-dominance heuristic for background detection → morphological cleanup (close/open) → Gaussian feathering → alpha-blended compositing with replacement background.

## Key Design Decisions

- All parameters hardcoded at top of each script (no config files)
- Scripts resolve paths relative to their own location, not CWD
- Gaussian blur chosen over morphological opening for preprocessing (better preserves thin road structures)
- Both RRT* demos produce MP4 video output showing tree growth frame-by-frame
- LaTeX report in `documentation.tex` / `documentation.pdf`
