# Approach Document — RRT* Path Planning on IIT Kharagpur Map

## Problem Statement

**Task 1 from WS25 Computer Vision Tasks:**  
Implement the RRT\* (Rapidly-exploring Random Tree Star) algorithm to find an optimal path between two designated points on the IIT Kharagpur campus map. The tree's exploration process must be visualized in real-time on the map image. Pre-processing with OpenCV is required to convert the map into a traversable format (identifying roads vs. obstacles).

---

## High-Level Approach

The problem breaks down into two sub-problems:

1. **Map → Binary Road Mask**: Convert a campus map image into a binary mask where white pixels represent traversable road and black pixels represent obstacles.
2. **Road Mask → Optimal Path**: Run skeleton-aware RRT\* on the mask to grow a tree along roads and find the shortest path between two points.

Two independent pipelines feed into a single unified RRT\* planner:

| Pipeline | Input | Mask Generation Method |
|----------|-------|----------------------|
| **Processing Demo** | Cropped campus map photo (`map_cropped.jpg`, 416×237 px) | Multi-stage computer vision preprocessing |
| **RRT\* Demo** | OpenStreetMap road network via OSMnx | Render roads → threshold → skeletonize |

Both produce a **1px-wide skeleton mask** that the same `rrt_star.py` consumes.

---

## Pipeline 1: Real Image Preprocessing

### Challenge

The campus map image has blue-ish road lines on a colored background with labels, buildings, greenery, and other visual clutter. Roads are only 1–3 pixels wide in many places. The goal is to isolate just the road network as a clean binary mask.

### Pipeline Stages

The final pipeline has 4 stages (plus a denoising sub-stage), arrived at through iterative experimentation:

#### Stage 1 — HSV Color Thresholding

- **Why HSV?** Road lines are blue-ish, which is hard to isolate in RGB but falls into a tight hue range in HSV. HSV separates color (hue) from brightness (value), making it able to detect faint lines as well.
- **Bounds chosen:** H ∈ [80, 150], S ≥ 10, V ≥ 50. This covers the blue-to-cyan range in OpenCV's 0–179 hue scale while excluding very dark or desaturated pixels.
- **Output:** A rough binary mask with road pixels plus some noise.

#### Stage 2 — Small Connected Component Removal

- **Method:** `cv2.connectedComponentsWithStats` with 8-connectivity. Any component with area < 18 pixels is removed.
- **Rationale:** After thresholding, the mask contains stray dots and single-pixel noise from map artifacts. These are disconnected from the road network and have very small area. Removing components below a threshold cleans them while preserving all road structures (which are contiguous and larger).

#### Stage 2b — Gaussian Blur Denoising

- **Method:** 3×3 Gaussian blur followed by re-thresholding at 127.
- **Why not morphological opening?** Opening (erosion → dilation) was the first approach tried, but it was too aggressive — it eroded thin 1–3 px road lines, fragmenting the network from ~23 components to 231 and destroying connectivity (see `initial_attempts.md`).
- **Why Gaussian works:** Unlike opening, Gaussian blur merely averages pixel values. An isolated white pixel surrounded by black gets averaged below 127 and disappears. But connected road lines have enough neighboring white pixels that they survive the threshold. This makes it ideal for thin structures.

#### Stage 3 — HoughLinesP Gap Filling

- **Problem:** Even after cleaning, roads have small gaps due to thresholding artifacts or map features (labels crossing roads, color inconsistencies).
- **Method:** Run Canny edge detection on the mask, then `cv2.HoughLinesP` to detect line segments. Detected lines are drawn back onto the mask.
- **Key parameter:** `maxLineGap=5` — the Hough transform bridges gaps up to 5 pixels between line segments, reconnecting broken road fragments.
- **Why before skeletonization:** Gap filling operates on the thresholded mask (roads are a few pixels wide) where Hough can detect clear line segments. If done after skeletonization, the 1px lines produce poor Hough detections. Order was validated experimentally — the reverse order failed.

#### Stage 4 — Skeletonization

- **Method:** Zhang-Suen morphological thinning via `skimage.morphology.skeletonize`.
- **Purpose:** Reduces all roads to 1px-wide center lines. This gives a clean topology with no redundant parallel paths, dramatically fewer road pixels (faster RRT\*), and consistent representation regardless of original line thickness.
- **No post-dilation:** The skeleton is output as-is. RRT\* handles the thin mask internally via dilated collision checking (see below).

### Pipeline Order Rationale

```
Threshold → Clean → Denoise → Gap Fill → Skeletonize
```

Each stage depends on the previous one's output characteristics:
- Cleaning removes small noise so denoising can focus on subtle artifacts.
- Gap filling runs before skeletonization because Hough needs multi-pixel-wide lines to detect segments reliably.
- Skeletonization is the final step, producing the cleanest possible 1px road network for RRT\*.

---

## Pipeline 2: OpenStreetMap Road Network

### Approach

For the RRT\* demo on a larger map, roads are obtained directly from OpenStreetMap rather than extracted via image processing:

1. **Download:** `osmnx.graph_from_point()` downloads all roads within 1800m of IIT KGP's center (22.319°N, 87.31°E).
2. **Render:** OSMnx/matplotlib renders the road graph as dark lines on a white background (800×800 px). This serves as the visual map.
3. **Mask generation:** The rendered image is converted to grayscale, thresholded (roads are dark → invert), then skeletonized to produce a 1px-wide road mask.

No morphological cleanup is needed since the rendered image is clean (computer-generated, no noise).

---

## RRT\* Algorithm

### Standard RRT\* Overview

RRT\* is a sampling-based motion planning algorithm that builds a space-filling tree from a start point toward a goal. Unlike basic RRT, it continuously optimizes path cost through rewiring — checking if rerouting existing nodes through new nodes reduces their total cost.

**Per-iteration loop:**

1. **Sample** a random road pixel (or the goal with probability `goal_bias`)
2. **Find nearest** tree node to the sample via KDTree — $O(\log n)$
3. **Steer** from nearest node toward sample by `step_size` pixels
4. **Collision check** the edge (nearest → new node) against the mask
5. **Choose best parent** among nearby nodes within `rewire_radius`
6. **Rewire** — reroute nearby nodes through the new node if cheaper
7. If new node is within `goal_radius` of the goal, record path and continue optimizing

### Skeleton-Aware Adaptations

Standard RRT\* assumes open 2D space with scattered obstacles. Our road masks are 1px-wide skeleton lines — effectively the navigable space is a thin graph embedded in 2D. This required two key adaptations:

#### 1. Snap-to-Road Steering

**Problem:** When RRT\* steers from a node toward a sampled point by `step_size` pixels, the resulting point almost never lands exactly on a 1px skeleton line. The node falls into obstacle space and is rejected, wasting iterations.

**Solution:** After computing the steered point, snap it to the nearest road pixel using a KDTree built over all road pixel coordinates.

```
steered_point = nearest + step_size * direction
snapped_point = road_pixels_kd.query(steered_point)  → nearest road pixel
```

This ensures every tree node sits exactly on the skeleton, and the tree grows along road paths.

#### 2. Dilated Collision Mask

**Problem:** Even with snap-to-road, two nodes that are both on the skeleton may have a straight-line connection that briefly passes through a non-road pixel (e.g., on a curve where the line cuts through the inside of the curve). The collision check would reject valid connections.

**Solution:** Create a dilated version of the road mask (2 iterations with a 3×3 kernel, producing ~5px corridors around the skeleton) and use it exclusively for collision checking.

```python
collision_mask = cv2.dilate(road_mask, np.ones((3,3)), iterations=2)
```

- **Tree sampling & node placement**: Uses the original skeleton mask (nodes are on exact 1px roads).
- **Collision checking, parent selection, rewiring**: Uses the dilated mask (short straight-line connections between nearby skeleton pixels are accepted).

### Optimizations

| Optimization | Benefit |
|-------------|---------|
| **Informed sampling** | Only samples from road pixels (not the full image). The tree never wastes iterations on obstacle space. |
| **KDTree for nearest-neighbor** | $O(\log n)$ instead of $O(n)$ brute-force. Rebuilt periodically as the tree grows. |
| **KDTree for snap-to-road** | Pre-built over all road pixels for $O(\log m)$ snapping (m = number of road pixels). |
| **Goal bias** | 5–8% of samples are directed at the goal, steering the tree toward it without sacrificing exploration. |
| **Auto endpoint selection** | If no start/end specified, picks two maximally distant road pixels from a random sample. |
| **Snap-to-road for endpoints** | Start and end points are snapped to the nearest road pixel if off-road. |

### Visualization

- **Video output:** Every N iterations, a frame is written showing the tree growth overlaid on the map image. The final path is held for 3 seconds.
- **Result image:** Always saved — on success, shows the path in red with a cost label; on failure, shows the tree with a red error message.

---

## Key Challenges & Solutions

### 1. Thin Road Lines (1–3 px) Break Morphological Operations

- **Challenge:** Standard denoising techniques like morphological opening erode thin structures.
- **Solution:** Gaussian blur + re-threshold for denoising. This averages pixel values without structural erosion, preserving thin roads while removing isolated noise.

### 2. Disconnected Road Fragments

- **Challenge:** HSV thresholding produces gaps in roads where map labels or color inconsistencies occur.
- **Solution:** HoughLinesP detects line segments and draws them back to bridge gaps. The `maxLineGap` parameter controls the maximum gap that can be bridged.

### 3. RRT\* on 1px Skeleton Masks

- **Challenge:** Standard RRT\* with step sizes > 1 constantly falls off 1px roads. Nodes land in obstacle space and are rejected, making convergence extremely slow or impossible.
- **Solution:** Snap-to-road steering (KDTree over road pixels) + dilated collision mask. Nodes are placed on exact road pixels; collision checks use a widened mask that tolerates short off-road connections between nearby road pixels.

### 4. Pipeline Stage Ordering

- **Challenge:** Gap filling after skeletonization failed because HoughLinesP cannot reliably detect 1px-wide lines.
- **Solution:** Gap fill runs on the multi-pixel-wide thresholded mask, then skeletonization thins everything uniformly. Validated by comparing both orderings experimentally.

---

## Configuration & Parameters

### Preprocessing Parameters (tuned for `map_cropped.jpg`)

| Parameter | Value | Tuning Rationale |
|-----------|-------|-----------------|
| HSV range | (80,10,50)–(150,255,255) | Covers the blue-ish road color on the IIT KGP map |
| Min component area | 18 px | Small enough to keep thin road segments, large enough to remove noise dots |
| Hough threshold | 12 | Low enough to detect faint lines |
| Hough min line length | 10 px | Filters out very short fragments |
| Hough max line gap | 5 px | Bridges small gaps without connecting unrelated roads |

### RRT\* Parameters (Auto-Computed from Image Size)

All RRT\* parameters are auto-computed from `dim = max(width, height)` so the same formula works for any map size. Users can still override individual values via CLI flags.

| Parameter | Formula | 416px image | 800px image |
|-----------|---------|-------------|-------------|
| Step size | `max(3, round(dim × 0.005))` | 3 | 4 |
| Goal radius | `max(8, round(dim × 0.02))` | 8 | 16 |
| Rewire radius | `max(15, round(dim × 0.025))` | 15 | 20 |
| Max iterations | `max(50000, dim × 100)` | 50,000 | 80,000 |
| Goal bias | `0.05` (constant) | 0.05 | 0.05 |

---

## Results

| Metric | Processing Demo | RRT\* Demo |
|--------|----------------|-----------|
| Road mask coverage | ~7.5% of image | ~4.1% of image |
| Goal reached | Yes (iter ~12,475) | Yes (iter ~35,221) |
| Final path cost | ~437.2 px | ~1236.0 px |
| Tree nodes at completion | ~50,000 | ~80,000 |

Both demos produce:
- `rrt_star_result.jpg` — final path overlaid on the map
- `rrt_star_output.mp4` — video of tree exploration and path discovery

---

## Tools & Libraries

| Library | Purpose |
|---------|---------|
| OpenCV (`cv2`) | HSV thresholding, connected components, HoughLinesP, Gaussian blur, dilation, video I/O |
| NumPy | Array operations, pixel coordinate manipulation |
| SciPy (`KDTree`) | $O(\log n)$ nearest-neighbor for tree nodes and snap-to-road |
| scikit-image (`skeletonize`) | Zhang-Suen morphological thinning to 1px lines |
| OSMnx | OpenStreetMap road network download |
| matplotlib | Road network rendering for OSM demo |
| argparse | Unified CLI for the RRT\* planner |

---

## References

- [Robotic Path Planning: RRT and RRT\*](https://theclassytim.medium.com/robotic-path-planning-rrt-and-rrt-212319121378)
