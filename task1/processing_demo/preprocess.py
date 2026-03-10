"""
Road Mask Preprocessing Pipeline for IIT Kharagpur Map
======================================================
Converts the campus map into a clean binary road mask through:
  1. HSV color thresholding (isolate blue-ish road lines)
  2. Small component removal (connectedComponentsWithStats)
  3. First skeletonization (thin roads to 1-pixel center lines)
  4. Predictive line filling (HoughLinesP to bridge gaps in skeleton)
  5. Second skeletonization (re-thin after gap filling)
  6. Light dilation (restore traversability on the skeleton)

Each stage saves its output image for visual inspection.
"""

import cv2
import numpy as np
import os
from skimage.morphology import skeletonize

# Ensure paths resolve relative to this script, not the caller's cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================== CONFIGURATION ========================
MAP_PATH = os.path.join(_SCRIPT_DIR, "map_cropped.jpg")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output")

# HSV thresholds for detecting blue-ish road lines
HSV_LOWER = (80, 10, 50)
HSV_UPPER = (150, 255, 255)

# Minimum connected component area (pixels) — removes stray dots/noise
MIN_COMPONENT_AREA = 18

# HoughLinesP parameters for gap filling
HOUGH_THRESHOLD = 12
HOUGH_MIN_LINE_LENGTH = 10
HOUGH_MAX_LINE_GAP = 5

# Dilation before skeletonization (helps merge close parallel lines)
PRE_SKELETON_DILATION = 0  # iterations with 3x3 kernel

# Dilation after skeletonization (no longer needed — RRT* handles this internally)
POST_SKELETON_DILATION = 0  # iterations with 3x3 kernel

KERNEL = np.ones((3, 3), np.uint8)


# ======================== PIPELINE STAGES ========================

def stage1_color_threshold(img):
    """
    Stage 1: HSV Color Thresholding
    --------------------------------
    The road lines on the IIT KGP map are blue-ish in color.
    We convert to HSV and apply inRange to isolate them.
    
    HSV ranges chosen:
      - Hue: 80-150 (covers blue to blue-cyan range in OpenCV's 0-179 scale)
      - Saturation: > 50 (even faint blue lines have some saturation)
      - Value: > 50 (exclude very dark pixels)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    pixel_count = cv2.countNonZero(mask)
    total = img.shape[0] * img.shape[1]
    print(f"[Stage 1] Color threshold: {pixel_count} road pixels "
          f"({100 * pixel_count / total:.1f}% of image)")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "stage1_color_threshold.png"), mask)
    return mask


def stage2_remove_small_components(mask):
    """
    Stage 2: Remove Small Connected Components
    --------------------------------------------
    Use cv2.connectedComponentsWithStats to identify all connected regions.
    Remove any component with area < MIN_COMPONENT_AREA pixels.
    
    This eliminates stray dots, single-pixel noise, and small artifacts 
    that are not part of the road network.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    # Label 0 is background — skip it
    areas = stats[1:, cv2.CC_STAT_AREA]
    print(f"[Stage 2] Found {num_labels - 1} connected components")
    print(f"          Area range: {areas.min()} - {areas.max()}, "
          f"median: {int(np.median(areas))}")

    # Build cleaned mask: keep only components with area >= threshold
    cleaned = np.zeros_like(mask)
    kept = 0
    removed = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            cleaned[labels == i] = 255
            kept += 1
        else:
            removed += 1

    pixel_count = cv2.countNonZero(cleaned)
    print(f"          Kept {kept} components, removed {removed} "
          f"(< {MIN_COMPONENT_AREA} px)")
    print(f"          Remaining: {pixel_count} road pixels")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "stage2_cleaned.png"), cleaned)
    return cleaned


def stage2b_denoise(mask):
    """
    Stage 2b: Gaussian Blur + Re-threshold Denoising
    --------------------------------------------------
    Apply a 3x3 Gaussian blur to smooth out isolated noise pixels,
    then re-threshold to get a clean binary mask.
    
    This is gentler than morphological opening — it softens noise
    without eroding thin road lines.
    """
    blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    _, denoised = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    before = cv2.countNonZero(mask)
    after = cv2.countNonZero(denoised)
    print(f"[Stage 2b] Gaussian blur (3x3) + threshold denoise:")
    print(f"           {before} -> {after} pixels "
          f"(removed {before - after} noise pixels)")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "stage2b_denoised.png"), denoised)
    return denoised


def stage3_fill_gaps_with_lines(mask):
    """
    Stage 3: Predictive Line Filling (HoughLinesP)
    ------------------------------------------------
    Roads may have small gaps due to thresholding artifacts or map artifacts.
    
    Strategy:
      1. Run Canny edge detection on the cleaned mask
      2. Use Probabilistic Hough Line Transform to detect line segments
      3. Draw detected lines back onto the mask to bridge gaps
    
    HoughLinesP parameters:
      - threshold: minimum votes (intersections) to accept a line
      - minLineLength: ignore short line fragments
      - maxLineGap: maximum gap between line segments to treat as one line
                    (this is the key parameter for gap filling)
    """
    # Canny edge detection on the binary mask
    edges = cv2.Canny(mask, 50, 150)

    # Detect line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    filled = mask.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(filled, (x1, y1), (x2, y2), 255, 1)
        print(f"[Stage 3] Detected {len(lines)} line segments, "
              f"filled gaps")
    else:
        print("[Stage 3] No line segments detected")

    pixel_count = cv2.countNonZero(filled)
    print(f"          Road pixels after fill: {pixel_count}")

    # Check connectivity improvement
    num_before, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    num_after, _, stats_after, _ = cv2.connectedComponentsWithStats(
        filled, connectivity=8
    )
    print(f"          Components: {num_before - 1} -> {num_after - 1}")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "stage3_gap_filled.png"), filled)
    return filled


def stage4_skeletonize(mask, stage_label="4", post_dilation=0):
    """
    Skeletonization Stage
    ----------------------
    Reduce all roads to 1-pixel-wide center lines.
    
    Benefits for RRT*:
      - Dramatically fewer road pixels -> faster collision checks
      - Cleaner topology, no redundant parallel paths
      - Consistent road representation regardless of original line thickness
    
    Process:
      1. Light dilation to merge nearby parallel lines before skeletonizing
      2. Apply morphological skeletonization (Zhang-Suen thinning)
      3. Light dilation to restore minimal traversable width for RRT*
    """
    # Pre-skeleton dilation: merge nearby lines
    if PRE_SKELETON_DILATION > 0:
        dilated = cv2.dilate(mask, KERNEL, iterations=PRE_SKELETON_DILATION)
    else:
        dilated = mask.copy()

    pre_pixels = cv2.countNonZero(dilated)
    print(f"[Stage {stage_label}] Pre-skeleton dilation: {pre_pixels} pixels")

    # Skeletonize (scikit-image)
    skeleton = skeletonize(dilated > 0).astype(np.uint8) * 255
    skel_pixels = cv2.countNonZero(skeleton)
    print(f"          Skeleton: {skel_pixels} pixels "
          f"({100 * skel_pixels / pre_pixels:.1f}% of dilated)")

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"stage{stage_label}_skeleton.png"), skeleton)

    # Post-skeleton dilation: restore minimal road width
    if post_dilation > 0:
        road_final = cv2.dilate(skeleton, KERNEL, iterations=post_dilation)
    else:
        road_final = skeleton.copy()

    final_pixels = cv2.countNonZero(road_final)
    print(f"          Final (after post-dilation): {final_pixels} pixels")

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"stage{stage_label}_final_road.png"), road_final)
    return road_final, skeleton


def verify_connectivity(mask, start=(15, 100), end=(380, 188)):
    """Verify that start and end points are connected on the road mask via BFS."""
    from collections import deque

    h, w = mask.shape
    sx, sy = start
    ex, ey = end

    # Snap points to nearest road pixel if needed
    if mask[sy, sx] == 0:
        ys, xs = np.where(mask > 0)
        dists = (xs - sx) ** 2 + (ys - sy) ** 2
        idx = dists.argmin()
        sx, sy = int(xs[idx]), int(ys[idx])
        print(f"[Verify] Start snapped to nearest road: ({sx}, {sy})")

    if mask[ey, ex] == 0:
        ys, xs = np.where(mask > 0)
        dists = (xs - ex) ** 2 + (ys - ey) ** 2
        idx = dists.argmin()
        ex, ey = int(xs[idx]), int(ys[idx])
        print(f"[Verify] End snapped to nearest road: ({ex}, {ey})")

    visited = np.zeros((h, w), dtype=bool)
    queue = deque([(sx, sy)])
    visited[sy, sx] = True
    found = False

    while queue:
        x, y = queue.popleft()
        if x == ex and y == ey:
            found = True
            break
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and mask[ny, nx] > 0:
                visited[ny, nx] = True
                queue.append((nx, ny))

    if found:
        print(f"[Verify] BFS: Path EXISTS from ({sx},{sy}) to ({ex},{ey})")
    else:
        reached = np.sum(visited)
        print(f"[Verify] BFS: NO PATH from ({sx},{sy}) to ({ex},{ey})")
        print(f"         Explored {reached} pixels")
    return found


def create_comparison_image(img, stages):
    """Create a side-by-side comparison of all stages for visualization."""
    h, w = img.shape[:2]
    labels = ["Original", "S1: Threshold", "S2: Cleaned",
              "S3: Skeleton1", "S4: Gap Fill", "S5: Final"]

    # Scale factor for fitting stages
    scale = 1
    sw, sh = w * scale, h * scale

    # Create a 2x3 grid
    grid = np.zeros((sh * 2 + 30, sw * 3, 3), dtype=np.uint8)

    for idx, (label, stage_img) in enumerate(zip(labels, stages)):
        row, col = divmod(idx, 3)
        x_off = col * sw
        y_off = row * (sh + 15)

        if len(stage_img.shape) == 2:
            vis = cv2.cvtColor(stage_img, cv2.COLOR_GRAY2BGR)
        else:
            vis = stage_img.copy()

        vis = cv2.resize(vis, (sw, sh))
        grid[y_off:y_off + sh, x_off:x_off + sw] = vis

        cv2.putText(grid, label, (x_off + 5, y_off + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "pipeline_comparison.png"), grid)
    print("\nSaved pipeline_comparison.png (all stages side by side)")


# ======================== MAIN ========================
def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the map
    img = cv2.imread(MAP_PATH)
    if img is None:
        print(f"Error: Could not load {MAP_PATH}")
        return
    h, w = img.shape[:2]
    print(f"Loaded {MAP_PATH}: {w}x{h}\n")

    # Run pipeline
    print("=" * 55)
    print("STAGE 1: HSV Color Thresholding")
    print("=" * 55)
    mask_threshold = stage1_color_threshold(img)

    print()
    print("=" * 55)
    print("STAGE 2: Remove Small Components (area < "
          f"{MIN_COMPONENT_AREA})")
    print("=" * 55)
    mask_cleaned = stage2_remove_small_components(mask_threshold)

    print()
    print("=" * 55)
    print("STAGE 2b: Denoise (Morphological Opening)")
    print("=" * 55)
    mask_denoised = stage2b_denoise(mask_cleaned)

    print()
    print("=" * 55)
    print("STAGE 3: Predictive Line Filling (HoughLinesP)")
    print("=" * 55)
    mask_filled = stage3_fill_gaps_with_lines(mask_denoised)

    print()
    print("=" * 55)
    print("STAGE 4: Skeletonization")
    print("=" * 55)
    mask_final, mask_skeleton = stage4_skeletonize(mask_filled, stage_label="4", post_dilation=POST_SKELETON_DILATION)

    # Verify connectivity
    print()
    print("=" * 55)
    print("CONNECTIVITY CHECK")
    print("=" * 55)
    verify_connectivity(mask_final)

    # Summary
    print()
    print("=" * 55)
    print("SUMMARY")
    print("=" * 55)
    total = h * w
    print(f"  Stage 1 (threshold):    {cv2.countNonZero(mask_threshold):6d} px "
          f"({100 * cv2.countNonZero(mask_threshold) / total:5.1f}%)")
    print(f"  Stage 2 (cleaned):      {cv2.countNonZero(mask_cleaned):6d} px "
          f"({100 * cv2.countNonZero(mask_cleaned) / total:5.1f}%)")
    print(f"  Stage 2b (denoised):    {cv2.countNonZero(mask_denoised):6d} px "
          f"({100 * cv2.countNonZero(mask_denoised) / total:5.1f}%)")
    print(f"  Stage 3 (gap filled):   {cv2.countNonZero(mask_filled):6d} px "
          f"({100 * cv2.countNonZero(mask_filled) / total:5.1f}%)")
    print(f"  Stage 4 (skeleton):     {cv2.countNonZero(mask_skeleton):6d} px "
          f"({100 * cv2.countNonZero(mask_skeleton) / total:5.1f}%)")

    # Save the final mask for RRT* to use
    cv2.imwrite(os.path.join(OUTPUT_DIR, "road_mask.png"), mask_final)
    print(f"\nFinal road mask saved to {OUTPUT_DIR}/road_mask.png")

    # Create comparison image
    create_comparison_image(
        img,
        [img, mask_threshold, mask_cleaned,
         mask_filled, mask_final],
    )

    print("\nOutput files:")
    print("  stage1_color_threshold.png  - Raw HSV threshold")
    print("  stage2_cleaned.png          - After removing small components")
    print("  stage2b_denoised.png        - After Gaussian blur denoising")
    print("  stage3_gap_filled.png       - After HoughLinesP gap filling")
    print("  stage4_skeleton.png         - Skeletonization")
    print("  stage4_final_road.png       - Final road mask (skeleton + dilation)")
    print("  road_mask.png               - Final mask for RRT* (same as above)")
    print("  pipeline_comparison.png     - Side-by-side comparison")


if __name__ == "__main__":
    main()
