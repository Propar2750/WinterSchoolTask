"""
RRT* Path Planning on IIT Kharagpur Road Network
=================================================
Implements the RRT* (Rapidly-exploring Random Tree Star) algorithm
to find an optimal path between two points on a binary road mask.

Key optimizations over standard RRT:
  - Informed sampling: only samples from road pixels (not uniform random)
  - KDTree: O(log n) nearest-neighbor queries instead of O(n) brute force
  - Rewiring: continuously improves existing paths through the tree
  - Goal bias: probabilistically steers toward the goal for faster convergence

Input (two modes):
  Mode A: --map image.png --mask road_mask.png    (pre-generated mask)
  Mode B: --map image.png --threshold 200         (binary threshold the map)

Output (saved to --output-dir):
  rrt_star_output.mp4   — Video of tree growth + final path
  rrt_star_result.jpg   — Final frame image

Parameters are auto-computed from image size (dim = max(width, height)):
  step_size    = max(3, round(dim * 0.005))
  goal_radius  = max(8, round(dim * 0.02))
  rewire_radius = max(15, round(dim * 0.025))
  max_iter     = max(50000, dim * 100)
  goal_bias    = 0.05

Examples:
  # Real image with preprocessed mask (params auto-scaled to 416px)
  python rrt_star.py --map processing_demo/map_cropped.jpg \\
      --mask processing_demo/output/road_mask.png \\
      --start 15,100 --end 380,188 --output-dir processing_demo/output

  # OSM demo with binary threshold (params auto-scaled to 800px)
  python rrt_star.py --map rrt_demo/output/osm_visual_map.png \\
      --threshold 200 --output-dir rrt_demo/output
"""

import argparse
import cv2
import numpy as np
import os
import random
import math
from scipy.spatial import KDTree


# ======================== COLORS (BGR) ========================
COLOR_TREE = (180, 130, 40)
COLOR_START = (0, 200, 0)
COLOR_END = (0, 0, 220)
COLOR_PATH = (0, 0, 255)
COLOR_TEXT_BG = (255, 255, 255)


# ======================== NODE CLASS ========================
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0


# ======================== RRT* HELPERS ========================
def distance(n1, n2):
    return math.hypot(n1.x - n2.x, n1.y - n2.y)


def sample_random(road_pixels, goal_node, goal_bias):
    if random.random() < goal_bias:
        return goal_node.x, goal_node.y
    idx = random.randint(0, len(road_pixels) - 1)
    return int(road_pixels[idx][0]), int(road_pixels[idx][1])


def steer(from_node, to_x, to_y, step_size, road_pixels_kd=None):
    dx = to_x - from_node.x
    dy = to_y - from_node.y
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return None
    if d <= step_size:
        new_x, new_y = to_x, to_y
    else:
        ratio = step_size / d
        new_x = int(from_node.x + dx * ratio)
        new_y = int(from_node.y + dy * ratio)
    # Snap to nearest road pixel so tree stays on the skeleton
    if road_pixels_kd is not None:
        _, idx = road_pixels_kd.query([new_x, new_y])
        pt = road_pixels_kd.data[idx]
        new_x, new_y = int(pt[0]), int(pt[1])
    return Node(new_x, new_y)


def collision_free(x1, y1, x2, y2, obstacle_map):
    """Check line stays on road. Uses a dilated mask for tolerance."""
    h, w = obstacle_map.shape
    num_checks = max(abs(x2 - x1), abs(y2 - y1), 1)
    for i in range(num_checks + 1):
        t = i / num_checks
        cx = int(x1 + t * (x2 - x1))
        cy = int(y1 + t * (y2 - y1))
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return False
        if obstacle_map[cy, cx] == 0:
            return False
    return True


def choose_best_parent(near_nodes, new_node, obstacle_map):
    best_parent = new_node.parent
    best_cost = new_node.cost
    for node in near_nodes:
        potential_cost = node.cost + distance(node, new_node)
        if potential_cost < best_cost:
            if collision_free(node.x, node.y, new_node.x, new_node.y, obstacle_map):
                best_parent = node
                best_cost = potential_cost
    new_node.parent = best_parent
    new_node.cost = best_cost
    return new_node


def rewire(tree, near_nodes, new_node, obstacle_map):
    for node in near_nodes:
        potential_cost = new_node.cost + distance(new_node, node)
        if potential_cost < node.cost:
            if collision_free(new_node.x, new_node.y, node.x, node.y, obstacle_map):
                node.parent = new_node
                node.cost = potential_cost


def extract_path(goal_node):
    path = []
    node = goal_node
    while node is not None:
        path.append((node.x, node.y))
        node = node.parent
    path.reverse()
    return path


def auto_pick_endpoints(road_pixels):
    """Pick two distant road pixels as start and end points."""
    n = len(road_pixels)
    sample_size = min(500, n)
    indices = np.random.choice(n, sample_size, replace=False)
    sampled = road_pixels[indices]

    best_dist = 0
    best_pair = (sampled[0], sampled[1])
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            d = (sampled[i][0] - sampled[j][0]) ** 2 + (sampled[i][1] - sampled[j][1]) ** 2
            if d > best_dist:
                best_dist = d
                best_pair = (sampled[i], sampled[j])

    start = (int(best_pair[0][0]), int(best_pair[0][1]))
    end = (int(best_pair[1][0]), int(best_pair[1][1]))
    print(f"Auto-picked start={start}, end={end} (dist={math.sqrt(best_dist):.0f}px)")
    return start, end


def snap_to_road(x, y, road_pixels, road_mask, label="Point"):
    """Snap a point to nearest road pixel if it's not on the road."""
    if road_mask[y, x] == 0:
        dists = (road_pixels[:, 0] - x) ** 2 + (road_pixels[:, 1] - y) ** 2
        idx = dists.argmin()
        nx, ny = int(road_pixels[idx, 0]), int(road_pixels[idx, 1])
        print(f"{label} snapped to nearest road pixel: ({nx}, {ny})")
        return nx, ny
    return x, y


# ======================== MAIN ========================
def run_rrt_star(args):
    # Load map image
    img = cv2.imread(args.map)
    if img is None:
        print(f"Error: Could not load {args.map}")
        return
    height, width = img.shape[:2]
    print(f"Map loaded: {width}x{height}")

    # Create or load road mask
    if args.mask:
        road_mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if road_mask is None:
            print(f"Error: Could not load mask {args.mask}")
            return
        if road_mask.shape[:2] != (height, width):
            road_mask = cv2.resize(road_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        print(f"Loaded mask: {args.mask}")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, road_mask = cv2.threshold(gray, args.threshold, 255, cv2.THRESH_BINARY_INV)
        print(f"Binary threshold at {args.threshold} (inverted)")

    free_count = cv2.countNonZero(road_mask)
    total = width * height
    print(f"Road pixels: {free_count} ({100 * free_count / total:.1f}%)")

    # Auto-compute parameters from image size if not explicitly set
    dim = max(width, height)
    if args.step_size is None:
        args.step_size = max(3, round(dim * 0.005))
    if args.goal_radius is None:
        args.goal_radius = max(8, round(dim * 0.02))
    if args.rewire_radius is None:
        args.rewire_radius = max(15, round(dim * 0.025))
    if args.max_iter is None:
        args.max_iter = max(50000, dim * 100)
    if args.goal_bias is None:
        args.goal_bias = 0.05
    if args.draw_every is None:
        args.draw_every = 5
    if args.fps is None:
        args.fps = 60
    print(f"Parameters (auto-scaled to {dim}px): step={args.step_size}, "
          f"goal_r={args.goal_radius}, rewire_r={args.rewire_radius}, "
          f"max_iter={args.max_iter}, bias={args.goal_bias}")

    # Create dilated mask for collision checking (tolerates 1px skeleton gaps)
    collision_mask = cv2.dilate(road_mask, np.ones((3, 3), np.uint8), iterations=2)

    road_ys, road_xs = np.where(road_mask > 0)
    road_pixels = np.column_stack((road_xs, road_ys))
    print(f"Road pixel array size: {len(road_pixels)}")

    # KDTree over road pixels for snap-to-road during steering
    road_pixels_kd = KDTree(road_pixels)

    # Determine start and end
    if args.start and args.end:
        sx, sy = args.start
        ex, ey = args.end
    else:
        (sx, sy), (ex, ey) = auto_pick_endpoints(road_pixels)

    sx, sy = snap_to_road(sx, sy, road_pixels, road_mask, "Start")
    ex, ey = snap_to_road(ex, ey, road_pixels, road_mask, "End")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    video_path = os.path.join(args.output_dir, "rrt_star_output.mp4")
    result_path = os.path.join(args.output_dir, "rrt_star_result.jpg")

    # Initialize RRT*
    start_pt = (sx, sy)
    end_pt = (ex, ey)
    start_node = Node(sx, sy)
    start_node.cost = 0.0
    goal_node = Node(ex, ey)
    tree = [start_node]

    # Marker size scales with image
    marker_r = max(3, min(width, height) // 100)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, args.fps, (width, height))

    canvas = img.copy()
    cv2.circle(canvas, start_pt, marker_r, COLOR_START, -1)
    cv2.circle(canvas, end_pt, marker_r, COLOR_END, -1)

    goal_reached = False
    best_goal_node = None

    print(f"Running RRT* (max {args.max_iter} iterations)...")
    print(f"  step_size={args.step_size}, goal_radius={args.goal_radius}, "
          f"rewire_radius={args.rewire_radius}, goal_bias={args.goal_bias}")

    tree_coords = np.array([[sx, sy]], dtype=np.float64)
    kd = KDTree(tree_coords)
    kd_rebuild_interval = max(20, args.max_iter // 2000)
    nodes_since_rebuild = 0

    for i in range(1, args.max_iter + 1):
        rx, ry = sample_random(road_pixels, goal_node, args.goal_bias)

        _, nearest_idx = kd.query([rx, ry])
        nearest = tree[nearest_idx]

        new_node = steer(nearest, rx, ry, args.step_size, road_pixels_kd)
        if new_node is None:
            continue

        if new_node.x < 0 or new_node.x >= width or new_node.y < 0 or new_node.y >= height:
            continue

        if not collision_free(nearest.x, nearest.y, new_node.x, new_node.y, collision_mask):
            continue

        new_node.parent = nearest
        new_node.cost = nearest.cost + distance(nearest, new_node)

        near_idxs = kd.query_ball_point([new_node.x, new_node.y], args.rewire_radius)
        near = [tree[j] for j in near_idxs]
        new_node = choose_best_parent(near, new_node, collision_mask)

        tree.append(new_node)
        tree_coords = np.vstack([tree_coords, [new_node.x, new_node.y]])
        nodes_since_rebuild += 1
        if nodes_since_rebuild >= kd_rebuild_interval:
            kd = KDTree(tree_coords)
            nodes_since_rebuild = 0

        rewire(tree, near, new_node, collision_mask)

        if new_node.parent is not None:
            cv2.line(canvas, (new_node.parent.x, new_node.parent.y),
                     (new_node.x, new_node.y), COLOR_TREE, 1)

        if i % args.draw_every == 0:
            frame = canvas.copy()
            cv2.circle(frame, start_pt, marker_r, COLOR_START, -1)
            cv2.circle(frame, end_pt, marker_r, COLOR_END, -1)
            text = f"Iter: {i}  |  Tree: {len(tree)}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (8, 6), (14 + tw, 30 + th), COLOR_TEXT_BG, -1)
            cv2.putText(frame, text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            out.write(frame)

        d_to_goal = math.hypot(new_node.x - ex, new_node.y - ey)
        if d_to_goal <= args.goal_radius:
            if not goal_reached or new_node.cost + d_to_goal < best_goal_node.cost:
                final = Node(ex, ey)
                final.parent = new_node
                final.cost = new_node.cost + d_to_goal
                tree.append(final)
                best_goal_node = final
                if not goal_reached:
                    print(f"  Goal reached at iteration {i}! Cost: {final.cost:.1f}")
                else:
                    print(f"  Better path at iteration {i}! Cost: {final.cost:.1f}")
                goal_reached = True

        if i % 2000 == 0:
            print(f"  Iteration {i}/{args.max_iter}, tree size: {len(tree)}")

    if not goal_reached:
        print("Goal NOT reached within max iterations.")
        print("Try adjusting: --step-size, --max-iter, or --start/--end points.")
        # Save current state as result even on failure
        fail_canvas = canvas.copy()
        text = f"Goal NOT reached  |  Tree: {len(tree)} nodes"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(fail_canvas, (8, 6), (14 + tw, 30 + th), COLOR_TEXT_BG, -1)
        cv2.putText(fail_canvas, text,
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(result_path, fail_canvas)
        print(f"Saved {result_path}")
    else:
        path = extract_path(best_goal_node)
        print(f"Final path: {len(path)} nodes, cost: {best_goal_node.cost:.1f}")

        final_canvas = canvas.copy()
        for j in range(len(path) - 1):
            cv2.line(final_canvas, path[j], path[j + 1], COLOR_PATH, max(2, marker_r // 2))
        cv2.circle(final_canvas, start_pt, marker_r + 1, COLOR_START, -1)
        cv2.circle(final_canvas, end_pt, marker_r + 1, COLOR_END, -1)
        cv2.circle(final_canvas, start_pt, marker_r + 1, (0, 0, 0), 2)
        cv2.circle(final_canvas, end_pt, marker_r + 1, (0, 0, 0), 2)
        text = f"Path found! Cost: {best_goal_node.cost:.1f}  |  Nodes: {len(path)}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(final_canvas, (8, 6), (14 + tw, 30 + th), COLOR_TEXT_BG, -1)
        cv2.putText(final_canvas, text,
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        for _ in range(args.fps * 3):
            out.write(final_canvas)

        cv2.imwrite(result_path, final_canvas)
        print(f"Saved {result_path}")

    out.release()
    print(f"Saved {video_path}")


def parse_point(s):
    """Parse 'x,y' string into (int, int) tuple."""
    parts = s.split(",")
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description="RRT* path planning on a road network mask.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    parser.add_argument("--map", required=True, help="Background map image path")
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument("--mask", help="Pre-generated binary road mask image")
    mask_group.add_argument("--threshold", type=int, default=None,
                            help="Binary threshold value to create mask from map "
                                 "(dark roads on light bg, inverted). Default: 200 if no --mask")

    # Endpoints
    parser.add_argument("--start", type=parse_point, default=None,
                        help="Start point as x,y (auto-picked if omitted)")
    parser.add_argument("--end", type=parse_point, default=None,
                        help="End point as x,y (auto-picked if omitted)")

    # RRT* parameters (all default to None = auto-computed from image size)
    parser.add_argument("--step-size", type=int, default=None,
                        help="Step size in pixels (default: auto = max(3, dim*0.005))")
    parser.add_argument("--goal-radius", type=int, default=None,
                        help="Goal radius in pixels (default: auto = max(8, dim*0.02))")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Max iterations (default: auto = max(50000, dim*100))")
    parser.add_argument("--rewire-radius", type=int, default=None,
                        help="Rewire radius in pixels (default: auto = max(15, dim*0.025))")
    parser.add_argument("--goal-bias", type=float, default=None,
                        help="Goal bias probability (default: 0.05)")

    # Output
    parser.add_argument("--output-dir", default=".", help="Directory for output files")
    parser.add_argument("--fps", type=int, default=None, help="Video FPS (default: 60)")
    parser.add_argument("--draw-every", type=int, default=None,
                        help="Write video frame every N iters (default: 5)")

    args = parser.parse_args()

    # Default: if no mask and no threshold specified, use threshold 200
    if args.mask is None and args.threshold is None:
        args.threshold = 200

    run_rrt_star(args)


if __name__ == "__main__":
    main()
