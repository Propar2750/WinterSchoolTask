"""
Task 2: Background Removal via K-Means Clustering
==================================================
Segments the input image using K-Means clustering in LAB color space,
identifies the background cluster (largest cluster touching the image border),
and replaces it with a stock background image.
"""

import cv2
import numpy as np
import os

# Resolve all paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================== CONFIGURATION ========================
INPUT_IMAGE = os.path.join(SCRIPT_DIR, "person1.jpg")
BG_IMAGE = os.path.join(SCRIPT_DIR, "pngtree-abstract-bg-image_914283.png")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

K = 3            # number of clusters
BLUR_KSIZE = 5     # GaussianBlur kernel size before clustering
MORPH_KSIZE = 7      # morphological cleanup kernel size
MORPH_ITERS = 3      # morphological close/open iterations
FEATHER_PX = 5       # Gaussian blur sigma for mask edge feathering


# ======================== PIPELINE ========================

def load_images(input_path, bg_path):
    img = cv2.imread(input_path)
    bg = cv2.imread(bg_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load input image: {input_path}")
    if bg is None:
        raise FileNotFoundError(f"Cannot load background image: {bg_path}")
    return img, bg


def kmeans_segment(img, k):
    """Run K-Means on the image in LAB color space and return labels + centers."""
    # Convert to LAB for perceptually uniform clustering
    blurred = cv2.GaussianBlur(img, (BLUR_KSIZE, BLUR_KSIZE), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # Using HSV for better color separation in this case
    h, w = hsv.shape[:2]

    # Flatten pixels to feature vectors
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    # K-Means criteria and run
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    labels = labels.reshape(h, w)
    return labels, centers


def identify_background_cluster(labels):
    """
    Identify the two background clusters as the two clusters most
    present along the image borders (top/bottom/left/right edges).
    """
    h, w = labels.shape

    # Collect labels along all four borders
    border_labels = np.concatenate([
        labels[0, :],       # top row
        #labels[h - 1, :],   # bottom row excluded 
        labels[:, 0],       # left column
        labels[:, w - 1],   # right column
    ])

    # Count how often each label appears on the border
    unique, counts = np.unique(border_labels, return_counts=True)

    # The two clusters most present on the border are background
    sorted_indices = np.argsort(-counts)  # descending by count
    bg_labels = unique[sorted_indices[:2]].tolist()
    return bg_labels


def create_foreground_mask(labels, bg_labels):
    """Create a binary mask where foreground=255 and background=0, with cleanup."""
    bg_mask = np.zeros(labels.shape, dtype=bool)
    for bl in bg_labels:
        bg_mask |= (labels == bl)
    mask = np.where(bg_mask, 0, 255).astype(np.uint8)

    # Morphological close (fill small holes in foreground) then open (remove noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KSIZE, MORPH_KSIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERS)

    # Feather edges with Gaussian blur for smoother blending
    if FEATHER_PX > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=FEATHER_PX)

    return mask


def build_segmented_image(img, labels, centers):
    """Reconstruct the image with each pixel replaced by its cluster center color."""
    h, w = img.shape[:2]
    centers_bgr = cv2.cvtColor(
        centers.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_Lab2BGR
    ).reshape(-1, 3)
    segmented = centers_bgr[labels.flatten()].reshape(h, w, 3)
    return segmented


def composite(fg_img, bg_img, mask):
    """Alpha-blend foreground onto resized background using the mask."""
    h, w = fg_img.shape[:2]
    bg_resized = cv2.resize(bg_img, (w, h), interpolation=cv2.INTER_AREA)

    # Normalize mask to [0, 1] for blending
    alpha = mask.astype(np.float32) / 255.0
    alpha = alpha[:, :, np.newaxis]  # (H, W, 1)

    result = (fg_img.astype(np.float32) * alpha +
              bg_resized.astype(np.float32) * (1 - alpha))
    return result.astype(np.uint8)


def save(name, img, output_dir):
    path = os.path.join(output_dir, name)
    cv2.imwrite(path, img)
    print(f"  Saved: {path}")


# ======================== MAIN ========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/5] Loading images ...")
    img, bg = load_images(INPUT_IMAGE, BG_IMAGE)
    print(f"  Input : {img.shape[1]}x{img.shape[0]}")
    print(f"  Background: {bg.shape[1]}x{bg.shape[0]}")

    print(f"[2/5] Running K-Means clustering (K={K}) ...")
    labels, centers = kmeans_segment(img, K)

    # Save the segmented (cluster-colored) visualization
    segmented = build_segmented_image(img, labels, centers)
    save("segmented_clusters.png", segmented, OUTPUT_DIR)

    print("[3/5] Identifying background clusters ...")
    bg_labels = identify_background_cluster(labels)
    print(f"  Background cluster labels: {bg_labels}")

    print("[4/5] Creating foreground mask ...")
    mask = create_foreground_mask(labels, bg_labels)
    save("foreground_mask.png", mask, OUTPUT_DIR)

    print("[5/5] Compositing with new background ...")
    result = composite(img, bg, mask)
    save("result.png", result, OUTPUT_DIR)

    # Also save a side-by-side comparison
    h, w = img.shape[:2]
    bg_resized = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)
    comparison = np.hstack([img, segmented, bg_resized, result])
    save("comparison.png", comparison, OUTPUT_DIR)

    print("\nDone! Check the output in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
