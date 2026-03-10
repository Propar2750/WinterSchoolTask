# Initial Attempts — Preprocessing Pipeline

## What Didn't Work

### 1. Morphological Opening (3×3 kernel) after Stage 2
- **Goal:** Denoise the binary mask after removing small connected components.
- **Method:** `cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3)))` 
- **Result:** Too aggressive — eroded thin road lines, fragmenting the network from ~23 components to 231. Removed 8,775 pixels (14,656 → 5,881), destroying connectivity between start and end.
- **Why it failed:** Road lines are only 1–3 px wide in many places. Opening (erosion then dilation) erases anything thinner than the kernel, which includes most of the road network.

## Next to Try

### 2. Gaussian Blur (3×3) + Re-threshold ✅ WORKED
- **Idea:** Apply a 3×3 Gaussian blur to the Stage 2 cleaned mask, then re-threshold to binarize. The blur should smooth out noise pixels while preserving larger road structures, and re-thresholding snaps it back to binary.
- **Method:** `cv2.GaussianBlur(mask, (3, 3), 0)` followed by `cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)`
- **Result:** Successfully removed isolated noise pixels while preserving road connectivity. Pixel count change was minimal, and BFS path still exists between start and end.
- **Why it worked:** Unlike morphological opening (which erodes then dilates), Gaussian blur merely averages pixel values. Isolated white pixels surrounded by black get averaged out, but connected road lines (with enough neighboring white pixels) survive the threshold. This makes it ideal for thin road structures.
