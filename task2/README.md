# Task 2: Background Removal via K-Means Clustering

## Objective

Use **K-Means clustering** to segment an image, identify the background cluster, and replace it with a stock background image (similar to a Zoom virtual background).

**Input image:** `individuals-a.png` — a photo with a roughly uniform background.  
**Replacement background:** `pngtree-abstract-bg-image_914283.png` — an abstract stock image.

---

## Approach

### 1. K-Means Clustering in LAB Color Space

Rather than clustering in RGB (where perceptual similarity does not map well to Euclidean distance), the image is first converted to **CIE LAB** color space. LAB is perceptually uniform — equal distances in LAB correspond to equal perceived color differences — making K-Means clusters more meaningful.

- The image is Gaussian-blurred (5×5 kernel) to reduce noise before clustering.
- Each pixel becomes a 3D feature vector `(L, a, b)`.
- OpenCV's `cv2.kmeans` partitions all pixels into **K = 5** clusters using K-Means++ initialization, run 10 times to avoid poor local minima.

### 2. Background Cluster Identification

The background cluster is identified using a **border-dominance heuristic**: pixels along all four image edges are collected, and the cluster label most frequent on the border is chosen as the background. This works because background regions typically extend to the image edges.

### 3. Foreground Mask Generation

A binary mask is created (`foreground = 255`, `background = 0`) and cleaned up with:

- **Morphological closing** (elliptical 7×7 kernel, 3 iterations) — fills small holes inside the foreground region.
- **Morphological opening** (same kernel, 3 iterations) — removes small noise blobs misclassified as foreground.
- **Gaussian feathering** (σ = 5 px) — softens mask edges for smoother blending.

### 4. Compositing

The replacement background is resized to match the input image dimensions. The final result is an **alpha blend**:

$$\text{result} = \alpha \cdot \text{foreground} + (1 - \alpha) \cdot \text{background}$$

where $\alpha$ is the feathered foreground mask normalized to $[0, 1]$.

---

## Pipeline Summary

```
Input Image
    │
    ▼
Gaussian Blur (5×5)
    │
    ▼
Convert BGR → LAB
    │
    ▼
K-Means Clustering (K=5)
    │
    ├──► Segmented cluster visualization
    │
    ▼
Border-based background cluster detection
    │
    ▼
Binary foreground mask
    │
    ▼
Morphological close → open → Gaussian feather
    │
    ▼
Alpha-blend with replacement background
    │
    ▼
Final Result
```

---

## Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `K` | 5 | Number of K-Means clusters |
| `BLUR_KSIZE` | 5 | Gaussian blur kernel size (pre-clustering) |
| `MORPH_KSIZE` | 7 | Morphological kernel size |
| `MORPH_ITERS` | 3 | Morphological close/open iterations |
| `FEATHER_PX` | 5 | Gaussian sigma for mask edge feathering |

---

## Output Files

| File | Description |
|---|---|
| `output/segmented_clusters.png` | Image recolored by cluster centers — visualizes what K-Means "sees" |
| `output/foreground_mask.png` | Binary/feathered mask separating foreground from background |
| `output/result.png` | Final composite with replaced background |
| `output/comparison.png` | Side-by-side: original → clusters → new background → result |

**Results:**

![Comparison](output/comparison.png)

---

## How to Run

```bash
cd task2
python kmeans_bg_removal.py
```

Requires: `opencv-python`, `numpy`

---

## Core Topics

- **Image Processing** — color space conversion, Gaussian blur, morphological operations
- **OpenCV** — `cv2.kmeans`, `cv2.morphologyEx`, `cv2.GaussianBlur`, alpha blending
- **Unsupervised Learning** — K-Means clustering applied to pixel color segmentation

---

## Notes

- Images with a **large, roughly uniform background** work best, since the background will form one of the dominant clusters and will be well-represented on the border.
- Increasing `K` can help separate foreground details but may also fragment the background into multiple clusters. K = 5 is a good default.
- For images where the foreground touches the border, the border-dominance heuristic may misidentify clusters — in such cases, manual cluster selection or additional spatial features would be needed.
