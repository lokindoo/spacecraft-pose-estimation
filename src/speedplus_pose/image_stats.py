"""
Image statistics that can tell a soft bright blob from a hard specular edge.

Figure 1 compared pooled pixel-intensity densities, which both reviewers
pointed out cannot separate the two: a smooth glare blob and a clipped,
hard-edged highlight can put the same mass in the same bright bins. The
mechanism the paper proposes rests on exactly that spatial difference, so it
needs statistics that see it:

  gradient magnitude   Sobel, scaled so a full black-to-white step reads 1. A
                       soft blob adds mass near 0; hard shadow boundaries and
                       specular edges add mass in the upper bins.
  edge density         share of pixels whose gradient magnitude exceeds
                       EDGE_THRESHOLD.
  clipped fraction     share of pixels at the sensor's ceiling (>= 254/255),
                       i.e. saturated, which sunlamp shows and the soft
                       photometric jitter rarely produces.

Everything is accumulated as bin counts and pixel totals, one image at a time,
so a 28,000-image pool costs one image of memory. Distances between pooled
densities are total variation, 0.5 * sum |p - q| over probability masses, with
the intensity binning Figure 1 uses (48 bins on [0, 1]).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

INTENSITY_EDGES = np.linspace(0.0, 1.0, 49)  # 48 bins, as in Figure 1
GRADIENT_EDGES = np.linspace(0.0, 1.0, 65)
EDGE_THRESHOLD = 0.1  # a tenth of a full black-to-white step
CLIP_LEVEL = 254 / 255

COUNT_KEYS = ("intensity_counts", "gradient_counts")
SUM_KEYS = ("pixels", "edge_pixels", "clipped_pixels", "gradient_sum", "images")


def gradient_magnitude(plane: np.ndarray) -> np.ndarray:
    """Sobel magnitude, divided by 4 so a unit step in either axis reads 1."""
    gx = ndimage.sobel(plane, axis=1, mode="reflect")
    gy = ndimage.sobel(plane, axis=0, mode="reflect")
    return np.hypot(gx, gy) / 4.0


def image_statistics(plane: np.ndarray) -> dict:
    """Counts and sums for one grayscale plane in [0, 1]."""
    plane = np.asarray(plane, dtype=np.float64)
    magnitude = gradient_magnitude(plane)
    return {
        "intensity_counts": np.histogram(plane, bins=INTENSITY_EDGES)[0],
        # Diagonal steps can exceed 1; they belong in the top bin, not off the end.
        "gradient_counts": np.histogram(np.minimum(magnitude, 1.0), bins=GRADIENT_EDGES)[0],
        "pixels": plane.size,
        "edge_pixels": int((magnitude > EDGE_THRESHOLD).sum()),
        "clipped_pixels": int((plane >= CLIP_LEVEL).sum()),
        "gradient_sum": float(magnitude.sum()),
        "images": 1,
    }


def empty_accumulator() -> dict:
    return {
        "intensity_counts": np.zeros(len(INTENSITY_EDGES) - 1, dtype=np.int64),
        "gradient_counts": np.zeros(len(GRADIENT_EDGES) - 1, dtype=np.int64),
        **{key: 0 for key in SUM_KEYS},
    }


def accumulate(total: dict, item: dict) -> dict:
    for key in COUNT_KEYS:
        total[key] = total[key] + item[key]
    for key in SUM_KEYS:
        total[key] += item[key]
    return total


def summarise_pool(total: dict) -> dict:
    """Pooled densities and rates for one split."""
    if total["pixels"] == 0:
        raise ValueError("empty pool")
    return {
        "images": int(total["images"]),
        "intensity_mass": total["intensity_counts"] / total["intensity_counts"].sum(),
        "gradient_mass": total["gradient_counts"] / total["gradient_counts"].sum(),
        "edge_density": total["edge_pixels"] / total["pixels"],
        "clipped_fraction": total["clipped_pixels"] / total["pixels"],
        "mean_gradient": total["gradient_sum"] / total["pixels"],
    }


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """0.5 * sum |p - q| over two probability mass vectors on the same bins."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    return 0.5 * float(np.abs(p / p.sum() - q / q.sum()).sum())
