"""Compute family-appropriate feature counts on a processed preview image.

Each algorithm in the firmware writes a different *kind* of buffer:

  - **edges**       : magnitude image where bright pixels mark gradients.
                      Metric = fraction of pixels above a threshold.
  - **keypoints**   : dimmed scene + bright markers (5-pixel cross stamps).
                      Metric = number of local maxima > 200.
  - **binary**      : 0/255 mask.  Metric = fraction set to 255.
  - **intensity**   : tone-mapped image, no detection.  Metric = mean.
  - **none**        : nothing meaningful.  Metric = 0.

These are cheap, opencv-free numpy ops so we can run them on every
training image without slowing down the per-algo throughput
measurement.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .algos import stat_kind


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return (0.299 * img[..., 0] + 0.587 * img[..., 1]
            + 0.114 * img[..., 2]).astype(np.uint8)


def edge_pixel_ratio(img: np.ndarray, thr: int = 64) -> float:
    g = _to_gray(img)
    return float((g > thr).mean())


def binary_ratio(img: np.ndarray) -> float:
    g = _to_gray(img)
    return float((g >= 200).mean())


def keypoint_count(img: np.ndarray, thr: int = 200) -> int:
    """Count bright local maxima (3x3 NMS)."""
    g = _to_gray(img).astype(np.int32)
    h, w = g.shape
    if h < 3 or w < 3:
        return int((g >= thr).sum())
    c = g[1:-1, 1:-1]
    is_local_max = (
        (c >= g[:-2, :-2]) & (c >= g[:-2, 1:-1]) & (c >= g[:-2, 2:])
        & (c >= g[1:-1, :-2]) & (c >= g[1:-1, 2:])
        & (c >= g[2:, :-2]) & (c >= g[2:, 1:-1]) & (c >= g[2:, 2:])
        & (c >= thr)
    )
    return int(is_local_max.sum())


def mean_intensity(img: np.ndarray) -> float:
    return float(_to_gray(img).mean())


def metrics_for(algo_name: str, processed: np.ndarray) -> Dict[str, float]:
    """Compute the most informative metric for the given algo's output.

    Always returns the same dict keys so downstream code can build a
    DataFrame: ``{kind, value, edge_ratio, kp_count, binary_ratio,
    mean_intensity}``.  ``value`` is the kind-appropriate scalar.
    """
    kind = stat_kind(algo_name)
    er = edge_pixel_ratio(processed)
    kp = keypoint_count(processed)
    br = binary_ratio(processed)
    mi = mean_intensity(processed)
    if kind == "edges":
        value = er
    elif kind == "keypoints":
        value = float(kp)
    elif kind == "binary":
        value = br
    elif kind == "intensity":
        value = mi
    else:
        value = 0.0
    return {
        "kind": kind,
        "value": value,
        "edge_ratio": er,
        "kp_count": float(kp),
        "binary_ratio": br,
        "mean_intensity": mi,
    }
