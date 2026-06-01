"""Algorithm enum names, descriptions, and statistic kinds.

Index in ``ALGO_NAMES`` = algorithm ID sent over the wire (mirrors firmware
``imgproc/imgproc.h``).

For every algorithm this module also exposes:

* ``ALGO_DESCRIPTIONS``  – one-paragraph description for the GUI
* ``ALGO_STAT_KIND``     – which statistic is meaningful on the preview frame:

  - ``"edges"``      – number of non-zero pixels (edge magnitude)
  - ``"keypoints"``  – number of bright local maxima (corner/keypoint count)
  - ``"binary"``     – number of foreground (255) pixels in a mask
  - ``"intensity"``  – mean pixel intensity (filters that just transform tones)
  - ``"none"``       – nothing meaningful to count
"""
from __future__ import annotations

from typing import Dict, List, Tuple

ALGO_NAMES: List[str] = [
    "passthrough",
    "grayscale",
    "invert",
    "hist_eq",
    "gaussian_3",
    "gaussian_5",
    "mean_3",
    "median_3",
    "bilateral",
    "sobel",
    "roberts",
    "prewitt",
    "scharr",
    "kirsch",
    "frei_chen",
    "canny",
    "marr_hildreth",
    "laplacian",
    "dog",
    "log",
    "doh",
    "harris",
    "shi_tomasi",
    "fast9",
    "otsu",
    "adaptive_mean",
    "adaptive_gaussian",
    "triangle",
    "niblack",
    "sauvola",
    "gabor",
    "lbp",
    "laws_energy",
    "frangi",
    "hessian_ridge",
    "hog_vis",
    "erode",
    "dilate",
    "open",
    "close",
    "morph_gradient",
    "region_grow",
    "watershed",
    "sharpen",
    "emboss",
    "mser",
    "agast",
    "brief",
    "akaze",
    "blob_log_multiscale",
    "fast12",
    # ----- end of firmware-supported set (id 0..50) -----
    # The entries below are *host-known names* for keypoint algorithms
    # that are not yet implemented on the board.  They appear in
    # ``ALGO_DESCRIPTIONS`` so the GUI can show docs once firmware adds
    # them; ``BoardLink.get_info()`` will only expose the algos the firmware
    # actually advertises.
    "orb",
    "sift",
    "surf",
]

#: number of algorithm slots the firmware currently implements
FIRMWARE_ALGO_COUNT = 51

# family id -> human label (matches firmware imgproc_info_t.family)
ALGO_FAMILIES: Dict[int, str] = {
    0: "basics",
    1: "edges",
    2: "blobs",
    3: "keypoints",
    4: "thresholding",
    5: "texture",
    6: "ridge",
    7: "morphology",
}


def family_name(fid: int) -> str:
    return ALGO_FAMILIES.get(fid, f"family_{fid}")


# Reverse: algorithm name -> family label.  Built lazily so that
# ALGO_STAT_KIND, ALGO_DESCRIPTIONS and the firmware family numbers stay
# the single source of truth and we don't double-define.
_NAME_TO_FAMILY: Dict[str, str] = {
    # basics
    "passthrough": "basics", "grayscale": "basics", "invert": "basics",
    "hist_eq": "basics", "gaussian_3": "basics", "gaussian_5": "basics",
    "mean_3": "basics", "median_3": "basics", "bilateral": "basics",
    "sharpen": "basics",
    # edges
    "sobel": "edges", "roberts": "edges", "prewitt": "edges",
    "scharr": "edges", "kirsch": "edges", "frei_chen": "edges",
    "canny": "edges", "marr_hildreth": "edges", "laplacian": "edges",
    # blobs
    "dog": "blobs", "log": "blobs", "doh": "blobs",
    "blob_log_multiscale": "blobs", "mser": "blobs",
    # keypoints
    "harris": "keypoints", "shi_tomasi": "keypoints", "fast9": "keypoints",
    "fast12": "keypoints", "agast": "keypoints", "orb": "keypoints",
    "brief": "keypoints", "sift": "keypoints", "surf": "keypoints",
    "akaze": "keypoints",
    # thresholding
    "otsu": "thresholding", "adaptive_mean": "thresholding",
    "adaptive_gaussian": "thresholding", "triangle": "thresholding",
    "niblack": "thresholding", "sauvola": "thresholding",
    # texture
    "gabor": "texture", "lbp": "texture", "laws_energy": "texture",
    "hog_vis": "texture", "emboss": "texture",
    # ridge
    "frangi": "ridge", "hessian_ridge": "ridge",
    # morphology
    "erode": "morphology", "dilate": "morphology", "open": "morphology",
    "close": "morphology", "morph_gradient": "morphology",
    "region_grow": "morphology", "watershed": "morphology",
}


def family_of(name: str) -> str:
    """Return the family label for an algorithm name."""
    return _NAME_TO_FAMILY.get(name, "other")


# ---------------------------------------------------------------- descriptions
ALGO_DESCRIPTIONS: Dict[str, str] = {
    "passthrough":      "No processing — the camera frame is forwarded unchanged. "
                        "Useful as a baseline for FPS and inference timing.",
    "grayscale":        "Converts RGB888 to a single luminance channel "
                        "(Y = 0.299R + 0.587G + 0.114B). Most edge / keypoint "
                        "operators run on this.",
    "invert":           "Pixel-wise photographic negative (255 - x). Highlights "
                        "low-intensity regions; sanity-check filter.",
    "hist_eq":          "Global histogram equalisation: redistributes intensities "
                        "so the cumulative distribution becomes linear, boosting "
                        "contrast.",
    "gaussian_3":       "3x3 Gaussian smoothing kernel (sigma ~ 0.85). Removes "
                        "high-frequency noise before edge detection.",
    "gaussian_5":       "5x5 Gaussian smoothing kernel (sigma ~ 1.4). Stronger "
                        "blur, larger support — recommended for noisy sensors.",
    "mean_3":           "3x3 box (mean) filter. Cheap denoiser that preserves "
                        "local average but blurs edges.",
    "median_3":         "3x3 median filter. Removes salt-and-pepper noise while "
                        "preserving edges better than the mean filter.",
    "bilateral":        "Edge-preserving smoothing: spatial and intensity Gaussian "
                        "kernels combined, so flat regions blur but edges stay "
                        "sharp.",
    "sobel":            "Sobel gradient magnitude |Gx|+|Gy| with separable 3x3 "
                        "kernels. Classic, fast first-derivative edge detector.",
    "roberts":          "Roberts cross 2x2 diagonal-difference operator. Cheapest "
                        "edge detector but very noise-sensitive.",
    "prewitt":          "Prewitt 3x3 gradient operator. Similar to Sobel but with "
                        "equal row weights.",
    "scharr":           "Scharr 3x3 operator with optimal rotation invariance "
                        "among 3x3 kernels.",
    "kirsch":           "Kirsch compass operator: takes the maximum response over "
                        "eight rotated 3x3 kernels.",
    "frei_chen":        "Frei-Chen 3x3 basis: projects the patch onto orthogonal "
                        "edge / line / mean basis vectors and reports the "
                        "edge-subspace ratio.",
    "canny":            "Canny multi-stage edge detector: Gaussian -> gradient -> "
                        "non-maxima suppression -> hysteresis thresholding. "
                        "Produces thin, single-pixel-wide edges.",
    "marr_hildreth":    "Marr-Hildreth: Gaussian blur followed by a Laplacian; "
                        "edges are the zero-crossings of the result.",
    "laplacian":        "Discrete Laplacian. Second-derivative response that "
                        "highlights blobs and intensity peaks.",
    "dog":              "Difference of Gaussians: blob detector built from two "
                        "blurred copies (sigma ratio ~ 1.6). Used by SIFT-style "
                        "scale spaces.",
    "log":              "Laplacian of Gaussian: convolve with a sigma-tuned LoG "
                        "kernel. Strong blob/spot detector at a single scale.",
    "doh":              "Determinant of Hessian: |Hxx*Hyy - Hxy^2|. SURF-style "
                        "blob response, fast via integral images.",
    "harris":           "Harris corner response R = det(M) - k * trace(M)^2. Marks "
                        "corners where local autocorrelation has two strong "
                        "eigenvalues.",
    "shi_tomasi":       "Shi-Tomasi (`good features to track`): R = min(l1, l2) of "
                        "the structure tensor. Slightly more stable than Harris.",
    "fast9":            "FAST-9 keypoint detector: a pixel is a corner if 9 "
                        "contiguous ring pixels are all brighter or darker than "
                        "centre +/- t.",
    "otsu":             "Otsu global threshold: picks T that maximises inter-class "
                        "variance of the histogram. Output is binary 0/255.",
    "adaptive_mean":    "Adaptive threshold using local mean (window minus C). "
                        "Robust to uneven lighting.",
    "adaptive_gaussian":"Adaptive threshold using a Gaussian-weighted local mean. "
                        "Smoother boundaries than the mean variant.",
    "triangle":         "Triangle thresholding: T at the maximum distance from the "
                        "histogram peak to a line ending at the rightmost non-empty "
                        "bin.",
    "niblack":          "Niblack local thresholding: T(x,y) = mu(x,y) + k*sigma(x,y). "
                        "Good for textured documents, sensitive to noise.",
    "sauvola":          "Sauvola thresholding: T(x,y) = mu * (1 + k*(sigma/R - 1)). "
                        "Modern improvement on Niblack used in OCR pre-processing.",
    "gabor":            "Gabor filter response: sinusoid modulated by a Gaussian, "
                        "tuned to a single orientation/frequency. Captures oriented "
                        "texture.",
    "lbp":              "Local Binary Patterns: each pixel encoded as the 8-bit "
                        "pattern of neighbours brighter/darker than itself. Texture "
                        "descriptor.",
    "laws_energy":      "Laws texture energy: convolve with five 1-D micro-kernels "
                        "(L,E,S,W,R), square, and average over a window.",
    "frangi":           "Frangi vesselness filter: ratio of Hessian eigenvalues — "
                        "high on tubular ridge structures (vessels, fingerprints).",
    "hessian_ridge":    "Generic ridge detector: sign and magnitude of the largest "
                        "Hessian eigenvalue. Highlights line-like structures.",
    "hog_vis":          "Histogram of Oriented Gradients visualisation: per-cell "
                        "oriented stripes whose intensity reflects gradient strength.",
    "erode":            "Morphological erosion (3x3 cross). Shrinks bright regions, "
                        "removes small specks.",
    "dilate":           "Morphological dilation (3x3 cross). Grows bright regions, "
                        "fills small holes.",
    "open":             "Opening = erode then dilate. Removes small bright noise "
                        "without shrinking large structures.",
    "close":            "Closing = dilate then erode. Fills small dark holes while "
                        "preserving overall shape.",
    "morph_gradient":   "Morphological gradient = dilate - erode. Cheap edge-strength "
                        "image highlighting object boundaries.",
    "region_grow":      "Region growing from seed pixels: floods neighbouring pixels "
                        "whose intensity is within a tolerance.",
    "watershed":        "Watershed segmentation: treats the image as a height map "
                        "and floods from local minima — boundaries are watershed "
                        "lines.",
    "sharpen":          "Unsharp mask sharpen on luminance: 3x3 cross kernel "
                        "(centre 5, neighbours -1) amplifies high-frequency "
                        "detail without changing average brightness.",
    "emboss":           "Emboss filter: directional 3x3 kernel produces a "
                        "pseudo-3D relief; flat regions become mid-grey, edges "
                        "show as light/dark depending on gradient direction.",
    "mser":             "Maximally Stable Extremal Regions: finds intensity-stable "
                        "blobs by sweeping the threshold and keeping connected "
                        "components whose area changes slowly.",
    "blob_log_multiscale":
                        "Multi-scale Laplacian-of-Gaussian blob detector: scans "
                        "several sigmas and reports the (x, y, sigma) of local "
                        "extrema in scale-space.",
    "fast12":           "FAST-12 keypoint detector: stricter variant of FAST that "
                        "requires 12 contiguous brighter/darker ring pixels. Fewer "
                        "but more stable corners than FAST-9.",
    "agast":            "AGAST corner detector (adaptive accelerated FAST). "
                        "Decision-tree learned per environment for higher repeat "
                        "rate than FAST.",
    "orb":              "ORB = oriented FAST + rotated BRIEF. Fast binary keypoint "
                        "detector and descriptor used in real-time SLAM.",
    "brief":            "BRIEF descriptor: 256-bit binary descriptor built from "
                        "intensity comparisons in a fixed sampling pattern. Run on "
                        "top of any keypoint detector.",
    "sift":             "SIFT keypoint + 128-D float descriptor. Scale-space DoG "
                        "extrema, orientation assignment, gradient histogram "
                        "descriptor. Reference quality, slower than ORB.",
    "surf":             "SURF keypoint + 64-D float descriptor. Hessian-determinant "
                        "blobs in box-filter scale-space; faster SIFT-style "
                        "alternative.",
    "akaze":            "AKAZE: Accelerated-KAZE non-linear scale-space corner / "
                        "blob detector with M-LDB binary descriptor. Better edges "
                        "than SIFT/SURF.",
}


# ---------------------------------------------------------------- stat kinds
ALGO_STAT_KIND: Dict[str, str] = {
    # basics
    "passthrough": "intensity",
    "grayscale": "intensity",
    "invert": "intensity",
    "hist_eq": "intensity",
    "gaussian_3": "intensity",
    "gaussian_5": "intensity",
    "mean_3": "intensity",
    "median_3": "intensity",
    "bilateral": "intensity",
    # edge magnitude images
    "sobel": "edges",
    "roberts": "edges",
    "prewitt": "edges",
    "scharr": "edges",
    "kirsch": "edges",
    "frei_chen": "edges",
    "canny": "edges",
    "marr_hildreth": "edges",
    "laplacian": "edges",
    "dog": "edges",
    "log": "edges",
    "doh": "edges",
    # keypoints
    "harris": "keypoints",
    "shi_tomasi": "keypoints",
    "fast9": "keypoints",
    # binary masks
    "otsu": "binary",
    "adaptive_mean": "binary",
    "adaptive_gaussian": "binary",
    "triangle": "binary",
    "niblack": "binary",
    "sauvola": "binary",
    # texture / ridge
    "gabor": "edges",
    "lbp": "intensity",
    "laws_energy": "edges",
    "frangi": "edges",
    "hessian_ridge": "edges",
    "hog_vis": "edges",
    # morphology
    "erode": "intensity",
    "dilate": "intensity",
    "open": "intensity",
    "close": "intensity",
    "morph_gradient": "edges",
    "region_grow": "binary",
    "watershed": "edges",
    "sharpen": "intensity",
    "emboss": "edges",
    # extensions
    "mser": "binary",
    "blob_log_multiscale": "keypoints",
    "fast12": "keypoints",
    "agast": "keypoints",
    "orb": "keypoints",
    "brief": "keypoints",
    "sift": "keypoints",
    "surf": "keypoints",
    "akaze": "keypoints",
}


def description(name: str) -> str:
    """Return a human description of an algorithm, or a placeholder."""
    return ALGO_DESCRIPTIONS.get(name, "(no description available)")


def stat_kind(name: str) -> str:
    """Return the preview-statistic kind: edges/keypoints/binary/intensity/none."""
    return ALGO_STAT_KIND.get(name, "none")


# ----------------------------------------------------------- preview statistics
# Thresholds tuned for 8-bit preview frames. Kept conservative so a single
# bright pixel (e.g. compression ringing) is not counted as an edge or keypoint.
_EDGE_THRESHOLD = 32         # edge-magnitude images: pixel >= this -> edge
_KEYPOINT_THRESHOLD = 200    # keypoint heatmaps: pixel >= this -> candidate
_KEYPOINT_NMS_RADIUS = 2     # non-maxima suppression radius (px)


def _count_above(buf: bytes, threshold: int, n_pixels: int) -> int:
    # avoid a numpy dependency; bytes iteration is cheap enough at 320x240
    cnt = 0
    for i in range(n_pixels):
        if buf[i] >= threshold:
            cnt += 1
    return cnt


def _count_keypoints(buf: bytes, w: int, h: int,
                     threshold: int = _KEYPOINT_THRESHOLD,
                     radius: int = _KEYPOINT_NMS_RADIUS) -> int:
    """Approximate keypoint count: bright local maxima after simple NMS."""
    if w <= 2 * radius or h <= 2 * radius:
        return 0
    count = 0
    for y in range(radius, h - radius):
        row = y * w
        for x in range(radius, w - radius):
            v = buf[row + x]
            if v < threshold:
                continue
            is_max = True
            for dy in range(-radius, radius + 1):
                base = (y + dy) * w
                for dx in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    if buf[base + x + dx] > v:
                        is_max = False
                        break
                if not is_max:
                    break
            if is_max:
                count += 1
    return count


def compute_preview_stats(algo_name: str, buf: bytes,
                          w: int, h: int) -> Tuple[str, int]:
    """Compute the headline statistic for a preview frame.

    Returns ``(label, value)`` where ``label`` is a short string suited to a
    GUI label (e.g. ``"edge px"``, ``"keypoints"``, ``"on px"``, ``"mean"``).
    """
    n = w * h
    if w <= 0 or h <= 0 or len(buf) < n:
        return ("", 0)

    kind = stat_kind(algo_name)
    if kind == "edges":
        return ("edge px", _count_above(buf, _EDGE_THRESHOLD, n))
    if kind == "keypoints":
        return ("keypoints", _count_keypoints(buf, w, h))
    if kind == "binary":
        return ("on px", _count_above(buf, 128, n))
    if kind == "intensity":
        return ("mean", sum(buf[: n]) // n)
    return ("", 0)
