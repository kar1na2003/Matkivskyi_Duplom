"""Host-side reference implementations of the 51 firmware imgproc algorithms.

These mirror the C implementations in the firmware tree at
``proj_cm55/source/imgproc/imgproc.c`` closely enough that the *kind* of
feature each algorithm extracts is the same (edges, keypoints, blob
maps, binary masks, ridge maps, etc.).  Pixel-perfect parity with the
embedded code is **not** required: the goal is to let us run the same
51 preprocessing pipelines over a labelled dataset on the laptop, train
a tiny NN per algorithm, and compare classification accuracy across
algos.

We use OpenCV where it provides a faithful equivalent to the firmware
operator, and pure NumPy elsewhere.  Every function takes a uint8 RGB
image of arbitrary size and returns a uint8 image of the same shape so
downstream training code can chain them transparently.

The function names match ``modusmate_host.algos.ALGO_NAMES[0..50]``
exactly; ``ALGO_FUNCS`` maps name -> callable.
"""
from __future__ import annotations

from typing import Callable, Dict

import cv2
import numpy as np

# ---------------------------------------------------------------- helpers


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _gray_to_rgb(g: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


def _abs_to_u8(x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    if a.max() > 255:
        a = a * (255.0 / max(a.max(), 1e-6))
    return a.astype(np.uint8)


def _draw_keypoints(base: np.ndarray, pts: np.ndarray | list,
                    val: int = 255) -> np.ndarray:
    """Stamp small markers (5-pixel cross) at integer (x, y) keypoints
    on a grayscale base image.  Mirrors firmware ``marker()``."""
    out = base.copy()
    h, w = out.shape
    for p in pts:
        x, y = int(round(p[0])), int(round(p[1]))
        if 1 <= x < w - 1 and 1 <= y < h - 1:
            out[y, x] = val
            out[y, x - 1] = val
            out[y, x + 1] = val
            out[y - 1, x] = val
            out[y + 1, x] = val
    return out


# ---------------------------------------------------------------- basics

def passthrough(rgb):  return rgb.copy()

def grayscale(rgb):    return _gray_to_rgb(_to_gray(rgb))

def invert(rgb):       return (255 - rgb).astype(np.uint8)

def hist_eq(rgb):
    g = _to_gray(rgb)
    return _gray_to_rgb(cv2.equalizeHist(g))

def gaussian_3(rgb):   return cv2.GaussianBlur(rgb, (3, 3), 0)
def gaussian_5(rgb):   return cv2.GaussianBlur(rgb, (5, 5), 0)
def mean_3(rgb):       return cv2.blur(rgb, (3, 3))
def median_3(rgb):     return cv2.medianBlur(rgb, 3)
def bilateral(rgb):    return cv2.bilateralFilter(rgb, 5, 30, 30)

def sharpen(rgb):
    g = _to_gray(rgb)
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    out = cv2.filter2D(g, -1, k)
    return _gray_to_rgb(out)


# ---------------------------------------------------------------- edges

def _grad_xy(g, kx, ky):
    gx = cv2.filter2D(g, cv2.CV_32F, kx)
    gy = cv2.filter2D(g, cv2.CV_32F, ky)
    return _abs_to_u8(np.abs(gx) + np.abs(gy))

def sobel(rgb):
    g = _to_gray(rgb)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return _gray_to_rgb(_abs_to_u8(np.abs(gx) + np.abs(gy)))

def roberts(rgb):
    g = _to_gray(rgb).astype(np.float32)
    kx = np.array([[1, 0], [0, -1]], np.float32)
    ky = np.array([[0, 1], [-1, 0]], np.float32)
    return _gray_to_rgb(_grad_xy(g, kx, ky))

def prewitt(rgb):
    g = _to_gray(rgb).astype(np.float32)
    kx = np.array([[-1, 0, 1]] * 3, np.float32)
    ky = kx.T
    return _gray_to_rgb(_grad_xy(g, kx, ky))

def scharr(rgb):
    g = _to_gray(rgb)
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    return _gray_to_rgb(_abs_to_u8(np.abs(gx) + np.abs(gy)))

def kirsch(rgb):
    g = _to_gray(rgb).astype(np.float32)
    base = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], np.float32)
    out = np.zeros_like(g)
    for _ in range(8):
        out = np.maximum(out, cv2.filter2D(g, cv2.CV_32F, base))
        base = np.rot90(base)
    return _gray_to_rgb(_abs_to_u8(out))

def frei_chen(rgb):
    g = _to_gray(rgb).astype(np.float32)
    s = np.sqrt(2)
    h1 = np.array([[1, s, 1], [0, 0, 0], [-1, -s, -1]], np.float32) / (2 * s)
    h2 = np.array([[1, 0, -1], [s, 0, -s], [1, 0, -1]], np.float32) / (2 * s)
    e1 = cv2.filter2D(g, cv2.CV_32F, h1) ** 2
    e2 = cv2.filter2D(g, cv2.CV_32F, h2) ** 2
    return _gray_to_rgb(_abs_to_u8(np.sqrt(e1 + e2)))

def canny(rgb):
    g = _to_gray(rgb)
    e = cv2.Canny(g, 60, 150)
    return _gray_to_rgb(e)

def marr_hildreth(rgb):
    g = _to_gray(rgb)
    blur = cv2.GaussianBlur(g, (5, 5), 1.4)
    log = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    # zero-cross approximation: |LoG| magnitude
    return _gray_to_rgb(_abs_to_u8(log))

def laplacian(rgb):
    g = _to_gray(rgb)
    return _gray_to_rgb(_abs_to_u8(cv2.Laplacian(g, cv2.CV_32F, ksize=3)))


# ---------------------------------------------------------------- blobs

def dog(rgb):
    g = _to_gray(rgb).astype(np.float32)
    a = cv2.GaussianBlur(g, (3, 3), 0.8)
    b = cv2.GaussianBlur(g, (5, 5), 1.6)
    return _gray_to_rgb(_abs_to_u8(a - b))

def log(rgb):
    g = _to_gray(rgb)
    blur = cv2.GaussianBlur(g, (5, 5), 1.4)
    return _gray_to_rgb(_abs_to_u8(cv2.Laplacian(blur, cv2.CV_32F, ksize=3)))

def doh(rgb):
    g = _to_gray(rgb).astype(np.float32)
    Ixx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
    det = Ixx * Iyy - Ixy * Ixy
    return _gray_to_rgb(_abs_to_u8(det))

def blob_log_multiscale(rgb):
    g = _to_gray(rgb).astype(np.float32)
    a = cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0.8), cv2.CV_32F)
    b = cv2.Laplacian(cv2.GaussianBlur(g, (5, 5), 1.6), cv2.CV_32F)
    return _gray_to_rgb(_abs_to_u8(np.maximum(np.abs(a), np.abs(b))))


# ---------------------------------------------------------------- keypoints

def _kp_overlay(g, pts):
    base = (g >> 1).astype(np.uint8)  # mirror firmware "dimmed scene"
    return _gray_to_rgb(_draw_keypoints(base, pts))

def harris(rgb):
    g = _to_gray(rgb).astype(np.float32)
    r = cv2.cornerHarris(g, 2, 3, 0.04)
    ys, xs = np.where(r > 0.01 * r.max())
    return _kp_overlay(_to_gray(rgb), np.stack([xs, ys], axis=1))

def shi_tomasi(rgb):
    g = _to_gray(rgb)
    pts = cv2.goodFeaturesToTrack(g, maxCorners=200, qualityLevel=0.01,
                                  minDistance=3)
    pts = pts.reshape(-1, 2) if pts is not None else np.zeros((0, 2))
    return _kp_overlay(g, pts)

def _fast_kp(g, threshold=20, n_min=9):
    fast = cv2.FastFeatureDetector_create(threshold=threshold)
    fast.setNonmaxSuppression(True)
    kps = fast.detect(g, None)
    return np.array([[kp.pt[0], kp.pt[1]] for kp in kps]) if kps else np.zeros((0, 2))

def fast9(rgb):  return _kp_overlay(_to_gray(rgb), _fast_kp(_to_gray(rgb), 20))
def fast12(rgb): return _kp_overlay(_to_gray(rgb), _fast_kp(_to_gray(rgb), 35))
def agast(rgb):
    g = _to_gray(rgb)
    det = cv2.AgastFeatureDetector_create(threshold=20) if hasattr(cv2, "AgastFeatureDetector_create") else None
    if det is None:
        return fast9(rgb)
    kps = det.detect(g, None)
    pts = np.array([[kp.pt[0], kp.pt[1]] for kp in kps]) if kps else np.zeros((0, 2))
    return _kp_overlay(g, pts)

def brief(rgb):
    # detector + descriptor; for the visualisation we only need the
    # keypoints (descriptors are computed but discarded).
    g = _to_gray(rgb)
    kps = _fast_kp(g, 25)
    return _kp_overlay(g, kps)

def akaze(rgb):
    g = _to_gray(rgb)
    det = cv2.AKAZE_create() if hasattr(cv2, "AKAZE_create") else None
    if det is None:
        return harris(rgb)
    kps = det.detect(g, None)
    pts = np.array([[kp.pt[0], kp.pt[1]] for kp in kps]) if kps else np.zeros((0, 2))
    return _kp_overlay(g, pts)

def mser(rgb):
    g = _to_gray(rgb)
    det = cv2.MSER_create()
    regions, _ = det.detectRegions(g)
    out = np.zeros_like(g)
    for r in regions:
        for p in r:
            out[p[1], p[0]] = 255
    return _gray_to_rgb(out)


# ---------------------------------------------------------------- thresholding

def otsu(rgb):
    g = _to_gray(rgb)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _gray_to_rgb(m)

def adaptive_mean(rgb):
    g = _to_gray(rgb)
    m = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, 11, 2)
    return _gray_to_rgb(m)

def adaptive_gaussian(rgb):
    g = _to_gray(rgb)
    m = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 11, 2)
    return _gray_to_rgb(m)

def triangle(rgb):
    g = _to_gray(rgb)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return _gray_to_rgb(m)

def _local_stats(g, w=15):
    g32 = g.astype(np.float32)
    mu = cv2.boxFilter(g32, -1, (w, w))
    sq = cv2.boxFilter(g32 * g32, -1, (w, w))
    var = np.maximum(sq - mu * mu, 0)
    return mu, np.sqrt(var)

def niblack(rgb):
    g = _to_gray(rgb)
    mu, sd = _local_stats(g, 15)
    thr = mu - 0.2 * sd
    return _gray_to_rgb(((g.astype(np.float32) > thr) * 255).astype(np.uint8))

def sauvola(rgb):
    g = _to_gray(rgb)
    mu, sd = _local_stats(g, 15)
    thr = mu * (1 + 0.5 * (sd / 128 - 1))
    return _gray_to_rgb(((g.astype(np.float32) > thr) * 255).astype(np.uint8))


# ---------------------------------------------------------------- texture

def gabor(rgb):
    g = _to_gray(rgb).astype(np.float32)
    out = np.zeros_like(g)
    for theta in np.linspace(0, np.pi, 4, endpoint=False):
        k = cv2.getGaborKernel((9, 9), 2.0, theta, 4.0, 0.5, 0, ktype=cv2.CV_32F)
        out = np.maximum(out, np.abs(cv2.filter2D(g, cv2.CV_32F, k)))
    return _gray_to_rgb(_abs_to_u8(out))

def lbp(rgb):
    g = _to_gray(rgb).astype(np.int32)
    h, w = g.shape
    out = np.zeros_like(g, dtype=np.uint8)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
            (1, 1), (1, 0), (1, -1), (0, -1)]
    p = g[1:-1, 1:-1]
    code = np.zeros_like(p, dtype=np.uint8)
    for i, (dy, dx) in enumerate(offs):
        n = g[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        code |= ((n >= p).astype(np.uint8) << i)
    out[1:-1, 1:-1] = code
    return _gray_to_rgb(out)

def laws_energy(rgb):
    g = _to_gray(rgb).astype(np.float32)
    L5 = np.array([1, 4, 6, 4, 1], np.float32)
    E5 = np.array([-1, -2, 0, 2, 1], np.float32)
    k1 = np.outer(L5, E5); k2 = np.outer(E5, L5)
    e = np.abs(cv2.filter2D(g, cv2.CV_32F, k1)) + \
        np.abs(cv2.filter2D(g, cv2.CV_32F, k2))
    return _gray_to_rgb(_abs_to_u8(cv2.boxFilter(e, -1, (7, 7))))

def hog_vis(rgb):
    g = _to_gray(rgb)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return _gray_to_rgb(_abs_to_u8(mag))

def emboss(rgb):
    g = _to_gray(rgb).astype(np.float32)
    k = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32)
    out = cv2.filter2D(g, cv2.CV_32F, k) + 128
    return _gray_to_rgb(np.clip(out, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- ridge

def _hessian_eig(g, frangi=True):
    g = g.astype(np.float32)
    Ixx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
    tr = Ixx + Iyy
    disc = np.sqrt(np.maximum((Ixx - Iyy) ** 2 + 4 * Ixy * Ixy, 0))
    l1 = (tr + disc) / 2; l2 = (tr - disc) / 2
    if frangi:
        # Frangi vesselness: ratio l2/l1 modulated by structure norm
        rb = np.divide(l2, l1, out=np.zeros_like(l1), where=np.abs(l1) > 1e-3)
        s = np.sqrt(l1 ** 2 + l2 ** 2)
        v = np.exp(-rb ** 2 / 0.5) * (1 - np.exp(-s ** 2 / 50.0))
        return _abs_to_u8(v * 255)
    else:
        return _abs_to_u8(np.maximum(np.abs(l1), np.abs(l2)))

def frangi(rgb):
    return _gray_to_rgb(_hessian_eig(_to_gray(rgb), frangi=True))

def hessian_ridge(rgb):
    return _gray_to_rgb(_hessian_eig(_to_gray(rgb), frangi=False))


# ---------------------------------------------------------------- morphology

def _kernel():
    return cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

def erode(rgb):           return cv2.erode(rgb, _kernel())
def dilate(rgb):          return cv2.dilate(rgb, _kernel())
def open(rgb):            return cv2.morphologyEx(rgb, cv2.MORPH_OPEN, _kernel())
def close(rgb):           return cv2.morphologyEx(rgb, cv2.MORPH_CLOSE, _kernel())

def morph_gradient(rgb):
    return cv2.morphologyEx(rgb, cv2.MORPH_GRADIENT, _kernel())

def region_grow(rgb):
    g = _to_gray(rgb)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _gray_to_rgb(cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                                         cv2.getStructuringElement(
                                             cv2.MORPH_ELLIPSE, (5, 5))))

def watershed(rgb):
    g = _to_gray(rgb)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    edges = (dist > 0.5 * dist.max()).astype(np.uint8) * 255
    edges = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, _kernel())
    return _gray_to_rgb(edges)


# ---------------------------------------------------------------- registry

ALGO_FUNCS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "passthrough": passthrough, "grayscale": grayscale, "invert": invert,
    "hist_eq": hist_eq, "gaussian_3": gaussian_3, "gaussian_5": gaussian_5,
    "mean_3": mean_3, "median_3": median_3, "bilateral": bilateral,
    "sobel": sobel, "roberts": roberts, "prewitt": prewitt,
    "scharr": scharr, "kirsch": kirsch, "frei_chen": frei_chen,
    "canny": canny, "marr_hildreth": marr_hildreth, "laplacian": laplacian,
    "dog": dog, "log": log, "doh": doh,
    "harris": harris, "shi_tomasi": shi_tomasi, "fast9": fast9,
    "otsu": otsu, "adaptive_mean": adaptive_mean,
    "adaptive_gaussian": adaptive_gaussian, "triangle": triangle,
    "niblack": niblack, "sauvola": sauvola,
    "gabor": gabor, "lbp": lbp, "laws_energy": laws_energy,
    "frangi": frangi, "hessian_ridge": hessian_ridge, "hog_vis": hog_vis,
    "erode": erode, "dilate": dilate, "open": open, "close": close,
    "morph_gradient": morph_gradient, "region_grow": region_grow,
    "watershed": watershed, "sharpen": sharpen, "emboss": emboss,
    "mser": mser, "agast": agast, "brief": brief, "akaze": akaze,
    "blob_log_multiscale": blob_log_multiscale, "fast12": fast12,
}


def apply(algo_name: str, rgb: np.ndarray) -> np.ndarray:
    """Apply one of the 51 firmware algos to an RGB uint8 image.

    Returns a same-shape uint8 image.  Raises KeyError for unknown names.
    """
    fn = ALGO_FUNCS[algo_name]
    out = fn(rgb)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    if out.ndim == 2:
        out = _gray_to_rgb(out)
    return out
