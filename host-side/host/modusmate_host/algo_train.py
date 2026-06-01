"""Train a tiny per-algorithm classifier and benchmark every preprocessing.

Goal
----
Answer the question: *for each of the 51 firmware imgproc algorithms,
does running it as a preprocessing step in front of a small NN improve
classification accuracy compared to passing the raw image through?*

Method
------
For each algorithm we:

1. Apply the algorithm to every dataset image (CPU, on the laptop, using
   :mod:`modusmate_host.algo_lib`).
2. Time the per-image preprocessing cost.
3. Compute family-appropriate feature metrics (edge density, keypoint
   count, etc.) via :mod:`modusmate_host.algo_metrics`.
4. Train a small MLP classifier on the processed images.  Tiny on
   purpose: 1 hidden layer, ``--hidden`` units (default 64).  The point
   is *relative* comparison across preprocessings, not absolute
   accuracy.
5. Time NN inference per image.
6. Record train/test accuracy and write a CSV + Markdown summary.

The classifier is intentionally cheap (sklearn MLP on flattened
low-resolution images), inspired by the firmware's stump-NN benchmark:
deterministic, fast, reproducible.

Datasets
--------
Two built-in dataset sources, plus a directory layout for custom data:

* ``--dataset digits``        : sklearn ``load_digits`` upscaled to 32x32.
* ``--dataset shapes``        : on-the-fly synthetic 32x32 RGB shapes
                                 (circle / square / triangle / cross).
* ``--dataset DIR``           : ``DIR/<class_name>/*.png|jpg|bmp``.

Output
------
* ``--out-csv``      : one row per algo, columns:
                       algo, family, prep_us, infer_us, train_acc,
                       test_acc, edge_ratio, kp_count, binary_ratio,
                       mean_intensity.
* ``--out-md``       : Markdown table grouped by family, sorted by
                       test accuracy.

Usage
-----
    python -m modusmate_host.algo_train \
        --dataset shapes --samples 600 --epochs 8 \
        --out-csv results.csv --out-md summary.md

    python -m modusmate_host.algo_train --dataset path/to/imgs \
        --algos sobel,canny,passthrough --hidden 32 --epochs 4
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import algo_lib
from . import c_export
from .algos import ALGO_NAMES, FIRMWARE_ALGO_COUNT, family_of
from .algo_metrics import metrics_for

# sklearn imported lazily inside main() so that simply importing this
# module (e.g. for testing the dataset loaders) doesn't pull the full
# sklearn binary into memory.


# ---------------------------------------------------------------- datasets


def _today_stamp() -> str:
    return _dt.date.today().isoformat()


def _resize_rgb(img: np.ndarray, size: int) -> np.ndarray:
    import cv2
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _load_digits(size: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """sklearn ``load_digits`` (8x8 grayscale, 10 classes) upscaled."""
    from sklearn.datasets import load_digits
    raw = load_digits()
    imgs = []
    for x in raw.images:
        u8 = (x / x.max() * 255).astype(np.uint8)
        rgb = np.stack([u8] * 3, axis=-1)
        imgs.append(_resize_rgb(rgb, size))
    return (np.stack(imgs).astype(np.uint8),
            raw.target.astype(np.int64),
            [str(i) for i in range(10)])


def _make_shape(size: int, label: int, rng: np.random.Generator) -> np.ndarray:
    """Tiny synthetic generator: 4 shape classes on a noisy background."""
    import cv2
    img = (rng.integers(0, 50, size=(size, size, 3))).astype(np.uint8)
    cx = rng.integers(size // 4, 3 * size // 4)
    cy = rng.integers(size // 4, 3 * size // 4)
    r = rng.integers(size // 6, size // 3)
    color = tuple(int(c) for c in rng.integers(150, 255, size=3))
    if label == 0:    # circle
        cv2.circle(img, (int(cx), int(cy)), int(r), color, -1)
    elif label == 1:  # square
        cv2.rectangle(img, (int(cx - r), int(cy - r)),
                      (int(cx + r), int(cy + r)), color, -1)
    elif label == 2:  # triangle
        pts = np.array([[cx, cy - r], [cx - r, cy + r],
                        [cx + r, cy + r]], np.int32)
        cv2.fillPoly(img, [pts], color)
    else:             # cross
        cv2.line(img, (int(cx - r), int(cy)),
                 (int(cx + r), int(cy)), color, 2)
        cv2.line(img, (int(cx), int(cy - r)),
                 (int(cx), int(cy + r)), color, 2)
    return img


def _make_shapes_dataset(samples: int, size: int, seed: int = 0
                         ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    n_classes = 4
    per = samples // n_classes
    imgs, lbls = [], []
    for c in range(n_classes):
        for _ in range(per):
            imgs.append(_make_shape(size, c, rng))
            lbls.append(c)
    idx = rng.permutation(len(imgs))
    X = np.stack(imgs)[idx]
    y = np.array(lbls, dtype=np.int64)[idx]
    return X.astype(np.uint8), y, ["circle", "square", "triangle", "cross"]


def _make_silhouette(size: int, label: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic silhouette generator.

    3 classes:
      0 = human  — head circle + torso rectangle + 2 leg lines + 2 arm lines
      1 = box    — solid rectangle (non-human)
      2 = empty  — pure noisy background, no foreground subject
    """
    import cv2
    img = (rng.integers(20, 80, size=(size, size, 3))).astype(np.uint8)
    fg = tuple(int(c) for c in rng.integers(180, 255, size=3))
    cx = int(rng.integers(size * 0.4, size * 0.6))

    if label == 0:  # human silhouette
        head_r = max(2, int(size * float(rng.uniform(0.07, 0.10))))
        head_y = int(size * float(rng.uniform(0.18, 0.25)))
        torso_h = int(size * float(rng.uniform(0.30, 0.38)))
        torso_w = int(size * float(rng.uniform(0.14, 0.20)))
        torso_top = head_y + head_r
        torso_bot = torso_top + torso_h
        cv2.circle(img, (cx, head_y), head_r, fg, -1)
        cv2.rectangle(img,
                      (cx - torso_w // 2, torso_top),
                      (cx + torso_w // 2, torso_bot), fg, -1)
        leg_len = int(size * float(rng.uniform(0.20, 0.28)))
        leg_th = max(2, int(size * 0.04))
        leg_bot = min(size - 1, torso_bot + leg_len)
        cv2.line(img, (cx - torso_w // 4, torso_bot),
                 (cx - torso_w // 3, leg_bot), fg, leg_th)
        cv2.line(img, (cx + torso_w // 4, torso_bot),
                 (cx + torso_w // 3, leg_bot), fg, leg_th)
        arm_len = int(size * float(rng.uniform(0.18, 0.26)))
        arm_th = max(2, int(size * 0.035))
        arm_y = torso_top + int(torso_h * 0.15)
        arm_dx = int(size * float(rng.uniform(0.05, 0.15)))
        arm_dy = int(size * float(rng.uniform(0.05, 0.20)))
        cv2.line(img, (cx - torso_w // 2, arm_y),
                 (cx - torso_w // 2 - arm_dx, arm_y + arm_dy), fg, arm_th)
        cv2.line(img, (cx + torso_w // 2, arm_y),
                 (cx + torso_w // 2 + arm_dx, arm_y + arm_dy), fg, arm_th)
    elif label == 1:  # box (non-human rectangle)
        bw = int(size * float(rng.uniform(0.30, 0.55)))
        bh = int(size * float(rng.uniform(0.30, 0.55)))
        bx = int(rng.integers(size * 0.1, size * 0.9 - bw))
        by = int(rng.integers(size * 0.1, size * 0.9 - bh))
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), fg, -1)
    # label == 2 -> empty: background noise only
    return img


def _make_silhouette_dataset(samples: int, size: int, seed: int = 0
                             ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    classes = ["human", "box", "empty"]
    n_classes = len(classes)
    per = samples // n_classes
    imgs, lbls = [], []
    for c in range(n_classes):
        for _ in range(per):
            imgs.append(_make_silhouette(size, c, rng))
            lbls.append(c)
    idx = rng.permutation(len(imgs))
    X = np.stack(imgs)[idx]
    y = np.array(lbls, dtype=np.int64)[idx]
    return X.astype(np.uint8), y, classes


def _load_directory(root: Path, size: int
                    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load ``root/<class>/*.{png,jpg,bmp}`` into memory at ``size x size``."""
    import cv2
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit(f"no class subdirectories in {root}")
    imgs, lbls = [], []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    for ci, cn in enumerate(classes):
        for f in sorted((root / cn).iterdir()):
            if f.suffix.lower() not in exts:
                continue
            bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            imgs.append(_resize_rgb(rgb, size))
            lbls.append(ci)
    if not imgs:
        raise SystemExit(f"no images found under {root}/<class>/")
    return (np.stack(imgs).astype(np.uint8),
            np.array(lbls, dtype=np.int64),
            classes)


# ---------------------------------------------------------------- kaggle loader

# Friendly aliases -> (kagglehub_slug, default class list to keep).
# Real photographic / rendered data; superior to the synthetic shapes
# and silhouette generators for actually differentiating preprocessing
# algorithms.
_KAGGLE_PRESETS: dict = {
    # ~16k 200x200 grayscale shape images (4 classes: circle, square,
    # triangle, star). Real rendered shapes with rotation/scale/noise.
    "four_shapes": ("smeschke/four-shapes",
                    ["circle", "square", "triangle"]),
    # ~2.2k photos. Folders are literally "0" and "1"; we rename to
    # empty / person.
    "people": ("constantinwerner/human-detection-dataset",
               ["empty", "person"]),
    # ~6.9k images, 8 classes (airplane, car, cat, dog, flower, fruit,
    # motorbike, person). Pair with --dataset-classes to pick a subset.
    "natural_images": ("prasunroy/natural-images", None),
    # 2.1k RPS hand photos (same source as the bench loader).
    "rps": ("drgfreeman/rockpaperscissors",
            ["rock", "paper", "scissors"]),
}


def _scan_kaggle_root(root: Path, size: int,
                      keep: Optional[Sequence[str]] = None
                      ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Walk a kagglehub-extracted directory tree and pull images grouped
    by their nearest class-named parent directory. ``keep`` filters the
    discovered class set; if the dataset uses numeric folder names like
    ``0``/``1`` we map them positionally onto ``keep`` (so the
    human-detection-dataset becomes empty/person).
    """
    import cv2
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    class_dirs: dict = {}  # class_name (lower) -> list[Path]
    for dirpath, _d, fns in os.walk(root, followlinks=True):
        files = [Path(dirpath) / n for n in fns
                 if Path(n).suffix.lower() in exts]
        if not files:
            continue
        cls = Path(dirpath).name.lower()
        if not cls or cls.startswith("."):
            continue
        class_dirs.setdefault(cls, []).extend(files)
    if not class_dirs:
        raise SystemExit(f"no images found under {root}")

    rename: dict = {}
    if keep:
        keep_l = [k.lower() for k in keep]
        if all(k in class_dirs for k in keep_l):
            classes = list(keep)
        else:
            avail = sorted(class_dirs.keys())
            if len(avail) < len(keep):
                raise SystemExit(
                    f"dataset has {len(avail)} class dir(s); --classes "
                    f"wants {len(keep)} ({list(keep)})")
            for src, dst in zip(avail, keep):
                rename[src] = dst
            classes = list(keep)
    else:
        classes = sorted(class_dirs.keys())

    name_to_id = {c: i for i, c in enumerate(classes)}
    imgs, lbls = [], []
    for src, files in class_dirs.items():
        cls = rename.get(src, src)
        if cls not in name_to_id:
            continue
        cid = name_to_id[cls]
        for f in files:
            bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            imgs.append(_resize_rgb(rgb, size))
            lbls.append(cid)
    if not imgs:
        raise SystemExit(f"matched 0 images for classes {classes}")
    return (np.stack(imgs).astype(np.uint8),
            np.array(lbls, dtype=np.int64), classes)


def _load_kaggle(slug: str, size: int,
                 keep: Optional[Sequence[str]] = None
                 ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Download a Kaggle dataset (cached via kagglehub) then scan it."""
    cache = Path.home() / ".modusmate" / "datasets" / slug.replace("/", "_")
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache / ".source"
    if marker.exists():
        src = Path(marker.read_text().strip())
    else:
        try:
            import kagglehub  # type: ignore
        except ImportError as e:
            raise SystemExit(
                f"--dataset {slug} needs kagglehub: pip install kagglehub "
                f"({e})") from e
        print(f"[modusmate] downloading {slug} via kagglehub...")
        src = Path(kagglehub.dataset_download(slug))
        marker.write_text(str(src))
    return _scan_kaggle_root(src, size, keep=keep)


def load_dataset(spec: str, samples: int, size: int,
                 classes: Optional[Sequence[str]] = None
                 ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Resolve the ``--dataset`` argument to (images, labels, class_names).

    ``classes`` (optional) overrides the preset's default class filter.
    """
    if spec == "digits":
        return _load_digits(size)
    if spec == "shapes":
        return _make_shapes_dataset(samples, size)
    if spec in ("silhouette", "silhouettes", "humans"):
        return _make_silhouette_dataset(samples, size)
    if spec in _KAGGLE_PRESETS:
        slug, default_keep = _KAGGLE_PRESETS[spec]
        return _load_kaggle(slug, size, keep=classes or default_keep)
    if spec.startswith("kaggle:"):
        return _load_kaggle(spec[len("kaggle:"):], size, keep=classes)
    p = Path(spec)
    if p.is_dir():
        return _load_directory(p, size)
    raise SystemExit(
        f"--dataset '{spec}' not understood; expected 'digits', 'shapes', "
        "'silhouette', a preset ({}) or 'kaggle:<owner/slug>', "
        "or a directory with class-named subfolders".format(
            ', '.join(sorted(_KAGGLE_PRESETS))))


# ---------------------------------------------------------------- augmentation


def _augment_train(X: np.ndarray, y: np.ndarray, modes: Sequence[str],
                   factor: int, seed: int
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Augment a uint8 image batch ``factor``-fold using the requested modes.

    Each augmented copy applies a random subset of ``modes`` per image.
    Originals are always kept, so ``factor=1`` returns the input unchanged.

    Modes: ``flip`` (horizontal), ``rotate`` (±15°), ``scale`` (0.85–1.15x),
    ``shift`` (±15% translation), ``brightness`` (±15%), ``noise``
    (gaussian σ=8 on uint8).
    """
    if factor <= 1 or not modes:
        return X.astype(np.uint8), y
    import cv2
    rng = np.random.default_rng(seed)
    out_imgs = [X.astype(np.uint8)]
    out_lbls = [y]
    h, w = X.shape[1], X.shape[2]
    for _ in range(factor - 1):
        batch = X.copy()
        for i in range(batch.shape[0]):
            img = batch[i]
            if "flip" in modes and rng.random() < 0.5:
                img = img[:, ::-1, :]
            if "rotate" in modes:
                ang = float(rng.uniform(-15.0, 15.0))
                M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
                img = cv2.warpAffine(img, M, (w, h),
                                     borderMode=cv2.BORDER_REFLECT_101)
            if "scale" in modes:
                s = float(rng.uniform(0.85, 1.15))
                M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, s)
                img = cv2.warpAffine(img, M, (w, h),
                                     borderMode=cv2.BORDER_REFLECT_101)
            if "shift" in modes:
                tx = int(rng.uniform(-0.15, 0.15) * w)
                ty = int(rng.uniform(-0.15, 0.15) * h)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                img = cv2.warpAffine(img, M, (w, h),
                                     borderMode=cv2.BORDER_REFLECT_101)
            if "brightness" in modes:
                k = float(rng.uniform(0.85, 1.15))
                img = np.clip(img.astype(np.float32) * k, 0, 255).astype(
                    np.uint8)
            if "noise" in modes:
                noise = rng.normal(0.0, 8.0, size=img.shape)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(
                    np.uint8)
            batch[i] = img
        out_imgs.append(batch)
        out_lbls.append(y.copy())
    return np.concatenate(out_imgs, axis=0), np.concatenate(out_lbls, axis=0)


def _parse_augment(spec: str) -> List[str]:
    if not spec or spec.lower() in ("none", "off"):
        return []
    s = spec.lower().strip()
    if s == "all":
        return ["flip", "rotate", "scale", "shift", "brightness", "noise"]
    if s == "basic":
        return ["flip", "rotate", "shift"]
    return [t.strip() for t in s.split(",") if t.strip()]


def _simulate_board_geometry(X: np.ndarray, side: int,
                             camera_w: int = 320,
                             camera_h: int = 240,
                             image_dim: int = 320) -> np.ndarray:
    """[deprecated] Reproduce the board's geometry collapse on (N,S,S,3)
    images by upsampling, zero-padding and area-down-sampling. Kept for
    callers that still want a single-tensor preview; the training path
    now uses :func:`_to_board_resolution` + :func:`_process_all_board`
    so preprocessing runs at the same 320x320 the firmware uses.
    """
    import cv2
    if X.ndim != 4 or X.shape[1] != side or X.shape[2] != side:
        return X
    block = image_dim // side
    if block < 1:
        return X
    valid_rows = (camera_h * side) // image_dim
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        big = cv2.resize(X[i], (camera_w, camera_h),
                         interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((image_dim, image_dim, 3), dtype=np.uint8)
        padded[:camera_h, :camera_w, :] = big
        small = cv2.resize(padded, (side, side),
                           interpolation=cv2.INTER_AREA)
        small[valid_rows:, :, :] = 0
        out[i] = small
    return out


# Board image dimensions (must match firmware: CAMERA_WIDTH/HEIGHT and
# IMAGE_WIDTH/HEIGHT in inference_task.c, plus the 320x320 buffer that
# c_export.downsample() consumes).
_BOARD_CAMERA_W = 320
_BOARD_CAMERA_H = 240
_BOARD_IMAGE_DIM = 320


def _process_all_board(algo: str, X: np.ndarray,
                       width: int, height: int
                       ) -> Tuple[np.ndarray, float, dict]:
    """Apply preprocessing at 320x320 then area-down to ``height x width``.

    Mirrors the board pipeline:
      ``imgproc_apply(algo, frame, 320, 320, 240)`` (320x320 padded, top
      240 rows real, bottom 80 rows zero) followed by ``downsample()``
      from model.c (area-average 320 -> width / 320 -> height).

    ``X`` is the original (N, S, S, 3) tensor — we upsample every
    image to 320x320 on the fly so we don't have to materialise the
    full ~3GB intermediate batch.
    Returns (X_small, mean_us_per_image, mean_metrics).
    """
    import cv2
    n = X.shape[0]
    out = np.empty((n, height, width, 3), dtype=np.uint8)
    metric_acc = {"edge_ratio": 0.0, "kp_count": 0.0,
                  "binary_ratio": 0.0, "mean_intensity": 0.0}
    big = np.zeros((_BOARD_IMAGE_DIM, _BOARD_IMAGE_DIM, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    for i in range(n):
        # 1. upsample S x S -> 320 x 240 and zero-pad bottom 80 rows
        cam = cv2.resize(X[i], (_BOARD_CAMERA_W, _BOARD_CAMERA_H),
                         interpolation=cv2.INTER_LINEAR)
        big.fill(0)
        big[:_BOARD_CAMERA_H, :_BOARD_CAMERA_W, :] = cam
        # 2. preprocessing at 320x320 (matches firmware imgproc_apply)
        proc = algo_lib.apply(algo, big)
        # 3. area-down to width x height (matches model.c downsample())
        out[i] = cv2.resize(proc, (width, height),
                            interpolation=cv2.INTER_AREA)
    elapsed = time.perf_counter() - t0
    sample = out[: min(64, n)]
    for img in sample:
        m = metrics_for(algo, img)
        for k in metric_acc:
            metric_acc[k] += m[k]
    denom = max(len(sample), 1)
    for k in metric_acc:
        metric_acc[k] /= denom
    return out, (elapsed / max(n, 1)) * 1e6, metric_acc



def _save_test_split(out_dir: Path, X: np.ndarray, y: np.ndarray,
                     classes: Sequence[str], log=print,
                     firmware_perm: Optional[Sequence[int]] = None) -> None:
    """Write held-out test images to ``<out_dir>/<class>/<i>.png`` so the
    on-board sweep can use a clean test set the model never saw.

    Also writes ``classes.json`` (the class list in trained-model order)
    so :func:`modusmate_host.dataset.load_samples` reproduces the exact
    same class-id assignment the model uses.

    The firmware applies a class-id permutation
    ``model_to_dataset[N_CLASSES]`` to every BENCH_RESULT before sending
    it back, so ``res.class_id == s.label_id`` is only true if the
    saved label_ids match that permutation. Pass ``firmware_perm`` (a
    list of length N where ``perm[m]`` = the class_id the firmware will
    report for trained-model output ``m``) to remap; default identity.

    Each image is also up-sampled to 320x240 RGB888 because the firmware
    ``BENCH_BEGIN/CHUNK/END`` protocol expects that exact frame size; the
    on-board model.c then area-averages it back down to its native
    ``side`` for inference.
    """
    import cv2
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(classes)
    if firmware_perm is None:
        firmware_perm = list(range(n_classes))
    else:
        firmware_perm = list(firmware_perm)
        if len(firmware_perm) != n_classes:
            raise ValueError(
                f"firmware_perm len {len(firmware_perm)} != "
                f"n_classes {n_classes}")
    # Re-arrange the class list so that classes[label_id] picks the class
    # whose image will produce that firmware-reported label_id. I.e.
    # saved_classes_in_order[firmware_perm[m]] == classes[m].
    saved_classes: List[str] = [""] * n_classes
    for m, c in enumerate(classes):
        saved_classes[int(firmware_perm[m])] = c
    for c in saved_classes:
        (out_dir / c).mkdir(exist_ok=True)
    counters: dict = {c: 0 for c in saved_classes}
    for i in range(X.shape[0]):
        cls = classes[int(y[i])]
        img = X[i]
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bgr_320 = cv2.resize(bgr, (320, 240), interpolation=cv2.INTER_LINEAR)
        idx = counters[cls]
        counters[cls] += 1
        cv2.imwrite(str(out_dir / cls / f"{idx:05d}.png"), bgr_320)
    (out_dir / "classes.json").write_text(
        json.dumps({"classes": saved_classes,
                    "trained_classes": list(classes),
                    "firmware_perm": list(firmware_perm),
                    "size": int(X.shape[1])}, indent=2))
    log(f"saved {int(X.shape[0])} held-out test images -> {out_dir} "
        f"({n_classes} classes, firmware_perm={firmware_perm})")


# ---------------------------------------------------------------- pipeline


@dataclass
class AlgoResult:
    algo: str
    family: str
    prep_us: float
    infer_us: float
    train_acc: float
    test_acc: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    per_class_f1: str   # "clsA=0.83;clsB=0.71;..."
    confusion: str      # "3,1,0|0,4,1|2,0,3" rows = true, cols = pred
    edge_ratio: float
    kp_count: float
    binary_ratio: float
    mean_intensity: float
    n_train: int
    n_test: int


def _process_all(algo: str, X: np.ndarray
                 ) -> Tuple[np.ndarray, float, dict]:
    """Apply ``algo`` to every image in X.  Returns (X_out, mean_us, mean_metrics)."""
    out = np.empty_like(X)
    metric_acc = {"edge_ratio": 0.0, "kp_count": 0.0,
                  "binary_ratio": 0.0, "mean_intensity": 0.0}
    t0 = time.perf_counter()
    for i in range(X.shape[0]):
        out[i] = algo_lib.apply(algo, X[i])
    elapsed = time.perf_counter() - t0
    # metrics on a sample (limit to 64 to keep the loop fast)
    sample = out[: min(64, out.shape[0])]
    for img in sample:
        m = metrics_for(algo, img)
        for k in metric_acc:
            metric_acc[k] += m[k]
    n = max(len(sample), 1)
    for k in metric_acc:
        metric_acc[k] /= n
    return out, (elapsed / max(X.shape[0], 1)) * 1e6, metric_acc


def _train_and_eval(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    hidden: int, epochs: int, seed: int,
                    classes: Sequence[str],
                    alpha: float = 1e-4,
                    early_stopping: bool = True,
                    n_iter_no_change: int = 12,
                    validation_fraction: float = 0.1,
                    ) -> Tuple[float, float, float, object, dict]:
    """Tiny MLP on flattened images.

    Returns ``(train_acc, test_acc, infer_us, clf, scores)`` where ``scores``
    is a dict of precision/recall/F1 (macro & weighted) plus per-class F1
    and the confusion matrix.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import (precision_recall_fscore_support,
                                 confusion_matrix)
    Xt = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
    Xv = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255.0
    es = bool(early_stopping)
    nicc = int(n_iter_no_change) if es else int(epochs)
    clf = MLPClassifier(hidden_layer_sizes=(hidden,),
                        max_iter=epochs,
                        solver="adam", learning_rate_init=1e-3,
                        alpha=float(alpha),
                        random_state=seed,
                        early_stopping=es,
                        validation_fraction=validation_fraction,
                        n_iter_no_change=nicc,
                        verbose=False)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            clf.fit(Xt, y_train)
        except ValueError:
            # tiny / class-imbalanced batches can break early-stopping's
            # internal validation split; fall back without it.
            clf = MLPClassifier(hidden_layer_sizes=(hidden,),
                                max_iter=epochs,
                                solver="adam", learning_rate_init=1e-3,
                                alpha=float(alpha),
                                random_state=seed,
                                early_stopping=False,
                                n_iter_no_change=int(epochs),
                                verbose=False)
            clf.fit(Xt, y_train)
    train_acc = float(clf.score(Xt, y_train))
    test_acc = float(clf.score(Xv, y_test))

    # full metric set on the test split
    y_pred = clf.predict(Xv)
    label_ids = list(range(len(classes)))
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=label_ids, average=None, zero_division=0)
    pr_m, rc_m, f1_m, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=label_ids, average="macro", zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=label_ids, average="weighted",
        zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=label_ids)
    per_class_f1 = ";".join(f"{classes[i]}={f1[i]:.3f}"
                            for i in range(len(classes)))
    confusion = "|".join(",".join(str(int(v)) for v in row) for row in cm)
    scores = {
        "precision_macro": float(pr_m),
        "recall_macro": float(rc_m),
        "f1_macro": float(f1_m),
        "f1_weighted": float(f1_w),
        "per_class_f1": per_class_f1,
        "confusion": confusion,
    }
    # time inference per sample
    n_timing = min(64, Xv.shape[0])
    if n_timing == 0:
        infer_us = 0.0
    else:
        t0 = time.perf_counter()
        clf.predict(Xv[:n_timing])
        infer_us = (time.perf_counter() - t0) / n_timing * 1e6
    return train_acc, test_acc, infer_us, clf, scores


def run(algos: Sequence[str], X: np.ndarray, y: np.ndarray,
        test_frac: float, hidden: int, epochs: int, seed: int,
        export_dir: Optional[Path] = None,
        classes: Optional[Sequence[str]] = None,
        dataset_name: str = "dataset",
        size: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        alpha: float = 1e-4,
        early_stopping: bool = True,
        patience: int = 12,
        augment_modes: Optional[Sequence[str]] = None,
        augment_factor: int = 1,
        save_test_dir: Optional[Path] = None,
        firmware_class_perm: Optional[Sequence[int]] = None,
        log=print) -> List[AlgoResult]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    if size is None:
        size = int(X.shape[1])
    if width is None:
        width = int(size)
    if height is None:
        height = int(size)
    if classes is None:
        classes = [str(c) for c in sorted(set(y.tolist()))]

    # Apply augmentation only to the training split so the test split
    # remains a clean unseen-data check.
    X_train_full = X[train_idx]
    y_train_full = y[train_idx]
    X_test_full = X[test_idx]
    y_test_full = y[test_idx]
    if augment_modes and augment_factor > 1:
        X_train_full, y_train_full = _augment_train(
            X_train_full, y_train_full, list(augment_modes),
            int(augment_factor), seed)
        log(f"augment: modes={list(augment_modes)} ×{augment_factor}  "
            f"train={X_train_full.shape[0]} test={X_test_full.shape[0]}")

    # Preprocessing must run at the firmware's 320x320 resolution
    # (filter responses scale with image size — Sobel etc on 32x32 is
    # nothing like Sobel on 320x320 after downsampling). The per-algo
    # loop below uses :func:`_process_all_board`, which upsamples each
    # image on the fly, so we don't materialise a giant ~3 GB tensor.
    log(f"board-resolution preprocessing enabled "
        f"(upsample {size}x{size} -> {_BOARD_IMAGE_DIM}x"
        f"{_BOARD_IMAGE_DIM}, top {_BOARD_CAMERA_H} rows real, "
        f"bottom {_BOARD_IMAGE_DIM - _BOARD_CAMERA_H} rows zero-padded; "
        f"model input {width}x{height}x3)")

    if save_test_dir is not None:
        # Save the *original* unsimulated test images; the dataset loader
        # in benchmark.py will resize them to 320x240 itself, matching the
        # firmware bench protocol exactly.
        _save_test_split(save_test_dir, X_test_full, y_test_full,
                         list(classes), log=log,
                         firmware_perm=firmware_class_perm)

    results: List[AlgoResult] = []
    for i, name in enumerate(algos):
        log(f"[{i+1}/{len(algos)}] {name:<22s} ...", end="", flush=True)
        try:
            X_train_proc, prep_us_tr, _ = _process_all_board(
                name, X_train_full, width=width, height=height)
            X_test_proc, prep_us_te, metrics = _process_all_board(
                name, X_test_full, width=width, height=height)
            prep_us = (prep_us_tr * X_train_full.shape[0]
                       + prep_us_te * X_test_full.shape[0]) / max(
                X_train_full.shape[0] + X_test_full.shape[0], 1)
            tr_acc, te_acc, inf_us, clf, scores = _train_and_eval(
                X_train_proc, y_train_full,
                X_test_proc, y_test_full,
                hidden=hidden, epochs=epochs, seed=seed,
                classes=classes,
                alpha=alpha, early_stopping=early_stopping,
                n_iter_no_change=patience)
        except Exception as e:
            log(f" FAILED ({e})")
            continue
        r = AlgoResult(
            algo=name, family=family_of(name),
            prep_us=prep_us, infer_us=inf_us,
            train_acc=tr_acc, test_acc=te_acc,
            precision_macro=scores["precision_macro"],
            recall_macro=scores["recall_macro"],
            f1_macro=scores["f1_macro"],
            f1_weighted=scores["f1_weighted"],
            per_class_f1=scores["per_class_f1"],
            confusion=scores["confusion"],
            edge_ratio=metrics["edge_ratio"],
            kp_count=metrics["kp_count"],
            binary_ratio=metrics["binary_ratio"],
            mean_intensity=metrics["mean_intensity"],
            n_train=int(X_train_full.shape[0]),
            n_test=int(X_test_full.shape[0]))
        results.append(r)
        log(f" prep={prep_us:7.1f}us  nn={inf_us:6.1f}us  "
            f"acc={te_acc:.3f}  f1m={scores['f1_macro']:.3f}  "
            f"P={scores['precision_macro']:.3f}  "
            f"R={scores['recall_macro']:.3f}")
        if export_dir is not None:
            try:
                mlp = c_export.ExportedMLP.from_sklearn(
                    clf, side=size, classes=list(classes),
                    algo_name=name, dataset=dataset_name,
                    model_w=width, model_h=height)
                stamp = _today_stamp()
                target = export_dir / f"{name}_{dataset_name}_{stamp}"
                c_export.write_imai_model(
                    mlp, target, test_acc=te_acc, prep_us=prep_us)
                log(f"          + exported {target.relative_to(export_dir.parent) if export_dir.parent in target.parents else target}")
            except Exception as e:
                log(f"          ! export failed: {e}")
    return results


def write_csv(results: Sequence[AlgoResult], path: Path) -> None:
    if not results:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


def write_markdown(results: Sequence[AlgoResult], path: Path,
                   dataset: str, n_total: int, n_classes: int) -> None:
    by_family: dict = {}
    for r in results:
        by_family.setdefault(r.family, []).append(r)
    baseline = next((r for r in results if r.algo == "passthrough"), None)
    lines = [
        f"# ModusMate per-algo NN benchmark", "",
        f"- Dataset: `{dataset}` ({n_total} samples, {n_classes} classes)",
        f"- Total algos: {len(results)}",
        ""]
    if baseline is not None:
        lines.append(f"**Baseline (passthrough)**: test_acc = "
                     f"`{baseline.test_acc:.3f}`")
        lines.append("")
    lines.append("Sorted within each family by test accuracy (best first).")
    lines.append("")
    for fam in sorted(by_family.keys()):
        lines.append(f"## {fam}")
        lines.append("")
        lines.append("| algo | test_acc | Δ vs passthrough | F1 macro | "
                     "F1 weighted | precision | recall | prep µs | nn µs |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(by_family[fam], key=lambda x: -x.f1_macro):
            delta = (r.test_acc - baseline.test_acc) if baseline else 0.0
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"| `{r.algo}` | {r.test_acc:.3f} | "
                f"{sign}{delta:+.3f} | {r.f1_macro:.3f} | "
                f"{r.f1_weighted:.3f} | {r.precision_macro:.3f} | "
                f"{r.recall_macro:.3f} | "
                f"{r.prep_us:.0f} | {r.infer_us:.0f} |")
        lines.append("")
    # Top-10 F1-macro leaderboard on its own.
    top = sorted(results, key=lambda x: -x.f1_macro)[:10]
    if top:
        lines.append("## Top 10 by F1 (macro)")
        lines.append("")
        lines.append("| rank | algo | family | F1 macro | F1 weighted | "
                     "accuracy | per-class F1 |")
        lines.append("|---:|---|---|---:|---:|---:|---|")
        for i, r in enumerate(top, 1):
            lines.append(
                f"| {i} | `{r.algo}` | {r.family} | "
                f"{r.f1_macro:.3f} | {r.f1_weighted:.3f} | "
                f"{r.test_acc:.3f} | `{r.per_class_f1}` |")
        lines.append("")
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------- CLI


def _parse_algos(spec: Optional[str]) -> List[str]:
    if not spec or spec == "all":
        return list(ALGO_NAMES[:FIRMWARE_ALGO_COUNT])
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in ALGO_NAMES:
            out.append(tok)
        else:
            raise SystemExit(f"unknown algo: {tok!r}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="modusmate_host.algo_train",
        description="Per-algo preprocessing-vs-NN-accuracy benchmark.")
    p.add_argument("--dataset", default="shapes",
                   help="'digits', 'shapes', or path to a class-folder dir")
    p.add_argument("--samples", type=int, default=400,
                   help="(shapes only) total synthetic samples to generate")
    p.add_argument("--size", type=int, default=32,
                   help="image side length the algos and NN see (default 32)")
    p.add_argument("--width", type=int, default=None,
                   help="model input width (defaults to --size). Use 320 with "
                        "--height 240 to feed the NN the native camera frame.")
    p.add_argument("--height", type=int, default=None,
                   help="model input height (defaults to --size).")
    p.add_argument("--algos", default="all",
                   help="comma-separated algo names, or 'all'")
    p.add_argument("--hidden", type=int, default=64,
                   help="MLP hidden units (default 64)")
    p.add_argument("--epochs", type=int, default=8,
                   help="MLP training epochs (max_iter)")
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1e-4,
                   help="L2 regularisation strength (default 1e-4)")
    p.add_argument("--early-stopping", dest="early_stopping",
                   action="store_true", default=True,
                   help="enable sklearn early stopping (default on)")
    p.add_argument("--no-early-stopping", dest="early_stopping",
                   action="store_false")
    p.add_argument("--patience", type=int, default=12,
                   help="early-stop patience in epochs (default 12)")
    p.add_argument("--augment", default="none",
                   help="'none', 'basic' (flip+rotate+shift), 'all', or "
                        "comma-separated of: flip,rotate,scale,shift,"
                        "brightness,noise")
    p.add_argument("--augment-factor", type=int, default=1,
                   help="multiply training set size N× via augmentation "
                        "(default 1 = off)")
    p.add_argument("--restrict-classes", type=int, default=0,
                   help="keep only the first N class IDs (drops samples "
                        "whose label_id >= N). Use this to make a 4-class "
                        "dataset like 'shapes' fit the firmware's 3-lane "
                        "output. 0 = keep all (default).")
    p.add_argument("--dataset-classes", default=None,
                   help="comma-separated class names to keep when using a "
                        "preset / kaggle:<slug> dataset (e.g. "
                        "'person,empty' for --dataset people, or "
                        "'circle,square,triangle' for --dataset four_shapes)")
    p.add_argument("--save-test-dir", default=None,
                   help="dump the held-out test split as PNGs under "
                        "<DIR>/<class>/<i>.png so the on-board sweep can "
                        "evaluate on data the model never saw. Also writes "
                        "<DIR>/classes.json with the trained-model class "
                        "order. Pass this dir to `sweep --test-dir`.")
    p.add_argument("--firmware-class-perm", default=None,
                   help="comma-separated permutation matching the firmware's "
                        "model_to_dataset[] mapping (e.g. '2,1,0' for the "
                        "current PSE84 RPS firmware). The on-board class_id "
                        "that the bench protocol returns is `perm[model_argmax]`. "
                        "--save-test-dir will arrange classes.json so that "
                        "the firmware-reported class_id matches the saved "
                        "label_id, making `res.class_id == s.label_id` correct. "
                        "Default: identity.")
    p.add_argument("--out-csv", default="algo_train_results.csv")
    p.add_argument("--out-md", default="algo_train_summary.md")
    p.add_argument("--export-models", default=None,
                   help="if set, write per-algo IMAI-compatible model.h/.c + "
                        "manifest.json under <DIR>/<algo>_<dataset>_<date>/. "
                        "Use 'auto' for the repo-root models/ directory.")
    args = p.parse_args(argv)

    algos = _parse_algos(args.algos)
    print(f"loading dataset '{args.dataset}'...")
    classes_filter = ([c.strip() for c in args.dataset_classes.split(",")]
                      if args.dataset_classes else None)
    X, y, classes = load_dataset(args.dataset, args.samples, args.size,
                                 classes=classes_filter)
    print(f"  -> {X.shape[0]} samples, {len(classes)} classes "
          f"({','.join(classes)}), shape={X.shape[1:]}")

    if args.restrict_classes and args.restrict_classes > 0:
        n_keep = int(args.restrict_classes)
        if n_keep < len(classes):
            keep_mask = y < n_keep
            X = X[keep_mask]
            y = y[keep_mask]
            classes = list(classes)[:n_keep]
            print(f"  restrict-classes: kept {n_keep} -> "
                  f"{X.shape[0]} samples, classes={classes}")
    print(f"running {len(algos)} algos "
          f"(hidden={args.hidden}, epochs={args.epochs})")

    export_dir: Optional[Path] = None
    if args.export_models:
        if args.export_models == "auto":
            export_dir = Path(__file__).resolve().parents[2] / "models"
        else:
            export_dir = Path(args.export_models)
        export_dir.mkdir(parents=True, exist_ok=True)
        print(f"exporting per-algo IMAI models to {export_dir}")
        if len(classes) > c_export.FIRMWARE_MAX_CLASSES:
            print(f"  WARNING: dataset has {len(classes)} classes but "
                  f"firmware decodes only {c_export.FIRMWARE_MAX_CLASSES}; "
                  f"extra class probabilities will be dropped on-board.")

    dataset_name = args.dataset if args.dataset in ("shapes", "digits") \
        else Path(args.dataset).name.replace(" ", "_")

    results = run(algos, X, y,
                  test_frac=args.test_frac, hidden=args.hidden,
                  epochs=args.epochs, seed=args.seed,
                  export_dir=export_dir, classes=classes,
                  dataset_name=dataset_name, size=args.size,
                  width=args.width, height=args.height,
                  alpha=args.alpha,
                  early_stopping=args.early_stopping,
                  patience=args.patience,
                  augment_modes=_parse_augment(args.augment),
                  augment_factor=int(args.augment_factor),
                  save_test_dir=Path(args.save_test_dir).expanduser()
                  if args.save_test_dir else None,
                  firmware_class_perm=(
                      [int(x) for x in args.firmware_class_perm.split(",")]
                      if args.firmware_class_perm else None))

    csv_path = Path(args.out_csv)
    md_path = Path(args.out_md)
    write_csv(results, csv_path)
    write_markdown(results, md_path, args.dataset, X.shape[0], len(classes))
    print(f"\nwrote {csv_path}  +  {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
