"""Kaggle Rock-Paper-Scissors dataset loader for the benchmark.

Uses kagglehub to fetch the public 'drgfreeman/rockpaperscissors' dataset
(no auth needed). Caches under ~/.modusmate/datasets/rps/.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image

# Class label mapping must match the firmware's dataset class IDs.
# Firmware: model_to_dataset[] = {2,1,0} -> dataset 0=Rock, 1=Paper, 2=Scissors
LABEL_TO_ID = {"rock": 0, "paper": 1, "scissors": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

CACHE_DIR = Path.home() / ".modusmate" / "datasets" / "rps"


@dataclass
class Sample:
    path: Path
    label: str
    label_id: int


def _scan_dataset(root: Path) -> List[Sample]:
    """Find images organised in <label>/ subdirectories.

    Walks via os.walk(followlinks=True) so symlinked class folders
    (the kagglehub cache layout) are traversed. Deduplicates by basename
    because the kaggle RPS dataset mirrors images under both top-level
    class folders and under rps-cv-images/<class>/.
    """
    seen: set = set()
    samples: List[Sample] = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            entry = Path(dirpath) / name
            if entry.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            if name in seen:
                continue
            for part in reversed(entry.parts):
                key = part.lower()
                if key in LABEL_TO_ID:
                    seen.add(name)
                    samples.append(Sample(path=entry, label=key,
                                          label_id=LABEL_TO_ID[key]))
                    break
    return samples


def download_dataset(force: bool = False) -> Path:
    """Download (if missing) and return the dataset root path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    marker = CACHE_DIR / ".ready"
    if marker.exists() and not force:
        return CACHE_DIR

    try:
        import kagglehub  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "kagglehub is required to download the RPS dataset. "
            "Install with `pip install kagglehub` or place images manually in "
            f"{CACHE_DIR} arranged as <class>/<image>."
        ) from e

    print(f"[modusmate] downloading drgfreeman/rockpaperscissors via kagglehub...")
    path = kagglehub.dataset_download("drgfreeman/rockpaperscissors")
    src = Path(path)
    # Copy/symlink into our cache location (kagglehub uses its own cache)
    if str(src) != str(CACHE_DIR):
        # create symlinks for the class dirs
        for sub in src.iterdir():
            target = CACHE_DIR / sub.name
            if not target.exists():
                try:
                    target.symlink_to(sub)
                except OSError:
                    # fall back to writing a pointer file
                    (CACHE_DIR / "source.txt").write_text(str(src))
                    break
    marker.write_text("ok")
    return CACHE_DIR


def _scan_generic_dataset(root: Path, classes: List[str]) -> List[Sample]:
    """Find images organised in ``root/<class>/*.{png,jpg,...}`` for an
    arbitrary class list. ``classes`` defines the label_id assignment
    (index in the list). Used by trained-model test splits saved with
    a ``classes.json`` companion file.
    """
    samples: List[Sample] = []
    name_to_id = {c: i for i, c in enumerate(classes)}
    for cls in classes:
        sub = root / cls
        if not sub.is_dir():
            continue
        for entry in sorted(sub.iterdir()):
            if entry.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            samples.append(Sample(path=entry, label=cls,
                                  label_id=name_to_id[cls]))
    return samples


def load_samples(limit: int = 0, seed: int = 42, dataset_dir: Path | None = None) -> List[Sample]:
    """Return a (shuffled, deterministic) sample list.

    If ``dataset_dir/classes.json`` exists, the directory is treated as a
    generic class-folder layout (trained-model test splits dumped by
    ``algo_train --save-test-dir``). Otherwise we fall back to the
    rock-paper-scissors loader for backwards compatibility.
    """
    root = Path(dataset_dir) if dataset_dir else download_dataset()
    classes_json = root / "classes.json"
    if classes_json.is_file():
        import json as _json
        meta = _json.loads(classes_json.read_text())
        classes = list(meta.get("classes") or [])
        samples = _scan_generic_dataset(root, classes)
    else:
        samples = _scan_dataset(root)
        if not samples:
            # try the kagglehub source path if symlinks failed
            ptr = CACHE_DIR / "source.txt"
            if ptr.exists():
                samples = _scan_dataset(Path(ptr.read_text().strip()))
    rng = random.Random(seed)
    rng.shuffle(samples)
    if limit and limit > 0:
        samples = samples[:limit]
    return samples


def load_image_320x240_rgb888(path: Path) -> bytes:
    """Open + resize + convert to raw RGB888 bytes (320x240, top-left aligned)."""
    img = Image.open(path).convert("RGB")
    img = img.resize((320, 240), Image.BILINEAR)
    return img.tobytes()
