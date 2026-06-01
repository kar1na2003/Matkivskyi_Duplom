"""Functional tests for modusmate_host.dataset."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from modusmate_host import dataset as D


def _make_synthetic_dataset(root: Path) -> None:
    """Create a tiny RPS dataset with 2 images per class."""
    for cls, color in [("rock", (255, 0, 0)), ("paper", (0, 255, 0)),
                       ("scissors", (0, 0, 255))]:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            img = Image.new("RGB", (64, 48), color)
            img.save(d / f"{cls}_{i}.png")


def test_scan_dataset_finds_all_classes(tmp_path):
    _make_synthetic_dataset(tmp_path)
    samples = D._scan_dataset(tmp_path)
    assert len(samples) == 6
    labels = {s.label for s in samples}
    assert labels == {"rock", "paper", "scissors"}
    # label_id mapping must match firmware
    for s in samples:
        assert D.LABEL_TO_ID[s.label] == s.label_id


def test_label_id_mapping_matches_firmware_class_ids():
    """Firmware: model_to_dataset[] = {2,1,0} -> dataset 0=Rock, 1=Paper, 2=Scissors."""
    assert D.LABEL_TO_ID == {"rock": 0, "paper": 1, "scissors": 2}
    assert D.ID_TO_LABEL == {0: "rock", 1: "paper", 2: "scissors"}


def test_load_samples_is_deterministic_with_seed(tmp_path):
    _make_synthetic_dataset(tmp_path)
    a = D.load_samples(seed=42, dataset_dir=tmp_path)
    b = D.load_samples(seed=42, dataset_dir=tmp_path)
    assert [s.path for s in a] == [s.path for s in b]


def test_load_samples_respects_limit(tmp_path):
    _make_synthetic_dataset(tmp_path)
    samples = D.load_samples(limit=3, dataset_dir=tmp_path)
    assert len(samples) == 3


def test_load_image_320x240_rgb888(tmp_path):
    src = tmp_path / "test.png"
    Image.new("RGB", (640, 480), (128, 64, 200)).save(src)
    raw = D.load_image_320x240_rgb888(src)
    assert len(raw) == 320 * 240 * 3
    # first pixel should be ~ original solid colour (allow bilinear roundoff)
    r, g, b = raw[0], raw[1], raw[2]
    assert abs(r - 128) <= 2
    assert abs(g - 64) <= 2
    assert abs(b - 200) <= 2


def test_scan_dataset_ignores_non_image_files(tmp_path):
    (tmp_path / "rock").mkdir()
    Image.new("RGB", (10, 10), (1, 2, 3)).save(tmp_path / "rock" / "a.png")
    (tmp_path / "rock" / "readme.txt").write_text("not an image")
    samples = D._scan_dataset(tmp_path)
    assert len(samples) == 1
    assert samples[0].label == "rock"
