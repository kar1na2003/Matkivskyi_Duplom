"""Smoke tests for the per-algo training pipeline.

Keeps the dataset tiny (40 samples, 16x16) and only exercises a handful
of algos so the test stays under a second.  The point is to verify the
plumbing - dataset loading, algo application, MLP training, CSV/MD
writing - not to assert a particular accuracy ranking.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("cv2")

from modusmate_host import algo_train, algo_lib
from modusmate_host.algos import FIRMWARE_ALGO_COUNT, ALGO_NAMES


def test_algo_lib_covers_all_firmware_algos():
    for name in ALGO_NAMES[:FIRMWARE_ALGO_COUNT]:
        assert name in algo_lib.ALGO_FUNCS, f"missing host impl for {name}"


def test_algo_lib_apply_returns_same_shape_uint8():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    for name in ["passthrough", "sobel", "canny", "harris", "otsu",
                 "gabor", "frangi", "erode", "mser", "akaze", "fast12"]:
        out = algo_lib.apply(name, img)
        assert out.dtype == np.uint8
        assert out.shape == img.shape, f"{name}: shape changed"


def test_shapes_dataset_loads():
    X, y, classes = algo_train.load_dataset("shapes", samples=40, size=16)
    assert X.dtype == np.uint8
    assert X.shape == (40, 16, 16, 3)
    assert y.shape == (40,)
    assert set(y.tolist()) <= set(range(len(classes)))


def test_train_pipeline_writes_outputs(tmp_path: Path):
    X, y, _ = algo_train.load_dataset("shapes", samples=40, size=16)
    algos = ["passthrough", "sobel", "canny", "harris", "otsu"]
    results = algo_train.run(
        algos, X, y, test_frac=0.25,
        hidden=16, epochs=3, seed=0, log=lambda *a, **k: None)
    assert len(results) == len(algos)
    for r in results:
        assert 0.0 <= r.test_acc <= 1.0
        assert 0.0 <= r.train_acc <= 1.0
        assert r.prep_us >= 0
        assert r.infer_us >= 0

    csv_path = tmp_path / "r.csv"
    md_path = tmp_path / "r.md"
    algo_train.write_csv(results, csv_path)
    algo_train.write_markdown(results, md_path,
                              dataset="shapes", n_total=40, n_classes=4)
    assert csv_path.exists() and csv_path.stat().st_size > 0
    assert md_path.exists() and md_path.stat().st_size > 0
    text = md_path.read_text()
    assert "passthrough" in text
    assert "test_acc" in text
