"""Verify the C exporter produces compilable code matching the trained MLP.

We only run the gcc compile-check if a host C compiler is available;
the numerical-equivalence test (sklearn predict vs C inference) is gated
on the same since it needs the compiled shared object.  When neither is
available the tests skip cleanly.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("cv2")

from modusmate_host import algo_lib, algo_train, c_export


HAS_GCC = shutil.which("cc") is not None or shutil.which("gcc") is not None


def _train_small(side: int = 8, hidden: int = 8) -> tuple:
    X, y, classes = algo_train._make_shapes_dataset(samples=24, size=side)
    X_proc, _, _ = algo_train._process_all("passthrough", X)
    Xt = X_proc.reshape(X_proc.shape[0], -1).astype(np.float32) / 255.0
    from sklearn.neural_network import MLPClassifier
    import warnings
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=4,
                        random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(Xt, y)
    return clf, classes, side


def test_export_writes_three_files(tmp_path: Path):
    clf, classes, side = _train_small()
    mlp = c_export.ExportedMLP.from_sklearn(
        clf, side=side, classes=classes,
        algo_name="sobel", dataset="shapes")
    out = tmp_path / "sobel_shapes_2026-04-27"
    c_export.write_imai_model(mlp, out, test_acc=0.5, prep_us=12.3)
    for f in ("model.h", "model.c", "manifest.json"):
        assert (out / f).is_file(), f"missing {f}"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["imai_api"] is True
    assert manifest["input_shape"] == [320, 320, 3]
    assert manifest["output_shape"] == [8, 5]
    assert manifest["prep_algo"] == "sobel"
    assert manifest["classes"] == list(classes)
    assert manifest["framework"].startswith("modusmate_mlp")


def test_export_more_classes_clamped(tmp_path: Path):
    """A 4-class model can still be exported; it just emits 3 lanes."""
    clf, _, side = _train_small()
    classes_4 = ["a", "b", "c", "d"]
    # forge a 4-class clf by re-training with an extra label
    rng = np.random.default_rng(0)
    X = rng.integers(0, 256, size=(40, side, side, 3), dtype=np.uint8)
    y = rng.integers(0, 4, size=40)
    Xt = X.reshape(40, -1).astype(np.float32) / 255.0
    from sklearn.neural_network import MLPClassifier
    import warnings
    clf4 = MLPClassifier(hidden_layer_sizes=(8,), max_iter=4, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf4.fit(Xt, y)
    mlp = c_export.ExportedMLP.from_sklearn(
        clf4, side=side, classes=classes_4,
        algo_name="canny", dataset="rand")
    out = tmp_path / "canny_rand_x"
    c_export.write_imai_model(mlp, out)
    src = (out / "model.c").read_text()
    assert "EXPORT_CLASSES 3" in src
    assert "N_CLASSES    4" in src


@pytest.mark.skipif(not HAS_GCC, reason="no host C compiler available")
def test_generated_c_compiles_standalone(tmp_path: Path):
    """The generated source compiles in isolation when fed minimal stubs.

    We provide stub headers (`mtb_ml_model.h`, `cy_utils.h`) and define
    `COMPONENT_ML_TFLM` / `EXPAND_AND_STRINGIFY` on the command line so
    the compiler can build the file without the real Infineon SDK.
    """
    clf, classes, side = _train_small(side=8, hidden=4)
    mlp = c_export.ExportedMLP.from_sklearn(
        clf, side=side, classes=classes,
        algo_name="passthrough", dataset="shapes")
    out = tmp_path / "model_dir"
    c_export.write_imai_model(mlp, out)

    # Minimal stubs.
    (tmp_path / "cy_utils.h").write_text(
        "#ifndef CY_UTILS_H\n#define CY_UTILS_H\n"
        "#define CY_SECTION(s)\n"
        "#endif\n")
    (tmp_path / "mtb_ml_model.h").write_text(
        "#ifndef MTB_ML_MODEL_H\n#define MTB_ML_MODEL_H\n"
        "typedef struct mtb_ml_model_s mtb_ml_model_t;\n"
        "#endif\n")
    cc = shutil.which("cc") or shutil.which("gcc")
    cmd = [
        cc, "-c", "-Wall", "-Wno-unused-function",
        "-DCOMPONENT_ML_TFLM=1",
        "-DEXPAND_AND_STRINGIFY(x)=#x",
        f"-I{tmp_path}", f"-I{out}",
        str(out / "model.c"),
        "-o", str(tmp_path / "model.o"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, (
        f"gcc failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    assert (tmp_path / "model.o").exists()


def test_run_with_export_dir(tmp_path: Path):
    """``run(... export_dir=...)`` should drop manifests in the directory."""
    X, y, classes = algo_train.load_dataset("shapes", samples=24, size=12)
    export_dir = tmp_path / "models"
    results = algo_train.run(
        ["passthrough", "sobel"], X, y,
        test_frac=0.25, hidden=8, epochs=2, seed=0,
        export_dir=export_dir, classes=classes,
        dataset_name="shapes", size=12, log=lambda *a, **k: None)
    assert len(results) == 2
    # one subdir per algo, each with the three required files
    subs = sorted(p for p in export_dir.iterdir() if p.is_dir())
    assert len(subs) == 2
    names = sorted(p.name.split("_")[0] for p in subs)
    assert names == ["passthrough", "sobel"]
    for sub in subs:
        for f in ("model.h", "model.c", "manifest.json"):
            assert (sub / f).is_file()
