"""Tests for the model registry (host/modusmate_host/models.py).

These run without a board: they exercise the manifest parser, the install
copy logic and the flash CLI dispatch (with project_creator mocked out).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from modusmate_host import models as M


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"


def test_models_dir_exists():
    assert MODELS_DIR.is_dir()
    names = {p.name for p in MODELS_DIR.iterdir() if p.is_dir()}
    assert "stump_const" in names
    assert "object_detect_rps" in names


def test_list_models_returns_both():
    rows = M.list_models(MODELS_DIR)
    names = {r.name for r in rows}
    assert "stump_const" in names
    assert "object_detect_rps" in names


def test_load_manifest_stump():
    m = M.load_manifest("stump_const", MODELS_DIR)
    assert m.input_shape == [320, 320, 3]
    assert m.output_shape == [8, 5]
    assert m.input_dtype == "uint8"
    assert m.output_dtype == "float32"
    assert m.classes == ["Rock", "Paper", "Scissors"]
    assert m.expected_smoke_output == {"class_id": 0, "conf_x100": 100}
    assert (m.path / "model.h").is_file()
    assert (m.path / "model.c").is_file()


def test_load_manifest_rps_shapes_match_stump():
    """Stump must mirror real model's input/output shapes so firmware
    needs no conditional path."""
    s = M.load_manifest("stump_const", MODELS_DIR)
    r = M.load_manifest("object_detect_rps", MODELS_DIR)
    assert s.input_shape == r.input_shape
    assert s.output_shape == r.output_shape
    assert s.input_dtype == r.input_dtype
    assert s.output_dtype == r.output_dtype


def test_load_manifest_missing_raises():
    with pytest.raises(FileNotFoundError):
        M.load_manifest("does_not_exist", MODELS_DIR)


def test_install_model_copies_files_and_backs_up(tmp_path: Path):
    fw = tmp_path / "fw"
    target = fw / "proj_cm55" / "model"
    target.mkdir(parents=True)
    # Pre-existing files (simulate the firmware tree).
    (target / "model.h").write_text("// original header\n")
    (target / "model.c").write_text("// original impl\n")

    out = M.install_model("stump_const", fw_dir=fw, models_root=MODELS_DIR)
    assert out == target

    # Originals are backed up.
    assert (target / "model.h.orig").read_text() == "// original header\n"
    assert (target / "model.c.orig").read_text() == "// original impl\n"
    # Stump content is in place.
    assert "MODUSMATE_STUMP_MODEL_H" in (target / "model.h").read_text()
    assert "IMAI_compute" in (target / "model.c").read_text()

    # Re-install should NOT clobber the .orig backups.
    M.install_model("object_detect_rps", fw_dir=fw, models_root=MODELS_DIR)
    assert (target / "model.h.orig").read_text() == "// original header\n"


def test_install_model_no_fw_dir(tmp_path: Path):
    with pytest.raises(M.ModelInstallError):
        M.install_model("stump_const", fw_dir=tmp_path / "nope",
                        models_root=MODELS_DIR)


def test_flash_invokes_build_and_qprogram(tmp_path: Path):
    fw = tmp_path / "fw"
    (fw / "proj_cm55" / "model").mkdir(parents=True)
    (fw / "proj_cm55" / "model" / "model.h").write_text("// orig\n")
    (fw / "proj_cm55" / "model" / "model.c").write_text("// orig\n")

    fake_bw = mock.MagicMock()
    fake_bw.BuildError = type("BuildError", (Exception,), {})
    fake_bw.build = mock.MagicMock(return_value="build ok")
    fake_bw.qprogram = mock.MagicMock(return_value="flash ok")

    with mock.patch.object(M, "_project_creator", return_value=fake_bw):
        M.flash("stump_const", fw_dir=fw, models_root=MODELS_DIR,
                port=None, verify=False, on_progress=lambda *_: None)

    fake_bw.build.assert_called_once()
    fake_bw.qprogram.assert_called_once()
    # And the install actually happened.
    assert "MODUSMATE_STUMP_MODEL_H" in (
        fw / "proj_cm55" / "model" / "model.h").read_text()


def test_flash_propagates_build_error(tmp_path: Path):
    fw = tmp_path / "fw"
    (fw / "proj_cm55" / "model").mkdir(parents=True)
    (fw / "proj_cm55" / "model" / "model.h").write_text("// orig\n")
    (fw / "proj_cm55" / "model" / "model.c").write_text("// orig\n")

    class _BE(Exception):
        output = "compiler exploded"

    fake_bw = mock.MagicMock()
    fake_bw.BuildError = _BE
    fake_bw.build = mock.MagicMock(side_effect=_BE("boom"))
    fake_bw.qprogram = mock.MagicMock()

    with mock.patch.object(M, "_project_creator", return_value=fake_bw):
        with pytest.raises(M.ModelInstallError) as excinfo:
            M.flash("stump_const", fw_dir=fw, models_root=MODELS_DIR,
                    port=None, verify=False, on_progress=lambda *_: None)
    assert "boom" in str(excinfo.value)
    fake_bw.qprogram.assert_not_called()


def test_cli_list(capsys):
    rc = M.main(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "stump_const" in out
    assert "object_detect_rps" in out


def test_resolve_prep_algo_id():
    """Firmware enum lookup honours both supported and host-only names."""
    # supported algos -> their ALGO_NAMES index
    assert M._resolve_prep_algo_id("passthrough") == 0
    assert M._resolve_prep_algo_id("sobel") is not None
    assert M._resolve_prep_algo_id("fast12") is not None
    # host-only names beyond FIRMWARE_ALGO_COUNT should be rejected
    assert M._resolve_prep_algo_id("orb") is None
    assert M._resolve_prep_algo_id("sift") is None
    # unknown name
    assert M._resolve_prep_algo_id("nope") is None


def test_flash_calls_apply_prep_algo_when_manifest_has_one(tmp_path: Path):
    """A manifest carrying ``prep_algo`` triggers an automatic CMD_SET_ALGO
    after the post-flash ping, using the firmware's algo enum index."""
    # synthesise a tiny model dir with prep_algo in the manifest
    models_root = tmp_path / "models"
    src = models_root / "fake"
    src.mkdir(parents=True)
    # We just copy the stump's source files; only the manifest needs to
    # carry prep_algo for this test.
    import shutil as _sh
    _sh.copy2(MODELS_DIR / "stump_const" / "model.h", src / "model.h")
    _sh.copy2(MODELS_DIR / "stump_const" / "model.c", src / "model.c")
    import json as _json
    manifest = _json.loads(
        (MODELS_DIR / "stump_const" / "manifest.json").read_text())
    manifest["name"] = "fake"
    manifest["prep_algo"] = "sobel"
    (src / "manifest.json").write_text(_json.dumps(manifest))

    fw = tmp_path / "fw"
    (fw / "proj_cm55" / "model").mkdir(parents=True)
    (fw / "proj_cm55" / "model" / "model.h").write_text("// orig\n")
    (fw / "proj_cm55" / "model" / "model.c").write_text("// orig\n")

    fake_bw = mock.MagicMock()
    fake_bw.BuildError = type("BuildError", (Exception,), {})
    fake_bw.build = mock.MagicMock()
    fake_bw.qprogram = mock.MagicMock()

    with mock.patch.object(M, "_project_creator", return_value=fake_bw), \
         mock.patch.object(M, "_verify_link", return_value=True), \
         mock.patch.object(M, "_apply_prep_algo",
                           return_value=True) as apply_mock, \
         mock.patch.object(M.time, "sleep"):
        M.flash("fake", fw_dir=fw, models_root=models_root,
                port="/dev/cu.fake", verify=True, settle_s=0.0,
                on_progress=lambda *_: None)

    apply_mock.assert_called_once()
    args, _kwargs = apply_mock.call_args
    assert args[0] == "/dev/cu.fake"
    assert args[2] == "sobel"


def test_flash_skips_apply_prep_algo_when_disabled(tmp_path: Path):
    fw = tmp_path / "fw"
    (fw / "proj_cm55" / "model").mkdir(parents=True)
    (fw / "proj_cm55" / "model" / "model.h").write_text("// orig\n")
    (fw / "proj_cm55" / "model" / "model.c").write_text("// orig\n")

    fake_bw = mock.MagicMock()
    fake_bw.BuildError = type("BuildError", (Exception,), {})
    fake_bw.build = mock.MagicMock()
    fake_bw.qprogram = mock.MagicMock()

    with mock.patch.object(M, "_project_creator", return_value=fake_bw), \
         mock.patch.object(M, "_verify_link", return_value=True), \
         mock.patch.object(M, "_apply_prep_algo") as apply_mock, \
         mock.patch.object(M.time, "sleep"):
        # stump_const has no prep_algo; either way set_prep_algo=False
        # should leave the apply_prep_algo helper untouched.
        M.flash("stump_const", fw_dir=fw, models_root=MODELS_DIR,
                port="/dev/cu.fake", verify=True, settle_s=0.0,
                set_prep_algo=False,
                on_progress=lambda *_: None)
    apply_mock.assert_not_called()
