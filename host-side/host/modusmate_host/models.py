"""Model registry: install / flash a chosen NN onto the firmware.

The repo-root ``models/`` directory holds one subdirectory per neural
network the firmware can run.  Each subdir contains:

    model.h         - C header with IMAI_DATAIN_SHAPE / IMAI_DATAOUT_SHAPE
                      and the IMAI_compute / IMAI_init / IMAI_finalize
                      function declarations (Imagimob ABI).
    model.c         - Implementation. Compiled into proj_cm55 at build time.
    manifest.json   - Metadata: input/output shapes, class labels, optional
                      ``expected_smoke_output`` for the smoke test.

``install_model(name, fw_dir)`` copies ``model.h`` + ``model.c`` from the
registry into ``<fw_dir>/proj_cm55/model/``, taking a one-time ``.bak`` of
whatever was there before.  ``flash(name, port=...)`` does the install,
runs ``project_creator`` build + qprogram, then waits for the board's USB
serial endpoint to re-enumerate.

The firmware is sized for exactly one model in Flash + SoCMEM, so we use
"swap and reflash" rather than carrying multiple models on the board at
once.  Only one NN is resident at a time -> no RAM/Flash pressure.

CLI:
    python -m modusmate_host.models list
    python -m modusmate_host.models install <name> [--fw DIR]
    python -m modusmate_host.models flash <name> [--fw DIR] [--port PORT]
                                                 [--no-verify]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# ---- repo / firmware paths --------------------------------------------------

_HOST_DIR = Path(__file__).resolve().parent           # host/modusmate_host/
_REPO_ROOT = _HOST_DIR.parent.parent                  # repo root
MODELS_DIR = _REPO_ROOT / "models"

DEFAULT_FW_DIR = Path(
    os.environ.get(
        "MODUSMATE_FW_DIR",
        str(Path.home() / "mtw-cli" / "ws-camera-imgproc-usb"
            / "camera-imgproc-usb"),
    )
)

# Relative path inside the firmware tree where model.h/model.c live.
_FW_MODEL_SUBDIR = Path("proj_cm55") / "model"


# ---- manifest ---------------------------------------------------------------


@dataclass
class ModelManifest:
    name: str
    description: str
    framework: str
    imai_api: bool
    input_shape: List[int]
    input_dtype: str
    output_shape: List[int]
    output_dtype: str
    output_layout: str
    classes: List[str]
    flash_bytes: int
    ram_bytes: int
    expected_smoke_output: Optional[dict]
    files: List[str]
    path: Path
    prep_algo: Optional[str] = None

    @property
    def shape_str(self) -> str:
        ish = "x".join(str(x) for x in self.input_shape)
        osh = "x".join(str(x) for x in self.output_shape)
        return f"in[{self.input_dtype}:{ish}] -> out[{self.output_dtype}:{osh}]"


def load_manifest(name: str, models_root: Path = MODELS_DIR) -> ModelManifest:
    mdir = models_root / name
    mfile = mdir / "manifest.json"
    if not mfile.is_file():
        raise FileNotFoundError(f"no manifest at {mfile}")
    raw = json.loads(mfile.read_text(encoding="utf-8"))
    # Validate the source files are present.
    for f in raw.get("files", ["model.h", "model.c"]):
        if not (mdir / f).is_file():
            raise FileNotFoundError(f"model {name}: missing source file {f}")
    return ModelManifest(
        name=raw["name"],
        description=raw.get("description", ""),
        framework=raw.get("framework", ""),
        imai_api=bool(raw.get("imai_api", True)),
        input_shape=list(raw["input_shape"]),
        input_dtype=raw.get("input_dtype", "uint8"),
        output_shape=list(raw["output_shape"]),
        output_dtype=raw.get("output_dtype", "float32"),
        output_layout=raw.get("output_layout", "column_major"),
        classes=list(raw.get("classes", [])),
        flash_bytes=int(raw.get("flash_bytes", 0)),
        ram_bytes=int(raw.get("ram_bytes", 0)),
        expected_smoke_output=raw.get("expected_smoke_output"),
        files=list(raw.get("files", ["model.h", "model.c"])),
        path=mdir,
        prep_algo=raw.get("prep_algo"),
    )


def list_models(models_root: Path = MODELS_DIR) -> List[ModelManifest]:
    if not models_root.is_dir():
        return []
    out: List[ModelManifest] = []
    for entry in sorted(models_root.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "manifest.json").is_file():
            continue
        try:
            out.append(load_manifest(entry.name, models_root))
        except Exception as e:                      # pragma: no cover
            print(f"  warn: skipping {entry.name}: {e}", file=sys.stderr)
    return out


# ---- install ----------------------------------------------------------------


class ModelInstallError(RuntimeError):
    pass


def _resolve_fw_model_dir(fw_dir: Path) -> Path:
    fw_dir = Path(fw_dir).expanduser().resolve()
    if not fw_dir.is_dir():
        raise ModelInstallError(f"firmware directory not found: {fw_dir}")
    target = fw_dir / _FW_MODEL_SUBDIR
    if not target.is_dir():
        raise ModelInstallError(
            f"firmware tree has no {_FW_MODEL_SUBDIR}/ "
            f"(looked in {target}). Set MODUSMATE_FW_DIR or pass --fw.")
    return target


def install_model(name: str, fw_dir: Path = DEFAULT_FW_DIR,
                  models_root: Path = MODELS_DIR) -> Path:
    """Copy ``models/<name>/model.{h,c}`` into the firmware tree.

    On the first call for a given firmware tree we save the original
    ``model.h`` / ``model.c`` next to themselves as ``model.h.orig`` /
    ``model.c.orig`` so the user can restore the as-shipped model later.
    Subsequent installs overwrite without touching the ``.orig`` backups.
    """
    manifest = load_manifest(name, models_root)
    target = _resolve_fw_model_dir(fw_dir)

    for fname in manifest.files:
        src = manifest.path / fname
        dst = target / fname
        backup = dst.with_suffix(dst.suffix + ".orig")
        if dst.is_file() and not backup.exists():
            shutil.copy2(dst, backup)
        shutil.copy2(src, dst)
    return target


# ---- build + flash ----------------------------------------------------------


def _project_creator():
    """Lazy import so ``models.py`` can still be loaded in test environments
    that don't have project_creator on sys.path."""
    sys.path.insert(0, str(_REPO_ROOT))
    try:
        from project_creator import build_wrapper  # type: ignore
    except Exception as e:                           # pragma: no cover
        raise ModelInstallError(
            "project_creator package not importable from "
            f"{_REPO_ROOT}: {e}")
    return build_wrapper


def _verify_link(port: str, baud: int = 115200, attempts: int = 10,
                 delay: float = 0.5) -> bool:
    """Try to ping the board; returns True on first success."""
    from .link import BoardLink                      # local import
    last_err: Optional[Exception] = None
    for _ in range(attempts):
        try:
            with BoardLink(port, baudrate=baud) as link:
                if link.ping(timeout=1.0):
                    return True
        except Exception as e:
            last_err = e
        time.sleep(delay)
    if last_err is not None:
        print(f"  ping verify last error: {last_err}", file=sys.stderr)
    return False


def _resolve_prep_algo_id(prep_algo: str) -> Optional[int]:
    """Look up an algorithm name in the firmware enum and return its ID.

    Returns ``None`` if the name doesn't correspond to a firmware-supported
    algo (e.g. host-only ``orb`` / ``sift`` / ``surf``).
    """
    from .algos import ALGO_NAMES, FIRMWARE_ALGO_COUNT
    try:
        idx = ALGO_NAMES.index(prep_algo)
    except ValueError:
        return None
    if idx >= FIRMWARE_ALGO_COUNT:
        return None
    return idx


def _apply_prep_algo(port: str, baud: int, prep_algo: str,
                     on_progress) -> bool:
    """Open a transient link and tell the board to switch to ``prep_algo``."""
    from .link import BoardLink
    aid = _resolve_prep_algo_id(prep_algo)
    if aid is None:
        on_progress(
            f"[models] prep_algo {prep_algo!r} is not in the firmware "
            f"algorithm enum; skipping CMD_SET_ALGO.")
        return False
    try:
        with BoardLink(port, baudrate=baud) as link:
            if not link.ping(timeout=1.0):
                on_progress(
                    f"[models] could not ping {port}; skipping prep-algo set")
                return False
            if not link.set_algo(aid):
                on_progress(
                    f"[models] CMD_SET_ALGO({aid}={prep_algo}) was not ACKed")
                return False
    except Exception as e:
        on_progress(f"[models] prep-algo set failed: {e}")
        return False
    on_progress(f"[models] prep-algo on board now: {prep_algo} (id {aid})")
    return True


def flash(name: str,
          fw_dir: Path = DEFAULT_FW_DIR,
          models_root: Path = MODELS_DIR,
          port: Optional[str] = None,
          baud: int = 115200,
          verify: bool = True,
          settle_s: float = 3.0,
          set_prep_algo: bool = True,
          on_progress=print) -> None:
    """Install model + build + qprogram + (optional) ping verify.

    When the manifest carries a ``prep_algo`` field (set by the per-algo
    training pipeline) and ``set_prep_algo`` is true and a ``port`` is
    given, also issues ``CMD_SET_ALGO`` so the on-board imgproc
    pipeline runs the same preprocessing the model was trained for.
    """
    bw = _project_creator()
    fw_dir = Path(fw_dir).expanduser().resolve()

    manifest = load_manifest(name, models_root)

    on_progress(f"[models] install {name} -> {fw_dir / _FW_MODEL_SUBDIR}")
    install_model(name, fw_dir=fw_dir, models_root=models_root)

    on_progress(f"[models] build {fw_dir}")
    try:
        bw.build(str(fw_dir))
    except bw.BuildError as e:                       # type: ignore[attr-defined]
        raise ModelInstallError(
            f"build failed for model {name}: {e}\n{getattr(e, 'output', '')}")

    on_progress(f"[models] qprogram {fw_dir}")
    try:
        bw.qprogram(str(fw_dir))
    except bw.BuildError as e:                       # type: ignore[attr-defined]
        raise ModelInstallError(
            f"flash failed for model {name}: {e}\n{getattr(e, 'output', '')}")

    if verify:
        if port is None:
            on_progress("[models] verify skipped (no --port)")
            return
        on_progress(f"[models] settle {settle_s:.1f}s for USB re-enumeration")
        time.sleep(settle_s)
        on_progress(f"[models] ping {port} @ {baud}")
        if not _verify_link(port, baud=baud):
            raise ModelInstallError(
                f"flashed {name} but board did not respond on {port}")
        on_progress(f"[models] ok: {name} live on {port}")

        # Apply the preprocessing algorithm the model was trained against
        # so that the on-board imgproc pipeline matches what host training
        # saw.  Best-effort: a missing/unsupported prep_algo or a
        # transient link error is just logged, never fatal.
        if set_prep_algo and manifest.prep_algo:
            _apply_prep_algo(port, baud, manifest.prep_algo, on_progress)
        elif set_prep_algo and not manifest.prep_algo:
            on_progress("[models] no prep_algo in manifest; "
                        "on-board algo unchanged.")


# ---- CLI --------------------------------------------------------------------


def _cmd_list(args) -> int:
    rows = list_models()
    if not rows:
        print(f"(no models found in {MODELS_DIR})")
        return 1
    for m in rows:
        smoke = ""
        if m.expected_smoke_output:
            so = m.expected_smoke_output
            smoke = (f"  smoke: class_id={so.get('class_id')}"
                     f" conf_x100={so.get('conf_x100')}")
        prep = f"  prep_algo={m.prep_algo}" if m.prep_algo else ""
        print(f"{m.name:24s} {m.shape_str}{prep}{smoke}")
        if m.description:
            print(f"  {m.description.splitlines()[0]}")
    return 0


def _cmd_install(args) -> int:
    fw = Path(args.fw) if args.fw else DEFAULT_FW_DIR
    target = install_model(args.name, fw_dir=fw)
    print(f"installed {args.name} -> {target}")
    return 0


def _cmd_flash(args) -> int:
    fw = Path(args.fw) if args.fw else DEFAULT_FW_DIR
    try:
        flash(args.name,
              fw_dir=fw,
              port=args.port,
              baud=args.baud,
              verify=not args.no_verify,
              set_prep_algo=not args.no_set_prep_algo)
    except ModelInstallError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser("modusmate_host.models")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list models in the registry")

    pi = sub.add_parser("install", help="copy model.{h,c} into firmware tree")
    pi.add_argument("name")
    pi.add_argument("--fw", help="firmware project dir (default: "
                                  f"$MODUSMATE_FW_DIR or {DEFAULT_FW_DIR})")

    pf = sub.add_parser("flash", help="install + build + qprogram + verify")
    pf.add_argument("name")
    pf.add_argument("--fw")
    pf.add_argument("--port", help="serial port (e.g. /dev/cu.usbmodem3102)")
    pf.add_argument("--baud", type=int, default=115200)
    pf.add_argument("--no-verify", action="store_true",
                    help="skip post-flash ping check")
    pf.add_argument("--no-set-prep-algo", action="store_true",
                    help="do not auto-issue CMD_SET_ALGO from manifest's "
                         "prep_algo field")

    args = p.parse_args(argv)

    if args.cmd == "list":
        return _cmd_list(args)
    if args.cmd == "install":
        return _cmd_install(args)
    if args.cmd == "flash":
        return _cmd_flash(args)
    p.error("unknown command")          # pragma: no cover
    return 2


if __name__ == "__main__":              # pragma: no cover
    sys.exit(main())
