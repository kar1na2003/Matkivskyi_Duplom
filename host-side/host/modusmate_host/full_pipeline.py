"""End-to-end pipeline: train all algos -> dump test split -> on-board sweep -> plot.

Single command that runs the whole evaluation cycle:

  1. Train one MLP per preprocessing algorithm with augmentation + L2.
  2. Dump the held-out test split as PNGs (with a class permutation that
     matches the firmware's ``model_to_dataset[]`` mapping, so on-board
     ``class_id`` answers line up with the saved labels).
  3. For every trained model: install -> build -> qprogram ->
     ``CMD_SET_ALGO`` -> benchmark on the held-out test images.
  4. Compute precision / recall / F1 / confusion from the on-board
     predictions (sklearn).
  5. Render plots (host vs board, F1 leaderboard, confusion, etc.).

Run::

    python -m modusmate_host.full_pipeline \\
        --port /dev/cu.usbmodem3102 \\
        --dataset shapes --restrict-classes 3 \\
        --firmware-class-perm 2,1,0 \\
        --algos all \\
        --limit 60

Use ``--algos all`` to sweep the full firmware-supported set, or pass a
comma-separated subset (e.g. ``--algos passthrough,harris,sobel``).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

from . import models as _models
from .algos import ALGO_NAMES, FIRMWARE_ALGO_COUNT


def _train_step(*, dataset: str, algos: str, hidden: int, samples: int,
                size: int, width: Optional[int], height: Optional[int],
                augment: str, augment_factor: int, alpha: float,
                early_stopping: bool, patience: int, restrict_classes: int,
                firmware_class_perm: Optional[str],
                dataset_classes: Optional[str],
                test_frac: float, seed: int, epochs: int,
                save_test_dir: Path, export_dir: Path,
                out_csv: Path, out_md: Path) -> int:
    """Re-invoke ``algo_train.main`` in-process so the user sees one stream."""
    from . import algo_train
    argv = [
        "--dataset", dataset,
        "--algos", algos,
        "--hidden", str(hidden),
        "--samples", str(samples),
        "--size", str(size),
        "--augment", augment,
        "--augment-factor", str(augment_factor),
        "--alpha", str(alpha),
        "--patience", str(patience),
        "--test-frac", str(test_frac),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--export-models", str(export_dir),
        "--save-test-dir", str(save_test_dir),
        "--out-csv", str(out_csv),
        "--out-md", str(out_md),
    ]
    if width is not None:
        argv += ["--width", str(width)]
    if height is not None:
        argv += ["--height", str(height)]
    if early_stopping:
        argv.append("--early-stopping")
    else:
        argv.append("--no-early-stopping")
    if restrict_classes:
        argv += ["--restrict-classes", str(restrict_classes)]
    if firmware_class_perm:
        argv += ["--firmware-class-perm", firmware_class_perm]
    if dataset_classes:
        argv += ["--dataset-classes", dataset_classes]
    return algo_train.main(argv)


def _sweep_step(*, port: str, baud: int, upgrade_baud: Optional[int],
                limit: int, dataset_dir: Path, seed: int,
                per_image_timeout: float, models_root: Path, fw_dir: Path,
                include: Optional[str], exclude: Optional[str],
                out_csv: Path, out_md: Path) -> int:
    from . import sweep as _sweep
    rows = _sweep.sweep(
        port=port, baud=baud, upgrade_baud=upgrade_baud,
        limit=limit, dataset_dir=dataset_dir, seed=seed,
        per_image_timeout=per_image_timeout,
        models_root=models_root, fw_dir=fw_dir,
        include=include, exclude=exclude,
        out_csv=out_csv, out_md=out_md)
    return 0 if rows else 1


def _plot_step(*, train_csv: Path, sweep_csv: Path, out_dir: Path) -> int:
    from . import plot_results
    argv = ["--train", str(train_csv),
            "--sweep", str(sweep_csv),
            "--out", str(out_dir)]
    return plot_results.main(argv)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # board / sweep
    p.add_argument("--port", required=True)
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--upgrade-baud", type=int, default=1_000_000)
    p.add_argument("--limit", type=int, default=60,
                   help="images per algo on the board (default 60)")
    p.add_argument("--per-image-timeout", type=float, default=30.0)
    p.add_argument("--fw-dir", default=None,
                   help=f"firmware project (default {_models.DEFAULT_FW_DIR})")
    # training
    p.add_argument("--dataset", default="shapes",
                   help="'shapes', 'digits', or a path to a class-folder root")
    p.add_argument("--algos", default="all",
                   help="comma-separated list, or 'all' for every "
                        "firmware-supported algo")
    p.add_argument("--samples", type=int, default=4000)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--width", type=int, default=None,
                   help="model input width (defaults to --size). "
                        "Pair with --height 240 + --width 320 to train at "
                        "native camera resolution.")
    p.add_argument("--height", type=int, default=None,
                   help="model input height (defaults to --size).")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--alpha", type=float, default=1e-3)
    p.add_argument("--early-stopping", dest="early_stopping",
                   action="store_true", default=True)
    p.add_argument("--no-early-stopping", dest="early_stopping",
                   action="store_false")
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--augment", default="basic",
                   help="'none', 'basic', 'all', or comma-list")
    p.add_argument("--augment-factor", type=int, default=4)
    p.add_argument("--test-frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--restrict-classes", type=int, default=3,
                   help="keep only the first N class IDs (default 3 = "
                        "matches firmware's 3-lane decoder)")
    p.add_argument("--dataset-classes", default=None,
                   help="forwarded to algo_train --dataset-classes; "
                        "comma-separated class names to keep when using a "
                        "preset / kaggle:<slug> dataset")
    p.add_argument("--firmware-class-perm", default="2,1,0",
                   help="firmware's model_to_dataset[] permutation; "
                        "default '2,1,0' matches the current PSE84 RPS build")
    # output
    p.add_argument("--workdir", default="runs/full_pipeline",
                   help="all artefacts (CSVs, plots, test split) go under here")
    p.add_argument("--keep-models", action="store_true",
                   help="don't wipe the models/ tree before training")
    p.add_argument("--include", default=None,
                   help="regex; restrict the sweep to matching model names")
    p.add_argument("--exclude", default=None,
                   help="regex; skip matching model names")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-sweep", action="store_true")
    p.add_argument("--skip-plot", action="store_true")
    args = p.parse_args(argv)

    # ---- algo selection ---------------------------------------------------
    if args.algos.strip().lower() == "all":
        algos = ",".join(ALGO_NAMES[:FIRMWARE_ALGO_COUNT])
    else:
        algos = args.algos.strip()

    # ---- paths ------------------------------------------------------------
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    test_split = workdir / "test_split"
    train_csv = workdir / "algo_train_results.csv"
    train_md = workdir / "algo_train_summary.md"
    sweep_csv = workdir / "board_test.csv"
    sweep_md = workdir / "board_test.md"
    plots_dir = workdir / "plots"

    fw_dir = Path(args.fw_dir).expanduser() if args.fw_dir else _models.DEFAULT_FW_DIR
    models_root = _models.MODELS_DIR

    # The pipeline assumes the models/ tree contains only the freshly-
    # trained set, otherwise the sweep will iterate stale models too.
    # We never wipe known fixtures used by the test suite / smoke tests.
    PROTECTED = {"stump_const", "object_detect_rps"}
    if not args.skip_train and not args.keep_models and models_root.exists():
        wiped = 0
        for sub in models_root.iterdir():
            if sub.name in PROTECTED:
                continue
            if sub.is_dir() and (sub / "manifest.json").is_file():
                shutil.rmtree(sub)
                wiped += 1
        if wiped:
            print(f"[pipeline] wiped {wiped} stale model dir(s) under "
                  f"{models_root}")

    # ---- 1. train ---------------------------------------------------------
    t_total = time.time()
    if not args.skip_train:
        print(f"\n[pipeline] === training {algos.count(',') + 1} algo(s) "
              f"on '{args.dataset}' ===")
        rc = _train_step(
            dataset=args.dataset, algos=algos, hidden=args.hidden,
            samples=args.samples, size=args.size,
            width=args.width, height=args.height,
            augment=args.augment, augment_factor=args.augment_factor,
            alpha=args.alpha, early_stopping=args.early_stopping,
            patience=args.patience,
            restrict_classes=args.restrict_classes,
            firmware_class_perm=args.firmware_class_perm,
            dataset_classes=args.dataset_classes,
            test_frac=args.test_frac, seed=args.seed, epochs=args.epochs,
            save_test_dir=test_split, export_dir=models_root,
            out_csv=train_csv, out_md=train_md)
        if rc != 0:
            print(f"[pipeline] train step failed (rc={rc})")
            return rc
    else:
        print("[pipeline] --skip-train: re-using previous training artefacts")

    # ---- 2. sweep on board -----------------------------------------------
    if not args.skip_sweep:
        if not test_split.is_dir():
            print(f"[pipeline] no test split at {test_split}; "
                  "did training succeed?")
            return 1
        print(f"\n[pipeline] === on-board sweep ({args.port}) ===")
        rc = _sweep_step(
            port=args.port, baud=args.baud, upgrade_baud=args.upgrade_baud,
            limit=args.limit, dataset_dir=test_split, seed=args.seed,
            per_image_timeout=args.per_image_timeout,
            models_root=models_root, fw_dir=fw_dir,
            include=args.include, exclude=args.exclude,
            out_csv=sweep_csv, out_md=sweep_md)
        if rc != 0:
            print("[pipeline] sweep produced no rows; aborting")
            return rc
    else:
        print("[pipeline] --skip-sweep: re-using previous sweep CSV")

    # ---- 3. plots ---------------------------------------------------------
    if not args.skip_plot:
        if not train_csv.is_file() or not sweep_csv.is_file():
            print("[pipeline] missing CSVs; cannot plot")
            return 1
        print(f"\n[pipeline] === rendering plots -> {plots_dir} ===")
        rc = _plot_step(train_csv=train_csv, sweep_csv=sweep_csv,
                        out_dir=plots_dir)
        if rc != 0:
            print(f"[pipeline] plot step rc={rc}")
            return rc

    elapsed = time.time() - t_total
    print(f"\n[pipeline] done in {elapsed/60:.1f} min")
    print(f"  train csv : {train_csv}")
    print(f"  sweep csv : {sweep_csv}")
    print(f"  plots     : {plots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
