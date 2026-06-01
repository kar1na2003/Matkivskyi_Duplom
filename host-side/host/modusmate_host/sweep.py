"""Full per-algorithm sweep: flash → set algo → benchmark → repeat.

For every directory under ``models/`` whose manifest carries a
``prep_algo`` field (i.e. the per-algo MLPs produced by
``modusmate_host.algo_train --export-models``), this module:

  1. Flashes the model to the board (issues ``CMD_SET_ALGO`` for free
     because :func:`modusmate_host.models.flash` already wires that up
     from ``manifest.prep_algo``).
  2. Runs a short on-board benchmark for *just that one algo* using the
     existing :func:`modusmate_host.benchmark.run_benchmark` helper.
  3. Records the per-algo metrics (accuracy, mean algo µs, mean infer
     µs, no-detect count, mean confidence).
  4. Writes a single combined CSV + markdown leaderboard at the end.

Run::

    python -m modusmate_host.sweep --port /dev/cu.usbmodem3102 --limit 30

The CSV is suitable for sorting by accuracy or speed to pick the
production model::

    sort -t, -k4 -g -r sweep_results.csv | head
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from . import models as _models
from .algos import ALGO_NAMES, FIRMWARE_ALGO_COUNT
from .benchmark import BenchProgress, run_benchmark


@dataclass
class SweepRow:
    model: str
    prep_algo: str
    algo_id: int
    seen: int
    correct: int
    accuracy: float
    no_detect: int
    mean_conf: float
    mean_algo_us: int
    mean_infer_us: int
    elapsed_s: float
    flash_s: float
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    per_class_f1: str = ""
    confusion: str = ""


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def discover_trained_models(models_root: Path = _models.MODELS_DIR,
                            include: Optional[str] = None,
                            exclude: Optional[str] = None,
                            ) -> List[_models.ModelManifest]:
    """Return all models whose manifest has a firmware-supported ``prep_algo``.

    ``include`` / ``exclude`` are regexes matched against the model name.
    """
    inc_rx = re.compile(include) if include else None
    exc_rx = re.compile(exclude) if exclude else None
    out: List[_models.ModelManifest] = []
    for m in _models.list_models(models_root):
        if not m.prep_algo:
            continue
        if m.prep_algo not in ALGO_NAMES:
            continue
        if ALGO_NAMES.index(m.prep_algo) >= FIRMWARE_ALGO_COUNT:
            continue
        if inc_rx and not inc_rx.search(m.name):
            continue
        if exc_rx and exc_rx.search(m.name):
            continue
        out.append(m)
    out.sort(key=lambda m: m.name)
    return out


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def sweep(*,
          port: str,
          baud: int = 115200,
          upgrade_baud: Optional[int] = 1_000_000,
          limit: int = 30,
          dataset_dir: Optional[Path] = None,
          seed: int = 42,
          per_image_timeout: float = 30.0,
          models_root: Path = _models.MODELS_DIR,
          fw_dir: Path = _models.DEFAULT_FW_DIR,
          include: Optional[str] = None,
          exclude: Optional[str] = None,
          out_csv: Optional[Path] = Path("sweep_results.csv"),
          out_md: Optional[Path] = Path("sweep_results.md"),
          on_log=print,
          ) -> List[SweepRow]:
    """Iterate through every trained model and benchmark it on the board."""
    targets = discover_trained_models(models_root, include=include,
                                      exclude=exclude)
    if not targets:
        on_log(f"[sweep] no trained models found under {models_root}; "
               "did you run `modusmate_host.algo_train --export-models`?")
        return []

    on_log(f"[sweep] {len(targets)} model(s) to evaluate, "
           f"{limit} image(s) each -> ~{len(targets)} flash cycles.")

    rows: List[SweepRow] = []
    sweep_t0 = time.time()
    for i, m in enumerate(targets, 1):
        aid = ALGO_NAMES.index(m.prep_algo)        # type: ignore[arg-type]
        # Manifest 'name' is <algo>_<dataset>; the on-disk directory is
        # <algo>_<dataset>_<date>.  flash() / install_model() resolve by
        # directory name, so use that.
        dir_name = m.path.name
        on_log(f"\n[sweep] ({i}/{len(targets)}) {dir_name}  "
               f"prep_algo={m.prep_algo} (id {aid})")

        # ---- flash + auto CMD_SET_ALGO --------------------------------
        flash_t0 = time.time()
        try:
            _models.flash(dir_name, fw_dir=fw_dir, models_root=models_root,
                          port=port, baud=baud, verify=True,
                          set_prep_algo=True, on_progress=on_log)
        except _models.ModelInstallError as e:
            on_log(f"[sweep] flash failed for {dir_name}: {e}; skipping")
            continue
        flash_s = time.time() - flash_t0

        # ---- benchmark just this one algo -----------------------------
        def _bench_log(p: BenchProgress) -> None:
            if p.kind == "algo_end":
                on_log(f"[sweep] {dir_name}: acc={p.accuracy*100:5.1f}% "
                       f"({p.correct}/{p.seen}) "
                       f"meanAlgo={p.mean_algo_us}us "
                       f"meanInf={p.mean_infer_us}us")
            elif p.kind == "error":
                on_log(f"[sweep] bench error: {p.message}")

        summary = run_benchmark(
            port=port,
            baud=baud,
            upgrade_baud=upgrade_baud,
            algo_ids=[aid],
            limit=limit,
            dataset_dir=dataset_dir,
            seed=seed,
            per_image_timeout=per_image_timeout,
            csv_out=None,
            summary_out=None,
            progress=_bench_log,
            nn=None,                # already flashed above
            enable_preview=False,
        )
        per = summary.per_algo.get(aid)
        if per is None:
            on_log(f"[sweep] no result for {dir_name}; skipping row")
            continue
        # ---- compute F1 / precision / recall / confusion --------------
        algo_rows = [r for r in summary.rows
                     if int(r.get("algo_id", -1)) == aid]
        f1_metrics = _compute_classification_metrics(algo_rows)
        rows.append(SweepRow(
            model=dir_name,
            prep_algo=str(m.prep_algo),
            algo_id=aid,
            seen=int(per["seen"]),
            correct=int(per["correct"]),
            accuracy=float(per["accuracy"]),
            no_detect=int(per["no_detect"]),
            mean_conf=float(per["mean_conf"]),
            mean_algo_us=int(per["mean_algo_us"]),
            mean_infer_us=int(per["mean_infer_us"]),
            elapsed_s=float(per["elapsed_s"]),
            flash_s=flash_s,
            **f1_metrics,
        ))

    total_s = time.time() - sweep_t0
    on_log(f"\n[sweep] done in {total_s:.1f}s; {len(rows)} row(s)")

    if out_csv is not None and rows:
        _write_csv(out_csv, rows)
        on_log(f"[sweep] wrote {out_csv}")
    if out_md is not None and rows:
        _write_md(out_md, rows)
        on_log(f"[sweep] wrote {out_md}")
    return rows


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

_CSV_HEADER = ["model", "prep_algo", "algo_id", "seen", "correct",
               "accuracy", "no_detect", "mean_conf",
               "mean_algo_us", "mean_infer_us", "elapsed_s", "flash_s",
               "precision_macro", "recall_macro", "f1_macro",
               "f1_weighted", "per_class_f1", "confusion"]


def _compute_classification_metrics(rows: List[dict]) -> dict:
    """Return precision/recall/F1/confusion from BenchSummary.rows.

    Skips no-detect rows (pred_id == -1 or 255). Falls back to zeros if
    sklearn is unavailable so the sweep never crashes.
    """
    out = {"precision_macro": 0.0, "recall_macro": 0.0,
           "f1_macro": 0.0, "f1_weighted": 0.0,
           "per_class_f1": "", "confusion": ""}
    if not rows:
        return out
    try:
        from sklearn.metrics import (precision_recall_fscore_support,
                                     confusion_matrix)
    except ImportError:
        return out
    y_true: List[int] = []
    y_pred: List[int] = []
    label_to_name: dict = {}
    for r in rows:
        lid = int(r.get("label_id", -1))
        pid = int(r.get("pred_id", -1))
        if lid < 0:
            continue
        label_to_name[lid] = str(r.get("label", str(lid)))
        # treat no-detect as a separate class so it counts against recall
        if pid < 0 or pid == 0xFF:
            pid = -1
        y_true.append(lid)
        y_pred.append(pid)
    if not y_true:
        return out
    label_ids = sorted(label_to_name.keys())
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, average=None, zero_division=0)
    pr_m, rc_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, average="weighted",
        zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    per_class = ";".join(f"{label_to_name[label_ids[i]]}={f1[i]:.3f}"
                         for i in range(len(label_ids)))
    confusion = "|".join(",".join(str(int(v)) for v in row) for row in cm)
    out.update({"precision_macro": float(pr_m),
                "recall_macro": float(rc_m),
                "f1_macro": float(f1_m),
                "f1_weighted": float(f1_w),
                "per_class_f1": per_class,
                "confusion": confusion})
    return out


def _write_csv(path: Path, rows: Iterable[SweepRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_CSV_HEADER)
        for r in rows:
            w.writerow([r.model, r.prep_algo, r.algo_id, r.seen, r.correct,
                        f"{r.accuracy:.4f}", r.no_detect,
                        f"{r.mean_conf:.4f}", r.mean_algo_us,
                        r.mean_infer_us, f"{r.elapsed_s:.2f}",
                        f"{r.flash_s:.2f}",
                        f"{r.precision_macro:.4f}",
                        f"{r.recall_macro:.4f}",
                        f"{r.f1_macro:.4f}",
                        f"{r.f1_weighted:.4f}",
                        r.per_class_f1, r.confusion])


def _write_md(path: Path, rows: Iterable[SweepRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: r.f1_macro, reverse=True)
    lines = [
        "# ModusMate per-algo sweep",
        "",
        "Sorted by **F1 macro** (on-board, held-out test split).",
        "",
        "| Rank | Model | Algo | Acc | F1 macro | F1 weighted | "
        "Precision | Recall | Mean algo µs | Mean infer µs | "
        "No-detect | Per-class F1 |",
        "|---:|:--|:--|---:|---:|---:|---:|---:|---:|---:|---:|:--|",
    ]
    for i, r in enumerate(sorted_rows, 1):
        lines.append(
            f"| {i} | {r.model} | {r.prep_algo} | "
            f"{r.accuracy*100:.1f}% ({r.correct}/{r.seen}) | "
            f"{r.f1_macro:.3f} | {r.f1_weighted:.3f} | "
            f"{r.precision_macro:.3f} | {r.recall_macro:.3f} | "
            f"{r.mean_algo_us} | {r.mean_infer_us} | "
            f"{r.no_detect} | `{r.per_class_f1}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True,
                    help="serial port (e.g. /dev/cu.usbmodem3102 or COM4)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--upgrade-baud", type=int, default=1_000_000,
                    help="post-handshake baud (set to 115200 to disable)")
    ap.add_argument("--limit", type=int, default=30,
                    help="images per algo (default 30)")
    ap.add_argument("--dataset-dir", "--test-dir", default=None,
                    dest="dataset_dir",
                    help="path to a held-out test set produced by "
                         "`algo_train --save-test-dir DIR`. The directory "
                         "must contain a classes.json so the on-board "
                         "predictions are scored against the correct class "
                         "indices. Falls back to the Kaggle RPS dataset.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-image-timeout", type=float, default=30.0)
    ap.add_argument("--models-dir", default=None,
                    help=f"override models root (default {_models.MODELS_DIR})")
    ap.add_argument("--fw-dir", default=None,
                    help=f"firmware project dir (default {_models.DEFAULT_FW_DIR})")
    ap.add_argument("--include", default=None,
                    help="regex; only models whose name matches are run")
    ap.add_argument("--exclude", default=None,
                    help="regex; models whose name matches are skipped")
    ap.add_argument("--out-csv", default="sweep_results.csv")
    ap.add_argument("--out-md", default="sweep_results.md")
    ap.add_argument("--list", action="store_true",
                    help="just print the discovered targets and exit")
    args = ap.parse_args(argv)

    models_root = Path(args.models_dir) if args.models_dir else _models.MODELS_DIR
    fw_dir = Path(args.fw_dir) if args.fw_dir else _models.DEFAULT_FW_DIR

    if args.list:
        for m in discover_trained_models(models_root,
                                         include=args.include,
                                         exclude=args.exclude):
            print(f"{m.path.name:48s}  prep_algo={m.prep_algo}")
        return 0

    rows = sweep(
        port=args.port,
        baud=args.baud,
        upgrade_baud=args.upgrade_baud,
        limit=args.limit,
        dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
        seed=args.seed,
        per_image_timeout=args.per_image_timeout,
        models_root=models_root,
        fw_dir=fw_dir,
        include=args.include,
        exclude=args.exclude,
        out_csv=Path(args.out_csv) if args.out_csv else None,
        out_md=Path(args.out_md) if args.out_md else None,
    )
    return 0 if rows else 2


if __name__ == "__main__":
    sys.exit(main())
