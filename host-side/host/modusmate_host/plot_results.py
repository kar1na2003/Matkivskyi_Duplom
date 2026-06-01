"""Plot results from algo_train and/or sweep CSVs.

Reads two CSV families:

  * ``algo_train_results.csv`` (host-side training, produced by
    ``modusmate_host.algo_train``). Columns include
    ``algo, family, prep_us, infer_us, train_acc, test_acc, edge_ratio,
    kp_count, binary_ratio, mean_intensity, n_train, n_test``.

  * ``sweep_results.csv`` (on-board, produced by ``modusmate_host.sweep``).
    Columns: ``model, prep_algo, algo_id, seen, correct, accuracy,
    no_detect, mean_conf, mean_algo_us, mean_infer_us, elapsed_s,
    flash_s``.

Generates these PNGs in the output dir:

    01_test_accuracy_bar.png      ranked test_acc bar chart, coloured by family
    02_train_vs_test.png          scatter of train_acc vs test_acc
    03_acc_vs_prep_us.png         accuracy / preprocessing time pareto
    04_f1_macro_bar.png           ranked F1 macro bar chart
    05_precision_recall.png       precision vs recall scatter
    06_per_class_f1.png           top-N grouped bar of per-class F1
    07_confusion_<algo>.png       confusion matrices for top-N algos
    08_board_vs_host.png          (only if --sweep given) on-board vs host accuracy

Run::

    python -m modusmate_host.plot_results --train algo_train_results.csv
    python -m modusmate_host.plot_results \\
        --train algo_train_results.csv --sweep sweep_results.csv \\
        --out plots/
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

# Stable colour per family across all charts.
_FAMILY_COLORS: Dict[str, str] = {
    "basics":       "#4C78A8",
    "edges":        "#F58518",
    "blobs":        "#54A24B",
    "keypoints":    "#E45756",
    "thresholding": "#72B7B2",
    "texture":      "#EECA3B",
    "ridge":        "#B279A2",
    "morphology":   "#9D755D",
}
_DEFAULT_COLOR = "#888888"


def _load_train(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_sweep(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _color_for(family: str) -> str:
    return _FAMILY_COLORS.get(family, _DEFAULT_COLOR)


def _legend_for_families(families: List[str]) -> List:
    handles = []
    seen = []
    for fam in families:
        if fam in seen:
            continue
        seen.append(fam)
        handles.append(plt.Line2D([0], [0], marker="s", linestyle="",
                                  markersize=8, color=_color_for(fam),
                                  label=fam))
    return handles


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_test_accuracy(rows: List[Dict[str, str]], out: Path) -> None:
    rows = sorted(rows, key=lambda r: float(r["test_acc"]), reverse=True)
    names = [r["algo"] for r in rows]
    accs = [float(r["test_acc"]) * 100 for r in rows]
    fams = [r["family"] for r in rows]
    colors = [_color_for(f) for f in fams]

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.28), 6))
    ax.bar(range(len(rows)), accs, color=colors)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Per-algorithm host test accuracy (sorted)")
    ax.set_ylim(0, 100)
    ax.axhline(100.0 / 3, color="#999", linestyle="--", linewidth=0.8,
               label="3-class chance")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(handles=_legend_for_families(fams) +
              [plt.Line2D([0], [0], color="#999", linestyle="--",
                          label="3-class chance")],
              loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_train_vs_test(rows: List[Dict[str, str]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], color="#bbb", linestyle="--", linewidth=0.8,
            label="train = test (no overfit)")
    for r in rows:
        ax.scatter(float(r["train_acc"]) * 100,
                   float(r["test_acc"]) * 100,
                   color=_color_for(r["family"]), s=42, edgecolors="white",
                   linewidths=0.6)
        ax.annotate(r["algo"],
                    (float(r["train_acc"]) * 100,
                     float(r["test_acc"]) * 100),
                    fontsize=6, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Train accuracy (%)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Train vs test accuracy per algorithm")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.grid(linestyle=":", alpha=0.4)
    fams = [r["family"] for r in rows]
    ax.legend(handles=_legend_for_families(fams) +
              [plt.Line2D([0], [0], color="#bbb", linestyle="--",
                          label="train = test")],
              fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_acc_vs_prep(rows: List[Dict[str, str]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for r in rows:
        x = float(r["prep_us"])
        y = float(r["test_acc"]) * 100
        ax.scatter(x, y, color=_color_for(r["family"]), s=42,
                   edgecolors="white", linewidths=0.6)
        ax.annotate(r["algo"], (x, y), fontsize=6, alpha=0.85,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Host preprocessing time (µs, log)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs preprocessing cost (top-left = best)")
    ax.grid(which="both", linestyle=":", alpha=0.4)
    ax.axhline(100.0 / 3, color="#999", linestyle="--", linewidth=0.8)
    fams = [r["family"] for r in rows]
    ax.legend(handles=_legend_for_families(fams), fontsize=8, ncol=2,
              loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_f1_macro(rows: List[Dict[str, str]], out: Path) -> None:
    rows = [r for r in rows if "f1_macro" in r]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: float(r["f1_macro"]), reverse=True)
    names = [r["algo"] for r in rows]
    f1s = [float(r["f1_macro"]) * 100 for r in rows]
    fams = [r["family"] for r in rows]
    colors = [_color_for(f) for f in fams]

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.28), 6))
    ax.bar(range(len(rows)), f1s, color=colors)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("F1 macro (%)")
    ax.set_title("Per-algorithm F1 macro (sorted)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(handles=_legend_for_families(fams), loc="upper right",
              fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_precision_recall(rows: List[Dict[str, str]], out: Path) -> None:
    rows = [r for r in rows if "precision_macro" in r and "recall_macro" in r]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 100], [0, 100], color="#bbb", linestyle="--", linewidth=0.8,
            label="P = R")
    for r in rows:
        x = float(r["precision_macro"]) * 100
        y = float(r["recall_macro"]) * 100
        ax.scatter(x, y, color=_color_for(r["family"]), s=42,
                   edgecolors="white", linewidths=0.6)
        ax.annotate(r["algo"], (x, y), fontsize=6, alpha=0.85,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Precision macro (%)")
    ax.set_ylabel("Recall macro (%)")
    ax.set_title("Precision vs recall (macro avg)")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.grid(linestyle=":", alpha=0.4)
    fams = [r["family"] for r in rows]
    ax.legend(handles=_legend_for_families(fams) +
              [plt.Line2D([0], [0], color="#bbb", linestyle="--",
                          label="P = R")],
              fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _parse_per_class_f1(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not s:
        return out
    for tok in s.split(";"):
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            pass
    return out


def _parse_confusion(s: str):
    if not s:
        return None
    try:
        rows = [list(map(int, r.split(","))) for r in s.split("|")]
        return np.array(rows, dtype=int)
    except Exception:
        return None


def plot_per_class_f1(rows: List[Dict[str, str]], out: Path,
                      top_n: int = 10) -> None:
    rows = [r for r in rows if r.get("per_class_f1")]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: -float(r.get("f1_macro", 0.0)))[:top_n]
    parsed = [(r["algo"], r["family"], _parse_per_class_f1(r["per_class_f1"]))
              for r in rows]
    classes: List[str] = []
    for _, _, d in parsed:
        for c in d:
            if c not in classes:
                classes.append(c)
    if not classes:
        return
    n_alg = len(parsed)
    n_cls = len(classes)
    width = 0.8 / max(n_cls, 1)
    fig, ax = plt.subplots(figsize=(max(8, n_alg * 0.9), 6))
    x = np.arange(n_alg)
    cmap = plt.get_cmap("tab10")
    for ci, cls in enumerate(classes):
        vals = [parsed[ai][2].get(cls, 0.0) * 100 for ai in range(n_alg)]
        ax.bar(x + ci * width - 0.4 + width / 2, vals, width,
               color=cmap(ci % 10), label=cls)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in parsed], rotation=45, ha="right",
                       fontsize=9)
    ax.set_ylabel("Per-class F1 (%)")
    ax.set_title(f"Per-class F1 — top {len(parsed)} algos by F1 macro")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, ncol=min(n_cls, 4), loc="upper right",
              title="class")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_confusion_matrices(rows: List[Dict[str, str]], out_dir: Path,
                            top_n: int = 4) -> List[Path]:
    rows = [r for r in rows if r.get("confusion") and r.get("per_class_f1")]
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: -float(r.get("f1_macro", 0.0)))[:top_n]
    written: List[Path] = []
    for r in rows:
        cm = _parse_confusion(r["confusion"])
        if cm is None:
            continue
        labels = list(_parse_per_class_f1(r["per_class_f1"]).keys())
        if len(labels) != cm.shape[0]:
            labels = [str(i) for i in range(cm.shape[0])]
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{r['algo']}  (F1m={float(r['f1_macro']):.3f})")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                colour = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                        color=colour, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fname = out_dir / f"07_confusion_{r['algo']}.png"
        fig.savefig(fname, dpi=130)
        plt.close(fig)
        written.append(fname)
    return written


def plot_board_vs_host(train_rows: List[Dict[str, str]],
                       sweep_rows: List[Dict[str, str]],
                       out: Path) -> None:
    train_by_algo = {r["algo"]: r for r in train_rows}
    paired: List[Tuple[str, str, float, float, int]] = []
    for s in sweep_rows:
        algo = s.get("prep_algo") or s.get("algo")
        if algo not in train_by_algo:
            continue
        t = train_by_algo[algo]
        paired.append((algo, t["family"],
                       float(t["test_acc"]) * 100,
                       float(s["accuracy"]) * 100,
                       int(s["mean_infer_us"])))
    if not paired:
        print("[plot] no overlap between train and sweep CSVs; skipping "
              "board-vs-host plot", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 100], [0, 100], color="#bbb", linestyle="--", linewidth=0.8,
            label="board = host")
    for algo, fam, host_acc, board_acc, _ in paired:
        ax.scatter(host_acc, board_acc, color=_color_for(fam), s=48,
                   edgecolors="white", linewidths=0.6)
        ax.annotate(algo, (host_acc, board_acc), fontsize=7, alpha=0.9,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Host test accuracy (%)")
    ax.set_ylabel("On-board accuracy (%)")
    ax.set_title("Sim-to-real: host vs board accuracy")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.grid(linestyle=":", alpha=0.4)
    fams = [p[1] for p in paired]
    ax.legend(handles=_legend_for_families(fams) +
              [plt.Line2D([0], [0], color="#bbb", linestyle="--",
                          label="board = host")],
              fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="algo_train_results.csv",
                    help="path to algo_train results CSV")
    ap.add_argument("--sweep", default=None,
                    help="path to sweep_results.csv (optional, enables "
                         "board-vs-host plot)")
    ap.add_argument("--out", default="plots",
                    help="output directory (created if missing)")
    args = ap.parse_args(argv)

    train_path = Path(args.train)
    if not train_path.is_file():
        print(f"error: {train_path} not found", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_train(train_path)
    print(f"[plot] loaded {len(rows)} rows from {train_path}")

    p1 = out_dir / "01_test_accuracy_bar.png"
    p2 = out_dir / "02_train_vs_test.png"
    p3 = out_dir / "03_acc_vs_prep_us.png"
    plot_test_accuracy(rows, p1)
    plot_train_vs_test(rows, p2)
    plot_acc_vs_prep(rows, p3)
    print(f"[plot] wrote {p1}\n[plot] wrote {p2}\n[plot] wrote {p3}")

    if rows and "f1_macro" in rows[0]:
        p4 = out_dir / "04_f1_macro_bar.png"
        p5 = out_dir / "05_precision_recall.png"
        p6 = out_dir / "06_per_class_f1.png"
        plot_f1_macro(rows, p4)
        plot_precision_recall(rows, p5)
        plot_per_class_f1(rows, p6, top_n=10)
        print(f"[plot] wrote {p4}\n[plot] wrote {p5}\n[plot] wrote {p6}")
        cms = plot_confusion_matrices(rows, out_dir, top_n=4)
        for c in cms:
            print(f"[plot] wrote {c}")
    else:
        print("[plot] no F1 columns in CSV; re-run algo_train to refresh")

    if args.sweep:
        sweep_path = Path(args.sweep)
        if not sweep_path.is_file():
            print(f"warn: {sweep_path} not found; skipping board plot",
                  file=sys.stderr)
        else:
            sweep_rows = _load_sweep(sweep_path)
            print(f"[plot] loaded {len(sweep_rows)} rows from {sweep_path}")
            p_b = out_dir / "08_board_vs_host.png"
            plot_board_vs_host(rows, sweep_rows, p_b)
            print(f"[plot] wrote {p_b}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
