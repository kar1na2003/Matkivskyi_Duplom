#!/usr/bin/env python3
"""Generate detailed per-family comparison plots + best-of-each-family summary.

Creates plots in /Users/maksum/Desktop/modusmate-bundle/plots/
Uses data from camera_bench_report.csv.
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------- Configuration ----------
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'camera_bench_report.csv')
OUT_DIR = '/Users/maksum/Desktop/modusmate-bundle/plots'
os.makedirs(OUT_DIR, exist_ok=True)

# Family colors
FAMILY_COLORS = {
    'basics': '#4CAF50',
    'edges': '#2196F3',
    'blobs': '#FF9800',
    'keypoints': '#9C27B0',
    'thresholding': '#F44336',
    'texture': '#795548',
    'ridge': '#607D8B',
    'morphology': '#009688',
}

SCENARIO_COLORS = {
    'UART': '#1976D2',
    'Екран': '#FF8F00',
    'Проектор': '#D32F2F',
}


def load_data():
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def plot_family_accuracy(rows, family, ax=None):
    """Grouped bar chart: accuracy across 3 scenarios for one family."""
    family_rows = [r for r in rows if r['family'] == family]
    family_rows.sort(key=lambda r: float(r['uart_acc']), reverse=True)

    algos = [r['algo'] for r in family_rows]
    uart = [float(r['uart_acc']) for r in family_rows]
    screen = [float(r['screen_acc']) for r in family_rows]
    proj = [float(r['proj_acc']) for r in family_rows]

    x = np.arange(len(algos))
    width = 0.25

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(10, len(algos) * 1.2), 5))

    bars1 = ax.bar(x - width, uart, width, label='UART', color=SCENARIO_COLORS['UART'], alpha=0.85)
    bars2 = ax.bar(x, screen, width, label='Екран', color=SCENARIO_COLORS['Екран'], alpha=0.85)
    bars3 = ax.bar(x + width, proj, width, label='Проектор', color=SCENARIO_COLORS['Проектор'], alpha=0.85)

    ax.set_xlabel('Алгоритм')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Родина «{family}» — порівняння accuracy у 3 сценаріях')
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.333, color='gray', linestyle='--', alpha=0.5, label='Random (33.3%)')

    # Value labels on UART bars
    for bar, val in zip(bars1, uart):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.0%}', ha='center', va='bottom', fontsize=7, color=SCENARIO_COLORS['UART'])

    return ax


def plot_family_f1_metrics(rows, family):
    """3-panel plot: F1, Precision, Recall for one family."""
    family_rows = [r for r in rows if r['family'] == family]
    family_rows.sort(key=lambda r: float(r['uart_f1']), reverse=True)

    algos = [r['algo'] for r in family_rows]
    metrics = {
        'F1': ('uart_f1', 'screen_f1', 'proj_f1'),
        'Precision': ('uart_precision', 'screen_precision', 'proj_precision'),
        'Recall': ('uart_recall', 'screen_recall', 'proj_recall'),
    }

    fig, axes = plt.subplots(1, 3, figsize=(max(14, len(algos) * 1.5), 5))
    fig.suptitle(f'Родина «{family}» — F1 / Precision / Recall', fontsize=13)

    for ax, (metric_name, (col_u, col_s, col_p)) in zip(axes, metrics.items()):
        x = np.arange(len(algos))
        width = 0.25

        uart_vals = [float(r[col_u]) for r in family_rows]
        screen_vals = [float(r[col_s]) for r in family_rows]
        proj_vals = [float(r[col_p]) for r in family_rows]

        ax.bar(x - width, uart_vals, width, color=SCENARIO_COLORS['UART'], alpha=0.85)
        ax.bar(x, screen_vals, width, color=SCENARIO_COLORS['Екран'], alpha=0.85)
        ax.bar(x + width, proj_vals, width, color=SCENARIO_COLORS['Проектор'], alpha=0.85)

        ax.set_title(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Score')
    # Common legend
    handles = [Patch(color=c, label=l) for l, c in SCENARIO_COLORS.items()]
    fig.legend(handles=handles, loc='upper right', ncol=3, fontsize=9)
    plt.tight_layout()
    return fig


def plot_family_degradation(rows, family):
    """Horizontal bar chart showing degradation Δ from UART for each algo."""
    family_rows = [r for r in rows if r['family'] == family]
    family_rows.sort(key=lambda r: float(r['uart_acc']) - float(r['proj_acc']))

    algos = [r['algo'] for r in family_rows]
    delta_screen = [float(r['screen_acc']) - float(r['uart_acc']) for r in family_rows]
    delta_proj = [float(r['proj_acc']) - float(r['uart_acc']) for r in family_rows]

    fig, ax = plt.subplots(figsize=(9, max(4, len(algos) * 0.5)))
    y = np.arange(len(algos))
    height = 0.35

    ax.barh(y - height/2, delta_screen, height, color=SCENARIO_COLORS['Екран'],
            alpha=0.85, label='Екран vs UART')
    ax.barh(y + height/2, delta_proj, height, color=SCENARIO_COLORS['Проектор'],
            alpha=0.85, label='Проектор vs UART')

    ax.set_yticks(y)
    ax.set_yticklabels(algos, fontsize=9)
    ax.set_xlabel('ΔAccuracy (відносно UART)')
    ax.set_title(f'Родина «{family}» — деградація accuracy при камерних сценаріях')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.legend(loc='lower left')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_best_per_family(rows):
    """Select best algo from each family, compare them side-by-side."""
    families = sorted(set(r['family'] for r in rows))
    best = []
    for fam in families:
        fam_rows = [r for r in rows if r['family'] == fam]
        # Best by UART accuracy
        top = max(fam_rows, key=lambda r: float(r['uart_acc']))
        best.append(top)

    best.sort(key=lambda r: float(r['uart_acc']), reverse=True)

    algos = [f"{r['algo']}\n({r['family']})" for r in best]
    uart = [float(r['uart_acc']) for r in best]
    screen = [float(r['screen_acc']) for r in best]
    proj = [float(r['proj_acc']) for r in best]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(algos))
    width = 0.25

    ax.bar(x - width, uart, width, color=SCENARIO_COLORS['UART'], alpha=0.85, label='UART')
    ax.bar(x, screen, width, color=SCENARIO_COLORS['Екран'], alpha=0.85, label='Екран')
    ax.bar(x + width, proj, width, color=SCENARIO_COLORS['Проектор'], alpha=0.85, label='Проектор')

    # Color bars by family
    for i, r in enumerate(best):
        color = FAMILY_COLORS.get(r['family'], '#999999')
        ax.scatter(i - width, uart[i] + 0.03, color=color, s=60, zorder=5, marker='s')

    ax.set_xlabel('Алгоритм (родина)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Найкращий алгоритм з кожної родини — порівняння у 3 сценаріях')
    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=9)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # Add family color legend
    family_handles = [Patch(color=FAMILY_COLORS[f], label=f) for f in sorted(FAMILY_COLORS.keys())]
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.legend(handles=family_handles, loc='lower right', ncol=2, fontsize=8, title='Родина')

    plt.tight_layout()
    return fig


def plot_family_pareto(rows, family):
    """Scatter: accuracy vs processing time for one family, 3 scenarios."""
    family_rows = [r for r in rows if r['family'] == family]

    fig, ax = plt.subplots(figsize=(9, 6))

    for r in family_rows:
        t = float(r['algo_us']) / 1000  # ms
        u = float(r['uart_acc'])
        s = float(r['screen_acc'])
        p = float(r['proj_acc'])

        ax.scatter(t, u, color=SCENARIO_COLORS['UART'], s=80, alpha=0.8, zorder=3)
        ax.scatter(t, s, color=SCENARIO_COLORS['Екран'], s=60, alpha=0.7, marker='^', zorder=3)
        ax.scatter(t, p, color=SCENARIO_COLORS['Проектор'], s=60, alpha=0.7, marker='v', zorder=3)

        # Connect same algo with vertical line
        ax.plot([t, t], [p, u], color='gray', alpha=0.3, linewidth=1)

        # Label
        ax.annotate(r['algo'], (t, u), fontsize=7, ha='left', va='bottom',
                    xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel('Час обробки (мс)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Родина «{family}» — Pareto: accuracy vs час')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.25, 1.0)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=SCENARIO_COLORS['UART'],
                   markersize=10, label='UART'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=SCENARIO_COLORS['Екран'],
                   markersize=10, label='Екран'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor=SCENARIO_COLORS['Проектор'],
                   markersize=10, label='Проектор'),
    ]
    ax.legend(handles=handles, loc='lower right')
    plt.tight_layout()
    return fig


def plot_stability_ranking(rows):
    """Rank all algos by stability (smallest drop from UART to Projector)."""
    data = []
    for r in rows:
        delta = float(r['uart_acc']) - float(r['proj_acc'])
        if float(r['uart_acc']) > 0.5:  # filter out random-level algos
            data.append((r['algo'], r['family'], delta, float(r['uart_acc']), float(r['proj_acc'])))

    data.sort(key=lambda x: x[2])  # smallest degradation first
    top20 = data[:20]

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(top20))
    algos = [f"{d[0]} ({d[1]})" for d in top20]
    deltas = [d[2] for d in top20]
    colors = [FAMILY_COLORS.get(d[1], '#999') for d in top20]

    bars = ax.barh(y, deltas, color=colors, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(algos, fontsize=9)
    ax.set_xlabel('ΔAccuracy (UART → Проектор)')
    ax.set_title('Топ-20 найстабільніших алгоритмів (мінімальна деградація)')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, deltas):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f'−{val:.1%}', va='center', fontsize=8)

    family_handles = [Patch(color=FAMILY_COLORS[f], label=f) for f in sorted(FAMILY_COLORS.keys())]
    ax.legend(handles=family_handles, loc='lower right', ncol=2, fontsize=8)
    plt.tight_layout()
    return fig


def main():
    rows = load_data()
    families = sorted(set(r['family'] for r in rows))
    print(f"Loaded {len(rows)} rows, {len(families)} families: {families}")

    # 1. Per-family accuracy comparison
    for fam in families:
        fig, ax = plt.subplots(figsize=(max(10, len([r for r in rows if r['family'] == fam]) * 1.3), 5))
        plot_family_accuracy(rows, fam, ax)
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f'family_{fam}_accuracy.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')

    # 2. Per-family F1/Precision/Recall
    for fam in families:
        fig = plot_family_f1_metrics(rows, fam)
        path = os.path.join(OUT_DIR, f'family_{fam}_f1_metrics.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')

    # 3. Per-family degradation
    for fam in families:
        fig = plot_family_degradation(rows, fam)
        path = os.path.join(OUT_DIR, f'family_{fam}_degradation.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')

    # 4. Per-family Pareto
    for fam in families:
        fig = plot_family_pareto(rows, fam)
        path = os.path.join(OUT_DIR, f'family_{fam}_pareto.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')

    # 5. Best per family
    fig = plot_best_per_family(rows)
    path = os.path.join(OUT_DIR, 'best_per_family.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

    # 6. Stability ranking
    fig = plot_stability_ranking(rows)
    path = os.path.join(OUT_DIR, 'stability_ranking.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

    print(f"\nTotal plots generated: {4 * len(families) + 2}")
    print("Per-family: accuracy, f1_metrics, degradation, pareto")
    print("Summary: best_per_family, stability_ranking")


if __name__ == '__main__':
    main()
