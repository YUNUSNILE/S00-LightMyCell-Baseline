"""Report figure: Val LMC-score per validation for E1b (team baseline), E1g (S01 v2) and E1h (S00 v2), Task 1 and Task 2.

  python tools/plot_report_curves.py   ->  reports/S00_compare/fig_val_curves.png

Reads val_log.csv of each run (Task 1: lmc_score; Task 2: lmc_score_mip). E1b has no Task 2 head.
Colours: the first three slots of the dataviz reference categorical palette (validated all-pairs, light mode);
every line is also labelled directly, and the values are in the report tables.
"""
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = [  # (label, val_log glob, colour)
    ("E1g (S01_v2)", "S01_v2/runs/E1g_*/val_log.csv", "#257de0"),
    ("E1h (S00_v2)", "S00_v2/runs/E1h_*/val_log.csv", "#f47230"),
    ("E1b (baseline)", "S00_v2/reference_run/E1b_*/val_log.csv", "#00b894"),
]
TEXT, MUTED, GRID = "#191919", "#666666", "#e6e6e6"


def main():
    plt.rcParams.update({"font.family": "Calibri", "font.size": 8.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.75), sharey=True)
    for ax, col, title in zip(axes, ("lmc_score", "lmc_score_mip"), ("Task 1 (best-focus planes)", "Task 2 (MIP)")):
        ends = []
        for label, pat, colour in RUNS:
            vl = pd.read_csv(glob.glob(str(ROOT / pat))[0])
            if col not in vl or vl[col].isna().all():
                continue
            ax.plot(vl.epoch, vl[col], color=colour, lw=1.6, marker="o", ms=4.2, mec="white", mew=0.8, label=label)
            ends.append((float(vl[col].iloc[-1]), label.split()[0]))
        # direct labels at the line ends, nudged apart, in the text colour
        ends.sort()
        y_prev = -1.0
        for y, name in ends:
            y_lab = max(y, y_prev + 0.035)
            ax.annotate(name, (50, y), xytext=(53, y_lab), textcoords="data", color=TEXT, fontsize=8,
                        va="center", annotation_clip=False)
            y_prev = y_lab
        ax.set_title(title, color=TEXT, fontsize=9.5, fontweight="bold", loc="left")
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, 52)
        ax.set_xticks(range(0, 51, 10))
        ax.set_ylim(0.10, 0.65)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Val LMC-score")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 0.97, 1))
    out = ROOT / "S00_v2/reports/S00_compare/fig_val_curves.png"
    fig.savefig(out, dpi=220, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
