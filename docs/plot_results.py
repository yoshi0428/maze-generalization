"""Grouped bar charts of the test accuracy in docs/SUMMARY.md (Results table).

Run from the repository root:
    python docs/plot_results.py
Writes docs/results_perfect.png and docs/results_loops.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Values copied from the Results table in docs/SUMMARY.md (% of solved mazes).
# Each run: (color, perfect-maze results, loop-maze results).
PERFECT_SIZES = [31, 63, 127, 255, 511]
LOOPS_SIZES = [15, 31, 63, 127]
RUNS = {
    "dtnet_s15_noloops": ("#ee7fac", [100, 100, 99, 92.5, 62.5], [0, 0, 0, 0]),
    "dtnet_s15_noloops_zeroinit": ("#257de0", [100, 100, 100, 100, 100], [0, 0, 0, 0]),
    "dtnet_s15_dfs": ("#f47230", [90.8, 55.5, 12, 0, 0], [0, 0, 0, 0]),
    "dtnet_s15_loops": ("#00b894", [70.8, 16.5, 5, 2.5, 0], [94.4, 54, 14.5, 5]),
    "ff_s15_noloops": ("#f2a600", [28.8, 0, 0, 0, 0], [3.2, 0, 0, 0]),
}
INK, INK_2, GRID = "#191919", "#666666", "#e6e6e6"
OUT_DIR = Path(__file__).parent


def plot(sizes, idx, title, filename):
    fig, ax = plt.subplots(figsize=(1.9 * len(sizes) + 2, 5))
    n = len(RUNS)
    width = 0.84 / n
    x = np.arange(len(sizes))
    for i, (name, run) in enumerate(RUNS.items()):
        color, v = run[0], np.array(run[1 + idx], dtype=float)
        pos = x + (i - (n - 1) / 2) * width
        # Draw zero values as a thin stub so a 0% bar is visible.
        ax.bar(pos, np.maximum(v, 0.6), width * 0.9, color=color, label=name,
               edgecolor="white", linewidth=1)
        for p, val in zip(pos, v):
            ax.text(p, max(val, 0.6) + 1.5, f"{val:g}", ha="center", va="bottom",
                    fontsize=7, color=INK_2)
    ax.set_xticks(x, [f"{s}×{s}" for s in sizes])
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    ax.set_xlabel("Maze size", color=INK_2)
    ax.set_ylabel("Solved mazes (%)", color=INK_2)
    ax.set_ylim(0, 108)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3,
              frameon=False, fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / filename
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    plot(PERFECT_SIZES, 0, "Perfect mazes (no loops)", "results_perfect.png")
    plot(LOOPS_SIZES, 1, "Mazes with loops (10% of walls removed)", "results_loops.png")


if __name__ == "__main__":
    main()
