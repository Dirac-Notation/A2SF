"""Per-task action selection frequency for WAITS, 1B and 8B at B=128.

Reads the WAITS LongBench prediction JSONLs (each line records the (a, b) the
selector picked for that prompt) and aggregates a per-task selection ratio
over the 13-action grid. Renders a 1x2 stacked horizontal bar chart, one
panel per model. Each task bar always sums to 1 (a full-width bar), so
visual emphasis falls on the action mix rather than on grid sparsity.

Color scheme:
  a = 0    -> grey  (the H2O / uniform limit)
  a = 0.01 -> blue family,  centre b small -> light, b large -> dark
  a = 0.1  -> green family, same shading rule
  a = 10   -> red family,   same shading rule

Output:
  experiments/paper_figures/action_analysis/action_distribution.{pdf,png}
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.dpi": 100,
})

with open(os.path.join(ROOT, "config", "task2dataset.json")) as f:
    TASK2DATASET = json.load(f)
DATASET2TASK = {ds: t for t, dss in TASK2DATASET.items() for ds in dss}
TASKS = list(TASK2DATASET.keys())

ACTIONS = (
    [(0.0, 1.0)]
    + [(0.01, b) for b in (1.0, 16.0, 32.0, 128.0)]
    + [(0.1,  b) for b in (1.0, 16.0, 32.0, 128.0)]
    + [(10.0, b) for b in (1.0, 16.0, 32.0, 128.0)]
)


def _slope_shaded_palette() -> list:
    cmaps = {
        0.01: plt.get_cmap("Blues"),
        0.1:  plt.get_cmap("Greens"),
        10.0: plt.get_cmap("Reds"),
    }
    shades = [0.35, 0.55, 0.75, 0.92]
    palette = ["#9e9e9e"]
    for slope in (0.01, 0.1, 10.0):
        for shade in shades:
            palette.append(cmaps[slope](shade))
    return palette


def collect_action_dist(pred_dir: str) -> np.ndarray:
    counts = defaultdict(Counter)
    totals = defaultdict(int)
    for fn in sorted(os.listdir(pred_dir)):
        if not fn.endswith(".jsonl"):
            continue
        ds = fn[:-6]
        if ds not in DATASET2TASK:
            continue
        task = DATASET2TASK[ds]
        with open(os.path.join(pred_dir, fn)) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if "a" not in r or "b" not in r:
                    continue
                a = round(float(r["a"]), 3)
                b = round(float(r["b"]), 3)
                key = None
                for ca, cb in ACTIONS:
                    if abs(a - ca) < 1e-6 and abs(b - cb) < 1e-6:
                        key = (ca, cb)
                        break
                if key is None:
                    continue
                counts[task][key] += 1
                totals[task] += 1
    M = np.zeros((len(TASKS), len(ACTIONS)))
    for ti, t in enumerate(TASKS):
        if totals[t] == 0:
            continue
        for ai, action in enumerate(ACTIONS):
            M[ti, ai] = counts[t].get(action, 0) / totals[t]
    return M


def action_label(action) -> str:
    a, b = action
    if a == 0.0:
        return r"$a\!=\!0$"
    return f"({a:g}, {int(b)})"


def plot_stacked(M: np.ndarray, ax, title: str, palette: list, show_y: bool = True):
    n_tasks = M.shape[0]
    y = np.arange(n_tasks)[::-1]
    left = np.zeros(n_tasks)
    for ai in range(M.shape[1]):
        widths = M[:, ai]
        ax.barh(y, widths, left=left, color=palette[ai],
                edgecolor="white", linewidth=0.5, height=0.78)
        for ti in range(n_tasks):
            w = widths[ti]
            if w >= 0.08:
                ax.text(left[ti] + w / 2.0, y[ti], f"{w:.2f}",
                        ha="center", va="center", fontsize=11,
                        color="white" if w >= 0.25 else "black")
        left += widths
    if show_y:
        ax.set_yticks(y)
        ax.set_yticklabels(TASKS)
    else:
        ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("selection ratio")
    ax.set_title(title, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    pred_1b = os.path.join(ROOT, "result_txt", "backup", "llama3-1b", "128",
                           "llama3-1b_WAITS_128")
    pred_8b = os.path.join(ROOT, "result_txt", "backup", "llama3-8b", "128",
                           "llama3-8b_WAITS_128")

    M_1b = collect_action_dist(pred_1b)
    M_8b = collect_action_dist(pred_8b)
    palette = _slope_shaded_palette()

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4),
                             gridspec_kw={"width_ratios": [1, 1]})
    plot_stacked(M_1b, axes[0], "LLaMA-3.2-1B", palette, show_y=True)
    plot_stacked(M_8b, axes[1], "LLaMA-3.1-8B", palette, show_y=False)

    # Column-major legend: each column groups one slope family.
    #   col 0: a=0 (one entry, padded with blanks)
    #   col 1: (0.01, b) with b in {1, 16, 32, 128}
    #   col 2: (0.1,  b) with b in {1, 16, 32, 128}
    #   col 3: (10,   b) with b in {1, 16, 32, 128}
    blank = Patch(facecolor="none", edgecolor="none", label=" ")
    cols = [
        [ACTIONS[0]] + [None, None, None],
        list(ACTIONS[1:5]),
        list(ACTIONS[5:9]),
        list(ACTIONS[9:13]),
    ]
    # Re-emit in row-major order for matplotlib's row-major legend layout.
    ordered = []
    for row in range(4):
        for col in range(4):
            entry = cols[col][row]
            if entry is None:
                ordered.append(blank)
            else:
                idx = ACTIONS.index(entry)
                ordered.append(Patch(color=palette[idx],
                                     label=action_label(entry)))
    fig.legend(handles=ordered, loc="lower center",
               ncol=4, frameon=False, columnspacing=2.0, handlelength=1.6,
               bbox_to_anchor=(0.5, -0.05))

    fig.subplots_adjust(left=0.12, right=0.99, top=0.90, bottom=0.42, wspace=0.06)

    pdf = os.path.join(out_dir, "action_distribution.pdf")
    png = os.path.join(out_dir, "action_distribution.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {pdf}")
    print(f"saved: {png}")


if __name__ == "__main__":
    main()
