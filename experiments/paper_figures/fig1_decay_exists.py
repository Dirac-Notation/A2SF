"""Figure 1 (Observation 1): Individual query information quality decays with distance.

For a single query q, we pre-select top-B keys using only that query's attention
(no accumulation). We then measure the Jaccard overlap between the selected set and
the ground-truth answer positions. This is stored as `br` (block-hit rate) per sample.

Here `br[i, d]` = hit rate when using only the query at distance d from the most
recent position for sample i.

Large `br` means: that single query's attention accurately points at answer tokens.
The curve decaying with distance means: older queries carry less accurate signal.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.ndimage import uniform_filter1d


rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "figure.dpi": 150,
})

BASE = "/home/smp9898/A2SF/experiments/temporal_bias/plots"
TASKS = [
    ("Single-doc_QA/qasper", "Single-doc QA"),
    ("Multi-doc_QA/hotpotqa", "Multi-doc QA"),
    ("Summarization/gov_report", "Summarization"),
    ("Few_Shot/samsum", "Few-Shot"),
]


def main():
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), sharey=False)
    colors = plt.get_cmap("tab10").colors

    for ax, ((path, label), c) in zip(axes, zip(TASKS, colors)):
        data = np.load(os.path.join(BASE, path, "metrics.npz"), allow_pickle=True)
        br = data["br"]                                       # (n, W)
        mean = br.mean(axis=0)
        smooth = uniform_filter1d(mean, size=6, mode="nearest")
        d = np.arange(len(smooth))
        q25 = uniform_filter1d(np.percentile(br, 25, axis=0), size=8, mode="nearest")
        q75 = uniform_filter1d(np.percentile(br, 75, axis=0), size=8, mode="nearest")
        ax.fill_between(d, q25, q75, color=c, alpha=0.25, label="IQR across samples")
        ax.plot(d, smooth, color=c, lw=2.0, label="empirical mean")
        ax.set_xlabel("distance  $d$")
        ax.set_title(label, fontsize=12)
        ax.set_xlim(0, len(smooth) - 1)
        ax.set_ylim(0.0, None)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.95)

    axes[0].set_ylabel("single-query hit rate  $\\mathrm{HR}(d)$")

    plt.tight_layout()
    out_dir = "/home/smp9898/A2SF/experiments/paper_figures"
    fig.savefig(os.path.join(out_dir, "fig1_decay_exists.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig1_decay_exists.png"), bbox_inches="tight")
    print(f"saved → {out_dir}/fig1_decay_exists.{{pdf,png}}")


if __name__ == "__main__":
    main()
