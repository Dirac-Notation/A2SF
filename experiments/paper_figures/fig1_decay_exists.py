"""Figure 1 (Observation 1): Query importance decays with distance.

Pure empirical curve — shows monotonic decrease of the optimal per-position
coefficient as queries move away from the most recent position.

Data: experiments/temporal_bias/plots/<Task>/<Dataset>/metrics.npz
  - 'oc' (n_samples, W): greedy-optimal per-query weight found by
    analyze_optimal_coefficient() in optimal.py.

Single panel, overlayed tasks. No fits — those are in Figure 2.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.ndimage import uniform_filter1d


rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
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
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    colors = plt.get_cmap("tab10").colors

    for (path, label), c in zip(TASKS, colors):
        data = np.load(os.path.join(BASE, path, "metrics.npz"), allow_pickle=True)
        oc = data["oc"]                                       # (n, W)
        mean = oc.mean(axis=0)
        smooth = uniform_filter1d(mean, size=8, mode="nearest")
        d = np.arange(len(smooth))
        # Shaded IQR band for visual of per-sample spread
        q25 = np.percentile(oc, 25, axis=0)
        q75 = np.percentile(oc, 75, axis=0)
        q25s = uniform_filter1d(q25, size=12, mode="nearest")
        q75s = uniform_filter1d(q75, size=12, mode="nearest")
        ax.fill_between(d, q25s, q75s, color=c, alpha=0.15)
        ax.plot(d, smooth, color=c, lw=2.0, label=label)

    ax.set_xlabel("distance from most recent query position")
    ax.set_ylabel("optimal query weight  $w(q)$")
    ax.set_xlim(0, 128)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    out_dir = "/home/smp9898/A2SF/experiments/paper_figures"
    fig.savefig(os.path.join(out_dir, "fig1_decay_exists.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig1_decay_exists.png"), bbox_inches="tight")
    print(f"saved → {out_dir}/fig1_decay_exists.{{pdf,png}}")


if __name__ == "__main__":
    main()
