"""Figure 2 (Observation 2): Older queries still carry useful signal — residual
hit rate stays above random, and the decay shape fits simple exp/sigmoid families.

Per-query standalone hit rate (Fig 1) fit with:
  * sigmoid:     HR(d) = A / (1 + exp(a·d − b)) + HR_∞
  * exponential: HR(d) = (A − HR_∞) · exp(−a·d) + HR_∞

We fit with a non-zero floor so the "older query is not useless" asymptote is
explicit.  Caption makes the point: residual floor > random baseline.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
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


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)


def sigmoid_floor(d, A, a, b, floor):
    return floor + A / (1.0 + np.exp(a * d - b))


def exp_floor(d, A, a, floor):
    return floor + A * np.exp(-a * d)


def main():
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), sharey=False)
    colors = plt.get_cmap("tab10").colors

    for ax, ((path, label), c) in zip(axes, zip(TASKS, colors)):
        data = np.load(os.path.join(BASE, path, "metrics.npz"), allow_pickle=True)
        br = data["br"]                                       # (n, W)
        W = br.shape[1]
        y = uniform_filter1d(br.mean(axis=0), size=6, mode="nearest")
        d = np.arange(W)

        # Sigmoid with floor
        popt_s, _ = curve_fit(sigmoid_floor, d, y, p0=[y[0] - y[-1], 0.2, 5.0, y[-1]],
                              maxfev=10000)
        y_s = sigmoid_floor(d, *popt_s)
        A_s, a_s, b_s, f_s = popt_s

        # Exponential with floor
        popt_e, _ = curve_fit(exp_floor, d, y, p0=[y[0] - y[-1], 0.1, y[-1]],
                              maxfev=10000)
        y_e = exp_floor(d, *popt_e)
        A_e, a_e, f_e = popt_e

        ax.plot(d, y, color=c, lw=2.0, label="empirical mean")
        ax.plot(d, y_s, "--", color="black", lw=1.6,
                label=f"sigmoid  ($R^2$={r2(y, y_s):.2f})")
        ax.plot(d, y_e, ":", color="dimgray", lw=1.8,
                label=f"exponential  ($R^2$={r2(y, y_e):.2f})")

        # Residual floor annotation
        residual = (f_s + f_e) / 2
        ax.axhline(residual, color="gray", lw=1, ls="-.")
        ax.text(W * 0.55, residual + 0.01,
                f"residual ≈ {residual:.2f}", color="gray", fontsize=9)

        ax.set_xlabel("distance  $d$")
        ax.set_title(label, fontsize=12)
        ax.set_xlim(0, W - 1)
        ax.set_ylim(max(0.0, residual - 0.05), None)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.95)

    axes[0].set_ylabel("single-query hit rate  $\\mathrm{HR}(d)$")

    plt.tight_layout()
    out_dir = "/home/smp9898/A2SF/experiments/paper_figures"
    fig.savefig(os.path.join(out_dir, "fig2_parametric_fit.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig2_parametric_fit.png"), bbox_inches="tight")
    print(f"saved → {out_dir}/fig2_parametric_fit.{{pdf,png}}")


if __name__ == "__main__":
    main()
