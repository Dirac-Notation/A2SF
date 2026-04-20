"""Figure 2 (Observation 2): The decay shape fits exponential / sigmoid well.

Same oc data as Figure 1, but focus on parametric fits. Four subplots (one per
task) each showing empirical mean + exp/sigmoid fits with their R². The text
labels (caption in paper) make the case that simple 1-2 parameter families
describe the decay.
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


def sigmoid(d, a, b):
    return 1.0 / (1.0 + np.exp(a * d - b))


def exp_decay(d, a):
    return np.exp(-a * d)


def main():
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)

    for ax, (path, label) in zip(axes, TASKS):
        data = np.load(os.path.join(BASE, path, "metrics.npz"), allow_pickle=True)
        oc = data["oc"]
        W = oc.shape[1]
        y = uniform_filter1d(oc.mean(axis=0), size=8, mode="nearest")
        d = np.arange(W)

        popt_s, _ = curve_fit(sigmoid, d, y, p0=[0.1, 5.0], maxfev=5000)
        y_s = sigmoid(d, *popt_s)
        popt_e, _ = curve_fit(exp_decay, d, y, p0=[0.05],
                              maxfev=5000, bounds=(0, 2.0))
        y_e = exp_decay(d, *popt_e)

        ax.plot(d, y, color="black", lw=1.8, label="empirical mean")
        ax.plot(d, y_s, "--", color="tab:blue", lw=1.8,
                label=f"sigmoid  ($R^2$={r2(y, y_s):.2f})")
        ax.plot(d, y_e, ":", color="tab:red", lw=2.0,
                label=f"exponential  ($R^2$={r2(y, y_e):.2f})")
        ax.set_xlabel("distance $d$")
        ax.set_title(label, fontsize=12)
        ax.set_xlim(0, W - 1)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.95)

    axes[0].set_ylabel("optimal query weight  $w(q)$")

    plt.tight_layout()
    out_dir = "/home/smp9898/A2SF/experiments/paper_figures"
    fig.savefig(os.path.join(out_dir, "fig2_parametric_fit.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig2_parametric_fit.png"), bbox_inches="tight")
    print(f"saved → {out_dir}/fig2_parametric_fit.{{pdf,png}}")


if __name__ == "__main__":
    main()
