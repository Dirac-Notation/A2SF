"""Paper §3 Observation 2 — split into two complementary figures.

Generates:
  obs1_jaccard_recovery.pdf            — single-query vs cumulative optimal Jaccard
                                  (4 task panels, 1 row)
  obs1_sigmoid_band.pdf — optimal coefficient range with sigmoid fits
                                  (4 task panels, 1 row)

Both figures share the same task layout and use textwidth-friendly fonts.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit

rcParams.update({
    "font.family": "serif", "font.size": 12,
    "axes.labelsize": 13, "axes.titlesize": 14,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 13, "axes.linewidth": 1.0,
    "figure.dpi": 180,
})

BUDGET = int(os.environ.get("BUDGET", "128"))
_PLOTS = "/home/smp9898/A2SF/experiments/temporal_bias/plots"
BASE = _PLOTS if BUDGET == 128 else os.path.join(_PLOTS, f"b{BUDGET}")
SUFFIX = "" if BUDGET == 128 else f"_b{BUDGET}"
TASKS = [
    ("Single-doc_QA/qasper",     "Single-doc QA"),
    ("Multi-doc_QA/hotpotqa",    "Multi-doc QA"),
    ("Summarization/gov_report", "Summarization"),
    ("Few_Shot/samsum",          "Few-Shot"),
]


def sigmoid(x, a, b):
    """Descending 2-parameter sigmoid w(d) = 1 / (1 + e^{a (d - b)})."""
    return 1.0 / (1.0 + np.exp(a * (x - b)))


def fit_sigmoid(d, y, W):
    """Fit the standard 2-parameter descending sigmoid. Returns (fit_y, a, b).
    Min a bound is set so the sigmoid actually transitions over the panel
    (otherwise a noisy non-monotone envelope collapses to a flat 0.5)."""
    try:
        popt, _ = curve_fit(sigmoid, d.astype(float), y,
                            p0=[0.1, float(W) / 4.0],
                            bounds=([0.02, 0.0], [5.0, float(W)]),
                            maxfev=20000)
        a, b = popt
        return sigmoid(d, *popt), float(a), float(b)
    except Exception:
        return np.full_like(d, np.nan, dtype=float), np.nan, np.nan


def plot_jaccard(out_dir):
    """Top: single-query vs cumulative-optimal Jaccard."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2))
    colors = plt.get_cmap("tab10").colors
    h_proxy = []

    for col, ((path, label), c) in enumerate(zip(TASKS, colors)):
        m  = np.load(os.path.join(BASE, path, "metrics.npz"))
        cd = np.load(os.path.join(BASE, path, "coord_descent.npz"))
        br     = m["br"].mean(axis=0)          # single-query Jaccard, shape (W,)
        # j_weight: per-chunk running Jaccard (G,); expand to per-token grid
        jw_raw = cd["j_weight"].mean(axis=0)   # (G,)
        chunk  = int(cd["chunk"]) if "chunk" in cd.files else 1
        W      = len(br)
        # Repeat each chunk value across its chunk tokens so lengths match
        cs_opt = np.repeat(jw_raw, chunk)[:W]

        # Fixed-weight cumulative Jaccards (present only in re-run npz)
        def load_fixed(key):
            if key in cd.files:
                return np.repeat(cd[key].mean(axis=0), chunk)[:W]
            return None

        cs_unif  = load_fixed("j_uniform")   # H2O-like: all queries equal
        cs_snap  = load_fixed("j_snap16")    # SnapKV-16: only nearest 16 queries
        cs_sig   = load_fixed("j_sigmoid")   # sigmoid-fit weights

        d = np.arange(W)

        ax = axes[col]
        l1, = ax.plot(d, br,     color="gray",  lw=1.5, ls="--",
                      label=r"$J_{\mathrm{single}}(d)$: single query")
        l2, = ax.plot(d, cs_opt, color="black", lw=2.0,
                      label=r"$J_{\mathrm{weight}}(d)$: optimally weighted")

        l3 = l4 = l5 = None
        if cs_unif is not None:
            l3, = ax.plot(d, cs_unif, color="steelblue", lw=1.5, ls="-",
                          label=r"$J_{\mathrm{uniform}}(d)$: uniform (H2O)")
        if cs_snap is not None:
            l4, = ax.plot(d, cs_snap, color="forestgreen", lw=1.5, ls="-",
                          label=r"$J_{\mathrm{snap16}}(d)$: SnapKV-16")
        if cs_sig is not None:
            l5, = ax.plot(d, cs_sig,  color="darkorange",  lw=1.5, ls="--",
                          label=r"$J_{\mathrm{sigmoid}}(d)$: sigmoid fit")

        ax.set_xlim(0, W - 1)
        ax.set_ylim(0.0, 0.45)
        ax.invert_xaxis()
        if col == 0:
            ax.set_ylabel("Jaccard similarity")
            h_proxy = [h for h in [l1, l2, l3, l4, l5] if h is not None]
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    all_labels = [
        r"$J_{\mathrm{single}}$: single query",
        r"$J_{\mathrm{weight}}$: optimally weighted",
        r"$J_{\mathrm{uniform}}$: uniform (H2O)",
        r"$J_{\mathrm{snap16}}$: SnapKV-16",
        r"$J_{\mathrm{sigmoid}}$: sigmoid fit",
    ]

    fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.16,
                        wspace=0.30)
    fig.text(0.525, 0.03,
             r"query distance $d$   ($\leftarrow$ older  $\cdot$  recent $\rightarrow$)",
             ha="center", fontsize=13)
    fig.legend(h_proxy, all_labels[:len(h_proxy)],
               loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=min(5, len(h_proxy)), frameon=False)

    fig.savefig(os.path.join(out_dir, f"obs1_jaccard_recovery{SUFFIX}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"obs1_jaccard_recovery{SUFFIX}.png"), bbox_inches="tight")
    plt.close(fig)


def plot_optimal_coefficient(out_dir):
    """Per-prompt min-max range band + mean + (0,1)-pinned sigmoid fit."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.0))
    colors = plt.get_cmap("tab10").colors
    h_proxy = []

    for col, ((path, label), c) in enumerate(zip(TASKS, colors)):
        cd = np.load(os.path.join(BASE, path, "coord_descent.npz"))
        w_cd = cd["w_cd"]
        n_prompts, G = w_cd.shape
        chunk = int(cd["chunk"]) if "chunk" in cd.files else 1
        W = int(cd["window"]) if "window" in cd.files else G * chunk
        d = np.arange(G) * chunk + (chunk - 1) / 2.0

        w_min  = w_cd.min(axis=0)
        w_max  = w_cd.max(axis=0)
        w_mean = w_cd.mean(axis=0)
        fit_y, a_fit, b_fit = fit_sigmoid(d, w_max, W)
        # Visual shift: add (1 - σ(0)) so the displayed curve passes through
        # (0, 1) without altering the fitted (a, b) parameters.
        if np.isfinite(a_fit) and np.isfinite(b_fit):
            sigma0 = sigmoid(0.0, a_fit, b_fit)
            fit_y_disp = fit_y + (1.0 - sigma0)
        else:
            fit_y_disp = fit_y

        ax = axes[col]
        l_band = ax.fill_between(d, w_min, w_max, color=c, alpha=0.25,
                                  linewidth=0,
                                  label="optimal weight range")
        l_fit,  = ax.plot(d, fit_y_disp, "--", color="black", lw=1.8,
                          label="sigmoid fit")

        if np.isfinite(a_fit) and np.isfinite(b_fit):
            ax.text(0.05, 0.92,
                    f"$a = {a_fit:.2f}$\n$b = {b_fit:.1f}$",
                    transform=ax.transAxes, fontsize=11,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="white", edgecolor="0.8", alpha=0.85))
        if col == 0:
            h_proxy = [l_band, l_fit]

        ax.set_xlim(0, W - 1)
        ax.invert_xaxis()
        ax.set_ylim(-0.05, 1.10)
        if col == 0:
            ax.set_ylabel("optimal coefficient  $w_d$")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.16,
                        wspace=0.30)
    fig.text(0.525, 0.03,
             r"query distance $d$   ($\leftarrow$ older  $\cdot$  recent $\rightarrow$)",
             ha="center", fontsize=13)
    if h_proxy:
        fig.legend(h_proxy,
                   ["optimal weight range", "sigmoid fit"],
                   loc="upper center", bbox_to_anchor=(0.5, 0.99),
                   ncol=2, frameon=False)

    fig.savefig(os.path.join(out_dir, f"obs1_sigmoid_band{SUFFIX}.pdf"),
                 bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"obs1_sigmoid_band{SUFFIX}.png"),
                 bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_jaccard(out_dir)
    plot_optimal_coefficient(out_dir)
    print(f"saved → {out_dir}/obs1_jaccard_recovery{SUFFIX}.{{pdf,png}} (budget={BUDGET})")
    print(f"saved → {out_dir}/obs1_sigmoid_band{SUFFIX}.{{pdf,png}} (budget={BUDGET})")


if __name__ == "__main__":
    main()
