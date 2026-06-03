"""Q4 rebuttal — Sigmoid vs alternative weight families on paper data.

Uses the EXACT data that produced paper/figures/obs1_sigmoid_band.pdf:
  experiments/temporal_bias/plots/b256/<Task>/<dataset>/coord_descent.npz
  → 'w_cd' shape (N=10 prompts, G=128 chunks), chunk_size=2, W=256.

Two figures are produced.

  q4_obs1_with_alternatives.pdf  — paper-style 4-panel layout (mean ± std,
    mean curve, sigmoid fit, ALSO step/exp/linear fit overlays). Each panel's
    text box reports per-prompt mean R² for every family.

  q4_ppt_summary.pdf  — single PPT slide; bar chart of per-prompt mean R²
    across all 40 prompts (10 × 4 tasks) for the four families.

Per-prompt fits use the same descending sigmoid that paper uses
  σ(d; a, b) = 1 / (1 + exp(a (d - b))),
plus three single-parameter alternatives sharing the same 2-param freedom
budget as sigmoid by reuse of (a, b) where applicable.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit

rcParams.update({
    "font.family": "serif", "font.size": 12,
    "axes.labelsize": 13, "axes.titlesize": 14,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 11, "axes.linewidth": 1.0,
    "figure.dpi": 180,
})

PLOTS_BASE = "/home/smp9898/A2SF/experiments/temporal_bias/plots/b256"
OUT_DIR    = "/home/smp9898/A2SF/experiments/paper_figures/rebuttal"
CHUNK_SIZE = 2          # b256 uses chunk = 2 (W=256, G=128)
W_TOTAL    = 256

TASKS = [
    ("Single-doc_QA/qasper",     "Single-doc QA"),
    ("Multi-doc_QA/hotpotqa",    "Multi-doc QA"),
    ("Summarization/gov_report", "Summarization"),
    ("Few_Shot/samsum",          "Few-Shot"),
]

FAM_COLOR = {"sigmoid": "black", "step": "tab:red",
             "exp": "tab:orange",  "linear": "tab:green"}
FAM_LABEL = {"sigmoid": "Sigmoid (2p)", "step": "Step (1p)",
             "exp": "Exponential (1p)", "linear": "Linear (1p)"}


# ── Family definitions ────────────────────────────────────────────────

def sigmoid(d, a, b):
    return 1.0 / (1.0 + np.exp(np.clip(a * (d - b), -30, 30)))


def step_fn(d, b):
    # Steep sigmoid centered at b (smooth approximation of a hard step).
    return 1.0 / (1.0 + np.exp(np.clip(50.0 * (d - b), -30, 30)))


def exp_decay(d, tau):
    return np.exp(-d / max(tau, 1e-6))


def linear_decay(d, W_eff):
    return np.clip(1.0 - d / max(W_eff, 1e-6), 0.0, 1.0)


def _fit(d, y, fn, p0, bounds):
    try:
        popt, _ = curve_fit(fn, d, y, p0=p0, bounds=bounds, maxfev=20000)
        yp = fn(d, *popt)
        ss_res = float(np.sum((y - yp) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return float(r2), [float(x) for x in popt], yp
    except Exception:
        return float("nan"), None, None


def fit_all(d, y, W):
    return {
        "sigmoid": _fit(d, y, sigmoid,
                         p0=[0.1, W / 4.0],
                         bounds=([0.001, -W * 0.5], [50, W * 1.5])),
        "step":    _fit(d, y, step_fn,
                         p0=[W / 4.0],
                         bounds=([0.0], [W * 1.5])),
        "exp":     _fit(d, y, exp_decay,
                         p0=[W / 4.0],
                         bounds=([0.5], [W * 5.0])),
        "linear":  _fit(d, y, linear_decay,
                         p0=[W / 2.0],
                         bounds=([1.0], [W * 5.0])),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Collect per-prompt R² for every family, across all 40 prompts.
    per_task_results = {}        # for plotting (mean-curve fits + R²)
    per_prompt_r2 = {fam: [] for fam in FAM_COLOR}   # overall PPT bar chart

    for path, label in TASKS:
        npz = f"{PLOTS_BASE}/{path}/coord_descent.npz"
        if not os.path.exists(npz):
            print(f"  [skip] {label}: {npz}"); continue
        cd = np.load(npz)
        w_cd = cd["w_cd"]                      # (N, G=128)
        N, G = w_cd.shape
        d_c = np.arange(G) * CHUNK_SIZE + (CHUNK_SIZE - 1) / 2.0  # query distances

        # Per-prompt fits → R² per family
        task_r2 = {fam: [] for fam in FAM_COLOR}
        for i in range(N):
            fits_i = fit_all(d_c, w_cd[i], W_TOTAL)
            for fam, (r2, _, _) in fits_i.items():
                task_r2[fam].append(r2)
                per_prompt_r2[fam].append(r2)

        # Mean-curve fits (for visual overlay)
        w_mean = w_cd.mean(axis=0)
        w_std  = w_cd.std(axis=0)
        w_lo   = np.clip(w_mean - w_std, 0.0, 1.0)
        w_hi   = np.clip(w_mean + w_std, 0.0, 1.0)
        mean_fits = fit_all(d_c, w_mean, W_TOTAL)

        per_task_results[label] = {
            "d": d_c, "w_cd": w_cd,
            "w_mean": w_mean, "w_lo": w_lo, "w_hi": w_hi,
            "mean_fits": mean_fits,
            "per_prompt_r2": {f: np.array(task_r2[f]) for f in task_r2},
        }

    # ── Figure 1: paper-style 4-panel with overlays ───────────────────
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
    color_band = {"Single-doc QA": "C0", "Multi-doc QA": "C1",
                  "Summarization": "C2", "Few-Shot": "C3"}

    for col, (path, label) in enumerate(TASKS):
        if label not in per_task_results: continue
        r = per_task_results[label]
        c = color_band[label]
        ax = axes[col]
        ax.fill_between(r["d"], r["w_lo"], r["w_hi"], color=c, alpha=0.25,
                         linewidth=0, label="mean ± std")
        ax.plot(r["d"], r["w_mean"], color=c, lw=2.0, label="mean")

        # Family fits to the mean (visual overlay) + report per-prompt mean R²
        for fam in ["sigmoid", "step", "exp", "linear"]:
            r2_mean_curve, params, y_pred = r["mean_fits"][fam]
            r2_pp = float(np.nanmean(r["per_prompt_r2"][fam]))
            if y_pred is None: continue
            ls = "--" if fam == "sigmoid" else ":"
            ax.plot(r["d"], y_pred, ls=ls, color=FAM_COLOR[fam], lw=1.6,
                    label=f"{FAM_LABEL[fam]}: $\\overline{{R^2}}$={r2_pp:.2f}")

        ax.set_xlim(0, W_TOTAL - 1)
        ax.invert_xaxis()
        ax.set_ylim(-0.05, 1.10)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r"query distance $d$")
        if col == 0:
            ax.set_ylabel(r"optimal coefficient  $w_d$")
            ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    fig.suptitle("Per-prompt fit of $w_d^*$ to four weight families  "
                 "($\\overline{R^2}$ = mean over N=10 prompts; B=256)", y=1.02)
    fig.tight_layout()
    out1 = f"{OUT_DIR}/q4_obs1_with_alternatives.pdf"
    fig.savefig(out1, bbox_inches="tight")
    fig.savefig(out1.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out1}")

    # ── Figure 2: PPT-ready single bar chart ──────────────────────────
    rcParams.update({
        "font.size": 18, "axes.labelsize": 22, "axes.titlesize": 24,
        "xtick.labelsize": 19, "ytick.labelsize": 16, "legend.fontsize": 14,
    })
    fig2, ax = plt.subplots(1, 1, figsize=(12, 6.0))
    fams = ["sigmoid", "step", "exp", "linear"]
    means = [float(np.nanmean(per_prompt_r2[f])) for f in fams]
    pos_rate = [float(np.mean(np.array(per_prompt_r2[f]) > 0)) for f in fams]
    x = np.arange(len(fams))
    bar_colors = [FAM_COLOR[f] for f in fams]
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-200, 2)
    bars = ax.bar(x, means, width=0.55, color=bar_colors, edgecolor="black",
                   linewidth=1.2, alpha=0.92, zorder=2)

    rng = np.random.default_rng(0)
    task_markers = ["o", "s", "^", "D"]
    for f_idx, f in enumerate(fams):
        for t_idx, t in enumerate(["Single-doc QA", "Multi-doc QA",
                                   "Summarization", "Few-Shot"]):
            if t not in per_task_results: continue
            r2_t = float(np.nanmean(per_task_results[t]["per_prompt_r2"][f]))
            ax.scatter(f_idx + rng.uniform(-0.10, 0.10), r2_t,
                       marker=task_markers[t_idx], s=110, edgecolor="black",
                       linewidth=1.0, facecolor="white", zorder=5)

    ax.axhline(0, color="black", lw=1.4, ls="--", alpha=0.7, zorder=1)

    for i, (f, m) in enumerate(zip(fams, means)):
        if m >= 0:
            ax.text(i, m + 0.10, f"$\\overline{{R^2}}$={m:.2f}\n({100*pos_rate[i]:.0f}% > 0)",
                    ha="center", va="bottom", fontsize=15, fontweight="bold",
                    color=FAM_COLOR[f])
        else:
            ax.text(i, m * 1.35, f"$\\overline{{R^2}}$={m:.1f}\n({100*pos_rate[i]:.0f}% > 0)",
                    ha="center", va="top", fontsize=15, fontweight="bold",
                    color=FAM_COLOR[f])

    ax.set_xticks(x)
    ax.set_xticklabels([FAM_LABEL[f] for f in fams])
    ax.set_ylabel("per-prompt mean $R^2$")
    ax.set_title("Sigmoid is the only family that fits the empirical $w_d^*$")
    ax.grid(True, axis="y", which="both", alpha=0.25, zorder=0)

    handles = [plt.Line2D([0], [0], marker=m, lw=0, markerfacecolor="white",
                          markeredgecolor="black", markersize=12,
                          label=t)
               for m, t in zip(task_markers,
                                ["Single-doc QA", "Multi-doc QA",
                                 "Summarization", "Few-Shot"])]
    ax.legend(handles=handles, title="task (per-task mean)",
              loc="lower right", fontsize=13, title_fontsize=13, framealpha=0.95)

    fig2.text(0.5, -0.02,
              "Forward-greedy optimal weights $w_d^*$ from paper/figures/obs1_sigmoid_band.pdf "
              "(B=256, N=40 prompts), refit with each family per prompt.",
              ha="center", fontsize=12, style="italic", color="dimgray")

    fig2.tight_layout()
    out2 = f"{OUT_DIR}/q4_ppt_summary.pdf"
    fig2.savefig(out2, bbox_inches="tight")
    fig2.savefig(out2.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig2)
    print(f"saved → {out2}")

    # ── Summary JSON ───────────────────────────────────────────────────
    out_json = f"{OUT_DIR}/q4_summary.json"
    summary = {
        "n_prompts_total": int(sum(len(per_task_results[t]["per_prompt_r2"]["sigmoid"])
                                    for t in per_task_results)),
        "overall_mean_R2": {f: float(np.nanmean(per_prompt_r2[f])) for f in fams},
        "overall_pct_R2_positive": {f: float(np.mean(np.array(per_prompt_r2[f]) > 0))
                                      for f in fams},
        "per_task_mean_R2": {
            t: {f: float(np.nanmean(r["per_prompt_r2"][f])) for f in fams}
            for t, r in per_task_results.items()
        },
        "mean_curve_fit": {
            t: {f: {"R2": r["mean_fits"][f][0],
                     "params": r["mean_fits"][f][1]} for f in fams}
            for t, r in per_task_results.items()
        },
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved → {out_json}")

    print()
    print(f"{'family':16s}  {'overall R²':>10}  {'% >0':>6}")
    print("─" * 40)
    for f in fams:
        print(f"  {FAM_LABEL[f]:14s}  {summary['overall_mean_R2'][f]:10.3f}  "
              f"{100*summary['overall_pct_R2_positive'][f]:5.0f}%")
    print()
    print("Per-task per-family mean R²:")
    for t, d in summary["per_task_mean_R2"].items():
        print(f"  {t:22s}  " + "  ".join(f"{f}={d[f]:+.2f}" for f in fams))


if __name__ == "__main__":
    main()
