"""PPT-ready single-panel figure for Q4: sigmoid uniquely fits per-prompt w_d*.

Reads experiments/paper_figures/rebuttal/q4_summary.json (already produced by
q4_sigmoid_minimality.py) and renders ONE big, slide-friendly bar chart:

  X axis : 4 weight families
  Y axis : per-prompt mean R²  (averaged over all 40 prompts = 10 prompts × 4 tasks)
  Bars   : per-task dots overlaid for context

Output:
  experiments/paper_figures/rebuttal/q4_ppt_summary.{pdf,png}
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif", "font.size": 18,
    "axes.labelsize": 22, "axes.titlesize": 26,
    "xtick.labelsize": 22, "ytick.labelsize": 18,
    "legend.fontsize": 16, "axes.linewidth": 1.4,
    "figure.dpi": 200,
})

ROOT      = "/home/smp9898/A2SF"
SUMMARY   = f"{ROOT}/experiments/paper_figures/rebuttal/q4_summary.json"
OUT_DIR   = f"{ROOT}/experiments/paper_figures/rebuttal"

FAMILY_ORDER  = ["sigmoid", "step", "exp", "linear"]
FAMILY_LABEL  = {"sigmoid": "Sigmoid",
                 "step":    "Step",
                 "exp":     "Exponential",
                 "linear":  "Linear"}
PARAM_LABEL   = {"sigmoid": "(2 params)",
                 "step":    "(1 param)",
                 "exp":     "(1 param)",
                 "linear":  "(1 param)"}
FAMILY_COLOR  = {"sigmoid": "#1f77b4",
                 "step":    "#d62728",
                 "exp":     "#ff7f0e",
                 "linear":  "#2ca02c"}

TASKS = ["Single-doc QA", "Multi-doc QA", "Summarization", "Few-Shot"]
TASK_MARKERS = {
    "Single-doc QA": "o",
    "Multi-doc QA":  "s",
    "Summarization": "^",
    "Few-Shot":      "D",
}


def main():
    with open(SUMMARY) as f:
        data = json.load(f)

    # Gather all per-prompt R² for each family across all tasks
    all_r2 = {fam: [] for fam in FAMILY_ORDER}
    per_task_mean = {fam: {} for fam in FAMILY_ORDER}
    for task in TASKS:
        d = data[task]
        for fam in FAMILY_ORDER:
            arr = np.array(d["per_prompt"][fam], dtype=float)
            arr = arr[np.isfinite(arr)]
            all_r2[fam].extend(arr.tolist())
            per_task_mean[fam][task] = float(arr.mean()) if len(arr) else float("nan")

    # ── Figure ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 6.5))

    x = np.arange(len(FAMILY_ORDER))
    bar_means = [float(np.mean(all_r2[fam])) for fam in FAMILY_ORDER]
    bar_stds  = [float(np.std(all_r2[fam]))  for fam in FAMILY_ORDER]

    # Use symmetric log Y for huge negative numbers (step / exp / linear)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-1100, 2)

    # Bars
    bars = ax.bar(x, bar_means, width=0.55,
                  color=[FAMILY_COLOR[f] for f in FAMILY_ORDER],
                  edgecolor="black", linewidth=1.2, alpha=0.92,
                  zorder=2)

    # Per-task dots overlaid (jitter)
    rng = np.random.default_rng(0)
    for i, fam in enumerate(FAMILY_ORDER):
        for task in TASKS:
            v = per_task_mean[fam][task]
            ax.scatter(i + rng.uniform(-0.10, 0.10), v,
                       marker=TASK_MARKERS[task],
                       s=120, edgecolor="black", linewidth=1.0,
                       facecolor="white", zorder=5)

    # 0 baseline
    ax.axhline(0, color="black", lw=1.4, ls="--", alpha=0.7, zorder=1)
    ax.text(3.45, 0.7, "$R^2 = 0$\n(matches constant)",
            fontsize=11, ha="right", va="center", color="dimgray")

    # Annotate each bar with mean R² text
    for i, (fam, m) in enumerate(zip(FAMILY_ORDER, bar_means)):
        if m >= 0:
            y_text = m + 0.10
            va = "bottom"; color = FAMILY_COLOR[fam]
        else:
            y_text = m * 1.35
            va = "top"; color = FAMILY_COLOR[fam]
        ax.text(i, y_text, f"$\\overline{{R^2}}$ = {m:.2f}",
                ha="center", va=va, fontsize=18, fontweight="bold",
                color=color)

    # X labels (family name + param count)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{FAMILY_LABEL[f]}\n{PARAM_LABEL[f]}" for f in FAMILY_ORDER],
        fontsize=18,
    )
    ax.set_ylabel("per-prompt mean $R^2$")
    ax.set_title("Sigmoid is the only family that fits the empirical $w_d^*$",
                 pad=12)
    ax.grid(True, axis="y", which="both", alpha=0.25, zorder=0)

    # Task legend (small)
    handles = [plt.Line2D([0], [0], marker=TASK_MARKERS[t], lw=0,
                          markerfacecolor="white", markeredgecolor="black",
                          markersize=11, label=t)
               for t in TASKS]
    ax.legend(handles=handles, title="task (per-task mean)",
              loc="lower right", fontsize=12, title_fontsize=12, framealpha=0.95)

    # Subtitle / footnote
    fig.text(0.5, -0.02,
             "Forward-greedy optimal weights $w_d^*$ (paper Obs 1, Appendix C.2) "
             "refit with each family per prompt.  $N{=}40$ (10 prompts × 4 tasks).",
             ha="center", fontsize=12, style="italic", color="dimgray")

    fig.tight_layout()
    out_pdf = f"{OUT_DIR}/q4_ppt_summary.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out_pdf}")

    # Print summary
    print()
    print(f"{'Family':12s}  {'overall R²':>12}  {'positive prompts':>18}")
    print("─" * 50)
    for fam in FAMILY_ORDER:
        arr = np.array(all_r2[fam])
        pos = float(np.mean(arr > 0))
        print(f"  {FAMILY_LABEL[fam]:10s}  {arr.mean():12.3f}  {100*pos:14.0f}%")


if __name__ == "__main__":
    main()
