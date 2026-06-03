"""Q11 rebuttal: reward signal is informative (NOT sparse), supporting
NeuralUCB's standard regret bound.

We quantify informativeness on the SAME training data the agent learns from:

  (A) Per-prompt reward spread = max(R) - min(R) over 13 actions.
      "How much does picking a better action change the reward?"
  (B) Per-prompt best-vs-mean gap = max(R) - mean(R) over 13 actions.
      "How much regret a random policy would incur per step?"

Larger values → more informative signal → bandit problem is well-posed →
standard NeuralUCB Õ(d̃√T) regret bound applies directly.

Output:
  experiments/paper_figures/rebuttal/q11_reward_informativeness.{pdf,png}
  experiments/paper_figures/rebuttal/q11_summary.json
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif", "font.size": 12,
    "axes.labelsize": 13, "axes.titlesize": 14,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "axes.linewidth": 1.0,
    "figure.dpi": 150,
})

TRAIN   = "/home/smp9898/A2SF/datasets/training/scored/llama3-1b/train.jsonl"
OUT_DIR = "/home/smp9898/A2SF/experiments/paper_figures/rebuttal"

TASKS = ["Code Complete", "Few Shot", "Single-doc QA",
         "Multi-doc QA", "Summarization", "Passage Retrieval"]


def load_train():
    by_task = {}
    with open(TRAIN) as f:
        for line in f:
            r = json.loads(line)
            t = r.get("task_type")
            raw = r.get("action_scores_maxo_by_budget")
            scores = raw.get("128") if isinstance(raw, dict) else raw
            if not scores or len(scores) != 13: continue
            by_task.setdefault(t, []).append(scores)
    return {t: np.array(arr) for t, arr in by_task.items()}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    by_task = load_train()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # (A) Reward spread per task
    ax = axes[0]
    box_a = []
    labels_kept = []
    for t in TASKS:
        if t not in by_task: continue
        arr = by_task[t]
        spread = arr.max(axis=1) - arr.min(axis=1)
        box_a.append(spread)
        labels_kept.append(t)
    bp = ax.boxplot(box_a, positions=range(len(labels_kept)), widths=0.55,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.4))
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue"); patch.set_alpha(0.55)
    ax.set_xticks(range(len(labels_kept)))
    ax.set_xticklabels([l.replace(" ", "\n") for l in labels_kept], fontsize=9.5)
    for i, s in enumerate(box_a):
        ax.text(i, -0.04, f"μ={s.mean():.2f}", ha="center", fontsize=9.2, fontweight="bold")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("reward spread  ($\\max R - \\min R$ over 13 actions)")
    ax.set_title("(A) Action choice has measurable reward impact")
    ax.set_ylim(-0.10, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    # (B) Best-vs-mean gap (per-step regret of a random policy)
    ax = axes[1]
    box_b = []
    for t in labels_kept:
        arr = by_task[t]
        gap = arr.max(axis=1) - arr.mean(axis=1)
        box_b.append(gap)
    bp = ax.boxplot(box_b, positions=range(len(labels_kept)), widths=0.55,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.4))
    for patch in bp["boxes"]:
        patch.set_facecolor("seagreen"); patch.set_alpha(0.55)
    ax.set_xticks(range(len(labels_kept)))
    ax.set_xticklabels([l.replace(" ", "\n") for l in labels_kept], fontsize=9.5)
    for i, g in enumerate(box_b):
        ax.text(i, -0.03, f"μ={g.mean():.2f}", ha="center", fontsize=9.2, fontweight="bold")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("best-vs-mean gap  ($\\max R - \\overline{R}$)")
    ax.set_title("(B) Per-step regret of a random policy")
    ax.set_ylim(-0.08, 0.80)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Reward landscape is informative — sparsity does not threaten "
                 "NeuralUCB's regret bound", y=1.02)
    fig.tight_layout()
    out_pdf = f"{OUT_DIR}/q11_reward_informativeness.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out_pdf}")

    # Summary
    summary = {}
    for i, t in enumerate(labels_kept):
        arr = by_task[t]
        summary[t] = {
            "n":               int(len(arr)),
            "mean_spread":     float(box_a[i].mean()),
            "median_spread":   float(np.median(box_a[i])),
            "mean_best_vs_mean":float(box_b[i].mean()),
            "all_zero_pct":    float((arr.max(axis=1) == 0).mean()),
            "best_2nd_gap":    float((np.sort(arr, axis=1)[:, ::-1][:, 0]
                                      - np.sort(arr, axis=1)[:, ::-1][:, 1]).mean()),
        }
    out_json = f"{OUT_DIR}/q11_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved → {out_json}")
    print()
    print(f"{'Task':22s}  {'n':>5}  {'spread':>7}  {'gap':>6}  {'best-2nd':>8}  {'all-zero%':>9}")
    print("─" * 70)
    for t in labels_kept:
        s = summary[t]
        print(f"  {t:22s}  {s['n']:4d}  {s['mean_spread']:7.3f}  "
              f"{s['mean_best_vs_mean']:6.3f}  {s['best_2nd_gap']:8.3f}  "
              f"{100*s['all_zero_pct']:8.1f}%")


if __name__ == "__main__":
    main()
