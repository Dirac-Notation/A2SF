"""Figure 5: NIAH heatmaps (budget 128), score per (context length, needle depth) cell.
Soft score: for the best-matching number in the output, digits matched in order / max(length).

    python experiments/niah/fig_niah.py --data datasets/niah_single.jsonl --runs runs/niah_llama3-8b --out fig_niah.pdf
"""
import argparse, glob, json, re
from difflib import SequenceMatcher
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams["pdf.fonttype"] = 42; plt.rcParams["ps.fonttype"] = 42
PANELS = [("full", "Full cache"), ("tova", "TOVA"), ("h2o", "H2O"), ("snapkv", "SnapKV"), ("snapkv_pool", "SnapKV\n+ pooling"),
          ("qtrust", "Q-Trust (ours)")]

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--out", default="fig_niah.pdf")
ap.add_argument("--budget", type=int, default=128)
a = ap.parse_args()
ANS = {json.loads(l)["id"]: json.loads(l)["answer"] for l in open(a.data)}


def soft(r):
    ans = ANS[r["id"]]
    nums = re.findall(r"\d+", r["pred"])
    return max((sum(b.size for b in SequenceMatcher(None, n, ans).get_matching_blocks()) / max(len(ans), len(n)) for n in nums), default=0.0)


load = lambda k: [json.loads(l) for f in glob.glob(f"{a.runs}/{k}_{a.budget}_shard*.jsonl") for l in open(f)]
full = load("full")
lens = sorted({r["length"] for r in full}); deps = sorted({r["depth"] for r in full})
cmap = LinearSegmentedColormap.from_list("niah", ["#c92a2a", "#f59f00", "#fcc419", "#2f9e44"])
fig, axes = plt.subplots(1, len(PANELS), figsize=(7.0, 2.0), dpi=130, sharey=True)
for ax, (k, name) in zip(axes, PANELS):
    R = load(k)
    M = np.array([[100 * np.mean([soft(r) for r in R if r["length"] == L and r["depth"] == d]) for L in lens] for d in deps])
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(lens))); ax.set_xticklabels([f"{L // 1024}K" for L in lens], fontsize=7)
    ax.set_yticks(range(0, len(deps), 2)); ax.set_yticklabels([f"{int(round(100 * d))}" for d in deps][::2], fontsize=7.5)
    ax.set_title(f"{name}\n{np.mean([soft(r) for r in R]) * 100:.1f}", fontsize=7.5, fontweight="bold" if k == "qtrust" else "normal")
    ax.tick_params(length=2)
    print(f"{k:12s} soft {100 * np.mean([soft(r) for r in R]):.1f}  exact {100 * np.mean([r['correct'] for r in R]):.1f}  (n={len(R)})")
axes[0].set_ylabel("needle depth (%)", fontsize=8.5)
axes[2].set_xlabel("context length", fontsize=8.5)
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015); cb.ax.tick_params(labelsize=7.5); cb.set_label("score (%)", fontsize=8)
fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
