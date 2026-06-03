"""
Practical synthesis — can the optimal coefficient PROFILE be predicted from
a *prefill-only* feature (no decode needed)?

If yes, we have a recipe for A2SF that doesn't require any heuristic decay
(sigmoid/exponential) — just compute one feature on the prefill window.

Tests three single-feature predictors that use ONLY information available
during prefill:

  ap_recent[i] = corr(attn[i], attn[most_recent])
  aj_recent[i] = jaccard(topk(attn[i]), topk(attn[most_recent]))
  qq_recent[i] = cos(Q[i], Q[most_recent])

For each, fits a 2-parameter affine map (a*x + b) and reports R² to the
optimal coefficient profile, then plots predicted vs actual curves.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

WORKPATH = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(WORKPATH, "plots")


def find_metric_dirs():
    out = []
    for root, _d, files in os.walk(PLOT_DIR):
        if "metrics.npz" in files:
            ds = os.path.basename(root)
            tk = os.path.basename(os.path.dirname(root))
            out.append((tk, ds, os.path.join(root, "metrics.npz")))
    return sorted(out)


def affine_fit(x, y):
    """y ≈ a*x + b. Clip to [0,1]."""
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = np.clip(A @ coef, 0.0, 1.0)
    r2 = 1 - np.var(y - pred) / (np.var(y) + 1e-12)
    return coef, pred, r2


def main():
    mds = find_metric_dirs()
    summary = []

    fig, axes = plt.subplots(len(mds), 3, figsize=(18, 4 * len(mds)),
                             squeeze=False)

    for ridx, (task, ds, path) in enumerate(mds):
        m = np.load(path, allow_pickle=True)
        oc = m["oc"].mean(0)              # (W,)
        W = oc.shape[0]
        x_pos = np.arange(W)

        candidates = {
            "ap_recent": m["ap_recent"].mean(0),
            "aj_recent": m["aj_recent"].mean(0),
            "qq_recent": m["qq_recent"].mean(0),
        }

        print(f"\n  {task} / {ds}")
        for ci, (name, feat) in enumerate(candidates.items()):
            coef, pred, r2 = affine_fit(feat, oc)
            print(f"    {name:<10s}  affine fit R² = {r2:+.3f}   coef=({coef[0]:+.3f}, {coef[1]:+.3f})")

            ax = axes[ridx, ci]
            ax.plot(x_pos, oc, "o-", color="black", label="optimal coeff (mean)",
                    linewidth=2, markersize=4, alpha=0.85)
            ax.plot(x_pos, pred, "s--", color="coral",
                    label=f"affine({name})  R²={r2:.2f}", linewidth=2,
                    markersize=4, alpha=0.85)
            ax.plot(x_pos, feat, ":", color="steelblue", alpha=0.6,
                    label=f"raw {name}", linewidth=1.5)
            ax.set(xlabel="Query distance from end",
                   ylabel="value (clipped to [0,1])",
                   title=f"{ds}: predict optimal coeff from {name}",
                   ylim=(-0.05, 1.05))
            ax.legend(fontsize=10, loc="upper right")
            ax.grid(True, alpha=0.3)

            summary.append({"task": task, "dataset": ds, "feature": name,
                            "r2": float(r2),
                            "coef": [float(c) for c in coef]})

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "I_predict_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  saved {out_path}")

    with open(os.path.join(PLOT_DIR, "predict_curve.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
