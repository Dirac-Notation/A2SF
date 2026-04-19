"""
Deep-dive analysis — what is the MINIMAL set of factors that explains the
optimal coefficient, and how do the candidate predictors relate to each other?

Runs on saved metrics.npz files produced by optimal.py.

Questions:
  Q5. How redundant are the feature families?
       - corr(qq_recent, qd_sim) per position      (H5a ≈ H1?)
       - corr(ap_recent, qd_corr) per position     (H5b ≈ H2?)
       - corr(aj_recent, br) per position          (H5c ≈ block-hit?)
  Q6. Two-variable OLS: best pair, ΔR² vs full
  Q7. Where is the "knee" — position at which optimal coefficient first hits 0?
       Correlate knee position with task properties.
  Q8. Per-layer optimal coefficient: is the bias uniform across layers, or
       carried mainly by the late layers?
  Q9. Power-law vs sigmoid: which fits the MEAN-per-dataset decay best, and what
       are the implied exponents?
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import curve_fit

WORKPATH = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(WORKPATH, "plots")


def find_metric_dirs():
    out = []
    for root, _d, files in os.walk(PLOT_DIR):
        if "metrics.npz" in files:
            ds_name = os.path.basename(root)
            task_name = os.path.basename(os.path.dirname(root))
            out.append((task_name, ds_name, os.path.join(root, "metrics.npz")))
    return sorted(out)


def r2_from(y, pred):
    return 1 - np.var(y - pred) / (np.var(y) + 1e-12)


def ols_r2(X, y):
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ys = (y - y.mean()) / (y.std() + 1e-12)
    beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    return r2_from(ys, Xs @ beta), beta


def main():
    mds = find_metric_dirs()
    out_rows = []

    for task, ds, path in mds:
        print(f"\n{'=' * 70}\n  {task} / {ds}\n{'=' * 70}")
        m = np.load(path, allow_pickle=True)

        feats = {
            "qd_sim":    m["qd_sim"],
            "qd_corr":   m["qd_corr"],
            "qnorm":     m["qnorm"],
            "neg_ent":   -m["ent"],
            "sharp":     m["sharp"],
            "qq_recent": m["qq_recent"],
            "ap_recent": m["ap_recent"],
            "aj_recent": m["aj_recent"],
            "br":        m["br"],
        }
        oc = m["oc"]
        n, W = oc.shape
        y_flat = oc.flatten()

        # ── Q5: redundancy between H5 family and other features ──
        print("\n  [Q5] within-sample correlation of feature pairs (mean across samples):")
        pairs = [("qq_recent", "qd_sim"),
                 ("ap_recent", "qd_corr"),
                 ("aj_recent", "br")]
        for a, b in pairs:
            rs = []
            for i in range(n):
                x = feats[a][i]; z = feats[b][i]
                if np.std(x) < 1e-9 or np.std(z) < 1e-9:
                    continue
                rs.append(np.corrcoef(x, z)[0, 1])
            print(f"    corr({a:<10s}, {b:<10s}) = {np.mean(rs):+.3f} ± {np.std(rs):.3f}")

        # ── Q6: best 2-variable regression ──
        feat_names = list(feats.keys())
        flats = {k: v.flatten() for k, v in feats.items()}
        best_pair, best_r2, best_beta = None, -np.inf, None
        single_r2 = {}
        for name in feat_names:
            x = flats[name]
            r2, _ = ols_r2(x.reshape(-1, 1), y_flat)
            single_r2[name] = r2
        for a, b in combinations(feat_names, 2):
            X = np.stack([flats[a], flats[b]], axis=1)
            r2, beta = ols_r2(X, y_flat)
            if r2 > best_r2:
                best_r2, best_pair, best_beta = r2, (a, b), beta
        X_full = np.stack([flats[k] for k in feat_names], axis=1)
        full_r2, full_beta = ols_r2(X_full, y_flat)
        print(f"\n  [Q6] regression R²:")
        print(f"    best single        {max(single_r2, key=single_r2.get):<12s}  R² = {max(single_r2.values()):.3f}")
        print(f"    best pair          {best_pair[0]:<6s} + {best_pair[1]:<6s}  R² = {best_r2:.3f}  "
              f"β=({best_beta[0]:+.2f}, {best_beta[1]:+.2f})")
        print(f"    full (9 features)                          R² = {full_r2:.3f}")
        print(f"    Δ(full − best pair) = {full_r2 - best_r2:+.3f}")

        # ── Q7: knee position (first position where optimal coeff <= 0.05) ──
        knee_pos = []
        for i in range(n):
            below = np.where(oc[i] <= 0.05)[0]
            knee_pos.append(below[0] if len(below) > 0 else W)
        knee_pos = np.array(knee_pos)
        print(f"\n  [Q7] knee position (first idx where coeff ≤ 0.05): "
              f"mean = {knee_pos.mean():.1f}, std = {knee_pos.std():.1f}, range=[{knee_pos.min()}, {knee_pos.max()}]")

        # ── Q8: per-layer optimal coefficient — layer-level decay ──
        pl_coef = m["pl_coef"]       # (n_samples, L, W)
        pl_hit = m["pl_hit"]         # (n_samples, L, W)
        mean_coef_per_layer = pl_coef.mean(axis=(0, 2))  # mean over samples & positions → (L,)
        print(f"\n  [Q8] mean optimal coefficient by layer (over positions):")
        L = mean_coef_per_layer.shape[0]
        # Show bar-chart-style numbers
        for l in range(L):
            bar = "█" * int(mean_coef_per_layer[l] * 40)
            print(f"    L{l:02d}  {mean_coef_per_layer[l]:.3f}  {bar}")

        # Which layer(s) are most "temporally biased" — i.e., coef drops fastest?
        # Measure: slope of coefficient vs position (negative = biased)
        slopes = np.zeros(L)
        for l in range(L):
            y = pl_coef.mean(axis=0)[l]   # (W,)
            x = np.arange(W)
            slope = np.polyfit(x, y, 1)[0]
            slopes[l] = slope
        print(f"\n  [Q8'] per-layer slope (coef vs position, more negative = stronger temporal bias):")
        for l in range(L):
            print(f"    L{l:02d}  slope = {slopes[l]:+.4f}")

        # ── Q9: power-law vs sigmoid on MEAN optimal coeff ──
        mean_oc = oc.mean(0)
        x_pos = np.arange(W).astype(float)
        def fn_sig(x, a, b): return 1.0 / (1.0 + np.exp(a * (x - b)))
        def fn_pwr(x, a):    return 1.0 / (1.0 + a * x)
        def fn_exp(x, a):    return np.exp(-a * x)
        def fn_log(x, a):    return np.maximum(0, 1.0 - a * np.log1p(x))

        fits = {}
        for label, fn, p0, bnd in [
            ("sigmoid",    fn_sig, [0.2, 3.0], ((0, 0), (np.inf, np.inf))),
            ("power-law",  fn_pwr, [0.5],      ((0,),    (np.inf,))),
            ("exponential",fn_exp, [0.2],      ((0,),    (np.inf,))),
            ("log-decay",  fn_log, [0.3],      ((0,),    (np.inf,))),
        ]:
            try:
                popt, _ = curve_fit(fn, x_pos, mean_oc, p0=p0, bounds=bnd, maxfev=5000)
                pred = fn(x_pos, *popt)
                r2 = r2_from(mean_oc, pred)
            except Exception:
                popt, r2 = None, np.nan
            fits[label] = (popt, r2)
        print(f"\n  [Q9] decay-function fits on mean optimal coefficient:")
        for label, (popt, r2) in sorted(fits.items(), key=lambda kv: -kv[1][1] if not np.isnan(kv[1][1]) else 0):
            p_str = ", ".join(f"{v:.3f}" for v in popt) if popt is not None else "-"
            print(f"    {label:<12s}  R² = {r2:+.3f}  params = ({p_str})")

        out_rows.append({
            "task": task, "dataset": ds, "n": n, "W": W,
            "best_single": (max(single_r2, key=single_r2.get), max(single_r2.values())),
            "best_pair":   (best_pair, best_r2),
            "full_r2":     full_r2,
            "knee_mean":   float(knee_pos.mean()),
            "knee_std":    float(knee_pos.std()),
            "mean_coef_per_layer": mean_coef_per_layer.tolist(),
            "slopes_per_layer": slopes.tolist(),
            "fits": {k: (list(v[0]) if v[0] is not None else None, float(v[1]))
                     for k, v in fits.items()},
        })

    # ── Cross-dataset aggregate plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    labels = [f"{r['task']}\n{r['dataset']}" for r in out_rows]
    pair_r2s = [r["best_pair"][1] for r in out_rows]
    full_r2s = [r["full_r2"] for r in out_rows]
    single_r2s = [r["best_single"][1] for r in out_rows]
    x = np.arange(len(labels))
    ax.bar(x - 0.27, single_r2s, width=0.25, label="best single", color="steelblue")
    ax.bar(x,         pair_r2s,   width=0.25, label="best pair",   color="coral")
    ax.bar(x + 0.27,  full_r2s,   width=0.25, label="full 9-feat", color="forestgreen")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0, fontsize=10)
    ax.set_ylabel("R² of predicting optimal coefficient")
    ax.set_title("How much can we explain with few features?")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0, 1]
    knee_means = [r["knee_mean"] for r in out_rows]
    knee_stds  = [r["knee_std"]  for r in out_rows]
    ax.bar(x, knee_means, yerr=knee_stds, color="purple", alpha=0.7, edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0, fontsize=10)
    ax.set_ylabel("knee position (idx where coef ≤ 0.05)")
    ax.set_title("Where does the optimal coefficient collapse?")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    for r, color in zip(out_rows, plt.cm.Set2(np.linspace(0, 1, len(out_rows)))):
        L = len(r["mean_coef_per_layer"])
        ax.plot(np.arange(L), r["mean_coef_per_layer"],
                marker="o", linewidth=2, color=color,
                label=f"{r['task']}/{r['dataset']}")
    ax.set_xlabel("Layer index"); ax.set_ylabel("Mean optimal coefficient (over window positions)")
    ax.set_title("Q8: per-layer mean optimal coefficient")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    fit_names = ["sigmoid", "power-law", "exponential", "log-decay"]
    fit_colors = ["steelblue", "coral", "forestgreen", "purple"]
    xs = np.arange(len(out_rows))
    width = 0.2
    for fi, fn in enumerate(fit_names):
        vals = [r["fits"][fn][1] for r in out_rows]
        ax.bar(xs + (fi - 1.5) * width, vals, width=width, label=fn, color=fit_colors[fi])
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=0, fontsize=10)
    ax.set_ylabel("R² of decay function on mean optimal coeff")
    ax.set_title("Q9: which decay family fits?")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Deep analysis: what determines the optimal coefficient?",
                 fontsize=18, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "H_deep_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  saved {out_path}")

    with open(os.path.join(PLOT_DIR, "deep_analysis.json"), "w") as f:
        json.dump(out_rows, f, indent=2)


if __name__ == "__main__":
    main()
