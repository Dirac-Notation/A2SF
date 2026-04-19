"""
Deep secondary analysis on saved metrics.npz files.

Loads each dataset's metrics and asks:

  Q1. Which single feature best explains the optimal-coefficient sequence?
  Q2. Which COMBINATION of features (multivariate linear) explains the most variance?
  Q3. Per-position regimes — does the dominant explainer change with distance?
  Q4. How well does each candidate decay function fit the optimal coefficient?
        (sigmoid vs exponential vs power-law)
  Q5. Sigmoid parameters across datasets — is the transition point (b) consistent?
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

WORKPATH = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(WORKPATH, "plots")


# ── Decay function candidates ──
def fn_sigmoid(x, a, b):    return 1.0 / (1.0 + np.exp(a * (x - b)))
def fn_exp(x, a):           return np.exp(-a * x)
def fn_power(x, a):         return 1.0 / (1.0 + a * x)
def fn_step(x, b):          return (x < b).astype(float)


def fit_safe(fn, x, y, p0, bounds):
    try:
        popt, _ = curve_fit(fn, x, y, p0=p0, bounds=bounds, maxfev=10000)
        pred = fn(x, *popt)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
        r2 = 1 - ss_res / ss_tot
        return popt, r2
    except Exception:
        return None, np.nan


def find_metric_dirs():
    out = []
    for root, dirs, files in os.walk(PLOT_DIR):
        if "metrics.npz" in files:
            ds_name = os.path.basename(root)
            task_name = os.path.basename(os.path.dirname(root))
            out.append((task_name, ds_name, os.path.join(root, "metrics.npz")))
    return sorted(out)


def per_position_dominant(agg_arrays, oc):
    """For each window position, which feature has the highest |Pearson r| against oc[pos]
       across samples?"""
    n, W = oc.shape
    feature_names = list(agg_arrays.keys())
    rs = np.zeros((len(feature_names), W))
    for fi, name in enumerate(feature_names):
        feat = agg_arrays[name]            # (n, W)
        for p in range(W):
            x = feat[:, p]; y = oc[:, p]
            if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                rs[fi, p] = np.nan
            else:
                rs[fi, p] = pearsonr(x, y)[0]
    return feature_names, rs


def main():
    metric_dirs = find_metric_dirs()
    if not metric_dirs:
        print("No metrics.npz found yet. Run optimal.py first.")
        return

    print(f"Found {len(metric_dirs)} dataset metric files:")
    for t, d, p in metric_dirs:
        print(f"  {t} / {d}  →  {p}")

    summary_rows = []

    for task, ds, path in metric_dirs:
        print(f"\n{'='*70}\n  {task} / {ds}\n{'='*70}")
        m = np.load(path, allow_pickle=True)
        oc = m["oc"]                       # (n, W)
        n, W = oc.shape
        x_pos = np.arange(W).astype(float)
        oc_mean = oc.mean(0)

        # ── Q4: decay-function fits to mean optimal coefficient ──
        print("\n  [Q4] decay-function fits (R² on mean optimal coeff):")
        results_fit = {}
        for label, fn, p0, bnd in [
            ("sigmoid",  fn_sigmoid, [0.5, 16.0], ((0, 0), (np.inf, np.inf))),
            ("exponential", fn_exp,  [0.05],      ((0,),      (np.inf,))),
            ("power-law",   fn_power,[0.05],      ((0,),      (np.inf,))),
            ("step",        fn_step, [16.0],      ((0,),      (np.inf,))),
        ]:
            popt, r2 = fit_safe(fn, x_pos, oc_mean, p0, bnd)
            results_fit[label] = (popt, r2)
            popt_str = ("popt=" + ", ".join(f"{v:.3f}" for v in popt)) if popt is not None else "fit failed"
            print(f"    {label:<14s}  R² = {r2:+.3f}   {popt_str}")

        # ── Q1, Q2 prep: assemble feature dict ──
        feats = {
            "qd_sim":     m["qd_sim"],
            "qd_corr":    m["qd_corr"],
            "qnorm":      m["qnorm"],
            "neg_ent":    -m["ent"],
            "sharp":      m["sharp"],
            "qq_recent":  m["qq_recent"],
            "ap_recent":  m["ap_recent"],
            "aj_recent":  m["aj_recent"],
            "br":         m["br"],
        }

        # ── Q1: per-feature Pearson with oc, averaged per-sample ──
        print("\n  [Q1] per-feature Pearson r with optimal coeff (mean ± std over samples):")
        rows = []
        for name, feat in feats.items():
            rs = []
            for i in range(n):
                x = feat[i]; y = oc[i]
                if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                    continue
                rs.append(np.corrcoef(x, y)[0, 1])
            rows.append((name, float(np.mean(rs)) if rs else np.nan,
                         float(np.std(rs)) if rs else np.nan))
        rows.sort(key=lambda r: -abs(r[1]) if not np.isnan(r[1]) else 0)
        for name, mn, sd in rows:
            print(f"    {name:<14s}  r = {mn:+.3f} ± {sd:.3f}")

        # ── Q2: multivariate OLS on flattened (per-position) data ──
        # Feature matrix X (n*W, F), target y (n*W,)
        X = np.stack([feats[k].flatten() for k in feats], axis=1).astype(np.float64)
        y = oc.flatten().astype(np.float64)
        # Standardise
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
        ys = (y - y.mean()) / (y.std() + 1e-12)
        # OLS: beta = (X^T X)^-1 X^T y
        try:
            beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
            pred = Xs @ beta
            r2_full = 1 - np.var(ys - pred) / (np.var(ys) + 1e-12)
        except Exception:
            beta = np.zeros(Xs.shape[1]); r2_full = np.nan
        print(f"\n  [Q2] multivariate OLS (all features, standardised):")
        print(f"    R² = {r2_full:.3f}")
        for name, b in zip(feats.keys(), beta):
            print(f"    β[{name:<14s}] = {b:+.3f}")

        # ── Single-feature R² benchmark (for comparison with multivariate) ──
        single_r2 = {}
        for name, feat in feats.items():
            xs = feat.flatten().astype(np.float64)
            xs = (xs - xs.mean()) / (xs.std() + 1e-12)
            beta1, *_ = np.linalg.lstsq(xs[:, None], ys, rcond=None)
            pred1 = xs * beta1[0]
            single_r2[name] = 1 - np.var(ys - pred1) / (np.var(ys) + 1e-12)
        print(f"\n  [Q2'] single-feature R² (linear, flattened):")
        for name, r2 in sorted(single_r2.items(), key=lambda x: -x[1]):
            print(f"    {name:<14s}  R² = {r2:+.3f}")

        # ── Q3: per-position dominant feature ──
        feat_names, rs_per_pos = per_position_dominant(feats, oc)
        # find which feature has max |r| at each position (skip all-NaN positions)
        abs_rs = np.abs(rs_per_pos)
        dominant_idx = np.full(W, -1, dtype=int)
        dominant_r = np.full(W, np.nan)
        for p in range(W):
            col = abs_rs[:, p]
            if np.all(np.isnan(col)):
                continue
            dominant_idx[p] = int(np.nanargmax(col))
            dominant_r[p] = rs_per_pos[dominant_idx[p], p]

        # ── Plot Q3 ──
        fig, ax = plt.subplots(figsize=(14, 6))
        cmap = plt.cm.tab10
        for fi, name in enumerate(feat_names):
            ax.plot(np.arange(W), rs_per_pos[fi], alpha=0.6, color=cmap(fi % 10),
                    linewidth=1.5, label=name)
        ax.axhline(0, color="k", lw=0.8)
        ax.set(xlabel="Query position (0=most recent)",
               ylabel="Pearson r vs optimal coefficient (across samples)",
               title=f"Q3: per-position correlation between feature & optimal coeff",
               ylim=(-1, 1))
        ax.legend(loc="upper right", ncol=3, fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.suptitle(f"{task} / {ds}", fontsize=16, fontweight="bold")
        plt.tight_layout()
        save_dir = os.path.dirname(path)
        out_path = os.path.join(save_dir, "G_per_position_correlation.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved {out_path}")

        # store for cross-dataset summary
        summary_rows.append({
            "task": task, "dataset": ds, "n": n, "W": W,
            "fits": {k: (v[0].tolist() if v[0] is not None else None, float(v[1]))
                     for k, v in results_fit.items()},
            "per_feature_pearson": rows,
            "multivar_r2": float(r2_full),
            "multivar_beta": dict(zip(list(feats.keys()), beta.tolist())),
            "single_r2": {k: float(v) for k, v in single_r2.items()},
        })

    # ── Cross-dataset summary table ──
    print("\n" + "="*70)
    print("  Cross-dataset summary")
    print("="*70)

    # Sigmoid params across datasets
    print("\n  Sigmoid fit (slope=a, midpoint=b):")
    for r in summary_rows:
        s = r["fits"].get("sigmoid")
        if s and s[0]:
            print(f"    {r['task']:<18s} {r['dataset']:<14s}  a={s[0][0]:.3f}  b={s[0][1]:.2f}  R²={s[1]:.3f}")

    # Top single feature per dataset
    print("\n  Top single feature per dataset (by |r|):")
    for r in summary_rows:
        top = max(r["per_feature_pearson"], key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0)
        print(f"    {r['task']:<18s} {r['dataset']:<14s}  → {top[0]:<14s}  r={top[1]:+.3f}")

    # Save summary as JSON
    out_json = os.path.join(PLOT_DIR, "analysis_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\n  full summary saved → {out_json}")


if __name__ == "__main__":
    main()
