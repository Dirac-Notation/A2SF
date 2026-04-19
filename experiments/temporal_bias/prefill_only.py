"""
Prefill-only predictors of optimal coefficient.

All decode-dependent features (qd_sim, qd_corr, br) are EXCLUDED.
Tests whether we can predict the optimal coefficient profile using only
quantities available at prefill time.

Includes extended candidates:
  aj_to_last_k: mean Jaccard top-k overlap against the last k window queries
                (k = 1, 2, 4, 8, 16, 32)
  ap_to_last_k: mean Pearson attention-pattern corr against last k
  qq_to_last_k: mean cos Q-Q against last k

Tests affine fit: opt_coeff ≈ a · feature + b.
Also tests 2-variable prefill-only combinations.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

WORKPATH = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(WORKPATH, "plots")


def find_metric_dirs(base=None):
    base = base or PLOT_DIR
    out = []
    for root, _d, files in os.walk(base):
        if "metrics.npz" in files:
            ds = os.path.basename(root)
            tk = os.path.basename(os.path.dirname(root))
            out.append((tk, ds, os.path.join(root, "metrics.npz")))
    return sorted(out)


def affine_r2(x, y):
    """y ≈ a·x + b (clipped to [0, 1])."""
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = np.clip(A @ coef, 0.0, 1.0)
    r2 = 1 - np.var(y - pred) / (np.var(y) + 1e-12)
    return float(coef[0]), float(coef[1]), float(r2)


def ols_r2(X, y):
    """Standardised OLS; returns R² and standardised β."""
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ys = (y - y.mean()) / (y.std() + 1e-12)
    beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    r2 = 1 - np.var(ys - Xs @ beta) / (np.var(ys) + 1e-12)
    return float(r2), beta


def build_to_last_k(M_samples, k):
    """
    M_samples: (n_samples, W, W)  - recent-first matrix
    Returns:  (n_samples, W)  mean over columns 0..k-1  (mean over the k most recent
              reference queries, including self at col 0)
    """
    n, W, _ = M_samples.shape
    k = min(k, W)
    return M_samples[:, :, :k].mean(axis=2)


def main():
    for source_dir in ["plots", "plots/long32k"]:
        base = os.path.join(WORKPATH, source_dir)
        if not os.path.isdir(base):
            continue
        mds_all = find_metric_dirs(base)
        # Keep only entries 3 levels deep inside base ( base / task / dataset / metrics.npz )
        mds = [m for m in mds_all
               if os.path.dirname(os.path.dirname(os.path.dirname(m[2]))) == base]
        if not mds:
            continue
        print(f"\n{'#' * 70}\n  source: {source_dir}  ({len(mds)} datasets)\n{'#' * 70}")

        for task, ds, path in mds:
            m = np.load(path, allow_pickle=True)
            oc = m["oc"]                                   # (n, W)
            n_samples, W = oc.shape
            oc_mean = oc.mean(0)

            # Prefill-only features as (n, W)
            feats_pw = {
                "qnorm":       m["qnorm"],
                "neg_ent":    -m["ent"],
                "sharp":       m["sharp"],
                "qq_recent":   m["qq_recent"],
                "ap_recent":   m["ap_recent"],
                "aj_recent":   m["aj_recent"],
            }

            # Extended: to-last-k aggregates from pairwise matrices
            # aj_M, ap_M, qq_M have shape (n, W, W) recent-first
            for k in [2, 4, 8, 16, 32]:
                feats_pw[f"aj_last{k}"] = build_to_last_k(m["aj_M"], k)
                feats_pw[f"ap_last{k}"] = build_to_last_k(m["ap_M"], k)
                feats_pw[f"qq_last{k}"] = build_to_last_k(m["qq_M"], k)

            print(f"\n  ══ {task} / {ds}  (n={n_samples}, W={W}) ══")

            # 1) Affine fit per feature on the MEAN curve
            print("  Affine fit on mean optimal coefficient (prefill-only features):")
            rows = []
            for name, feat in feats_pw.items():
                feat_mean = feat.mean(0)
                a, b, r2 = affine_r2(feat_mean, oc_mean)
                rows.append((name, r2, a, b))
            rows.sort(key=lambda r: -r[1])
            for name, r2, a, b in rows[:12]:
                print(f"    {name:<14s}  R² = {r2:+.3f}   a={a:+.3f}  b={b:+.3f}")

            # 2) Per-sample Pearson r (mean over samples)
            print("\n  Per-sample Pearson r with optimal coefficient:")
            pr = []
            for name, feat in feats_pw.items():
                rs = []
                for i in range(n_samples):
                    x = feat[i]; y = oc[i]
                    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                        continue
                    rs.append(np.corrcoef(x, y)[0, 1])
                if rs:
                    pr.append((name, float(np.mean(rs)), float(np.std(rs))))
            pr.sort(key=lambda r: -r[1])
            for name, rm, rs in pr[:10]:
                print(f"    {name:<14s}  r = {rm:+.3f} ± {rs:.3f}")

            # 3) Best 2-feature OLS (flattened)
            flats = {k: v.flatten() for k, v in feats_pw.items()}
            y_flat = oc.flatten()
            names = list(flats.keys())
            single = {n_: ols_r2(flats[n_].reshape(-1, 1), y_flat)[0] for n_ in names}
            best_single = max(single, key=single.get)

            best_pair, best_r2 = None, -np.inf
            for a, b in combinations(names, 2):
                X = np.stack([flats[a], flats[b]], axis=1)
                r2, _ = ols_r2(X, y_flat)
                if r2 > best_r2:
                    best_pair, best_r2 = (a, b), r2

            X_full = np.stack([flats[k] for k in names], axis=1)
            full_r2, _ = ols_r2(X_full, y_flat)

            print(f"\n  Regression (flattened, prefill-only):")
            print(f"    best single    {best_single:<14s}  R² = {single[best_single]:.3f}")
            print(f"    best pair      {best_pair[0]:<12s} + {best_pair[1]:<12s}  R² = {best_r2:.3f}")
            print(f"    all {len(names)} features                         R² = {full_r2:.3f}")

            # Save ranking summary
            out_json = os.path.join(os.path.dirname(path), "prefill_only.json")
            summary = {
                "task": task, "dataset": ds, "n_samples": n_samples, "W": W,
                "affine_ranking": [
                    {"name": r[0], "r2": r[1], "a": r[2], "b": r[3]} for r in rows
                ],
                "per_sample_pearson": [
                    {"name": p[0], "r_mean": p[1], "r_std": p[2]} for p in pr
                ],
                "best_single": {"name": best_single, "r2": single[best_single]},
                "best_pair":   {"names": list(best_pair), "r2": best_r2},
                "full_r2":     full_r2,
            }
            with open(out_json, "w") as f:
                json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
