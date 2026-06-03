"""Compute R² of 4 weight-family fits to w_tan (Tanimoto-optimal weight).

For each task:
  (1) Task-mean R²: fit each family to mean(w_tan) across prompts.
  (2) Per-prompt R² (mean): fit each family to each prompt's w_tan, average R² across prompts.

Families (all in distance-units d = chunk_idx * chunk + (chunk-1)/2):
  sigmoid : w(d) = 1 / (1 + exp(a * (d - b)))                    [2 params]
  exp     : w(d) = exp(-d / tau)                                  [1 param]
  step    : w(d) = 1 if d < d0 else 0                             [1 param, integer-ish]
  linear  : w(d) = max(0, 1 - d / d0)                             [1 param]

All fits bounded so a/tau/d0 ≥ small positive.
"""
import os, sys, json
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar


# Tanimoto-optimal weights come from the obs1 pipeline
# (observations/data/<Task>__<dataset>.npz, key "w_tan"). Budget-independent.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TASKS = [
    ('Single-doc_QA/qasper',     'Single-doc QA'),
    ('Multi-doc_QA/hotpotqa',    'Multi-doc QA'),
    ('Summarization/gov_report', 'Summarization'),
    ('Few_Shot/samsum',          'Few-Shot'),
]


def sigmoid_fn(d, a, b):
    return 1.0 / (1.0 + np.exp(np.clip(a * (d - b), -30.0, 30.0)))


def exp_fn(d, tau):
    return np.exp(-d / max(tau, 1e-6))


def linear_fn(d, d0):
    return np.clip(1.0 - d / max(d0, 1e-6), 0.0, 1.0)


def step_fn(d, d0):
    return (d < d0).astype(np.float32)


def r_squared(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot < 1e-12:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def fit_sigmoid(d, y, W):
    try:
        popt, _ = curve_fit(sigmoid_fn, d, y,
                             p0=[0.05, float(W) / 4.0],
                             bounds=([0.0, 0.0], [5.0, float(W)]),
                             maxfev=5000)
        return r_squared(y, sigmoid_fn(d, *popt)), popt
    except Exception:
        return float('nan'), None


def fit_exp(d, y, W):
    try:
        popt, _ = curve_fit(exp_fn, d, y,
                             p0=[float(W) / 4.0],
                             bounds=([1e-3], [10.0 * float(W)]),
                             maxfev=5000)
        return r_squared(y, exp_fn(d, *popt)), popt
    except Exception:
        return float('nan'), None


def fit_linear(d, y, W):
    try:
        popt, _ = curve_fit(linear_fn, d, y,
                             p0=[float(W) / 4.0],
                             bounds=([1e-3], [10.0 * float(W)]),
                             maxfev=5000)
        return r_squared(y, linear_fn(d, *popt)), popt
    except Exception:
        return float('nan'), None


def fit_step(d, y, W):
    """Discrete d0 search over chunk grid (faster + more robust than curve_fit on discontinuous step)."""
    best_r2 = -float('inf'); best_d0 = None
    candidates = np.unique(np.concatenate([d, [d.max() + 1.0]]))
    for d0 in candidates:
        r2 = r_squared(y, step_fn(d, d0))
        if r2 > best_r2:
            best_r2 = r2; best_d0 = float(d0)
    return best_r2, [best_d0]


_K_PARAMS = {'sigmoid': 2, 'exp': 1, 'step': 1, 'linear': 1}


def _adj_r2(r2, n, k):
    """Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - k - 1)."""
    if n <= k + 1: return float('nan')
    return 1.0 - (1.0 - r2) * (n - 1.0) / (n - k - 1.0)


def _aic_bic(sse, n, k):
    """Gaussian-likelihood AIC/BIC. Lower = better."""
    if n <= 0 or sse <= 0: return float('nan'), float('nan')
    sigma2 = sse / n
    log_lik = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(n) - 2 * log_lik
    return float(aic), float(bic)


def _fit_with_sse(fit_fn, d, y, W):
    r2, popt = fit_fn(d, y, W)
    if popt is None:
        return float('nan'), float('nan'), None
    if fit_fn is fit_sigmoid:
        y_pred = sigmoid_fn(d, *popt)
    elif fit_fn is fit_exp:
        y_pred = exp_fn(d, *popt)
    elif fit_fn is fit_linear:
        y_pred = linear_fn(d, *popt)
    elif fit_fn is fit_step:
        y_pred = step_fn(d, *popt)
    sse = float(np.sum((y - y_pred) ** 2))
    return r2, sse, popt


def analyze_task(label, path):
    cd = np.load(os.path.join(DATA, path.replace('/', '__') + '.npz'))
    w_tan = cd['w_tan']                          # (N, G) — Tanimoto-optimal weight
    N, G = w_tan.shape
    chunk = int(cd['chunk']); W = int(cd['window'])
    d = np.arange(G).astype(float) * chunk + (chunk - 1) / 2.0

    fits = {'sigmoid': fit_sigmoid, 'exp': fit_exp, 'step': fit_step, 'linear': fit_linear}

    # (1) Task-mean fit
    y_mean = w_tan.mean(axis=0)
    r2_mean = {}; sse_mean = {}
    for fam, fn in fits.items():
        r2, sse, _ = _fit_with_sse(fn, d, y_mean, W)
        r2_mean[fam] = r2; sse_mean[fam] = sse

    # (2) Per-prompt fits + per-prompt SSE/SST 누적
    per_prompt_r2 = {fam: [] for fam in fits}
    sse_total = {fam: 0.0 for fam in fits}     # for pooled R²
    sst_total = 0.0
    n_points = 0
    for i in range(N):
        y = w_tan[i]
        if y.var() < 1e-12:
            continue
        sst_i = float(np.sum((y - y.mean()) ** 2))
        sst_total += sst_i
        n_points += len(y)
        for fam, fn in fits.items():
            r2, sse, _ = _fit_with_sse(fn, d, y, W)
            per_prompt_r2[fam].append(r2)
            sse_total[fam] += sse

    # Pooled R² (전체 prompt 의 SSE 합 / SST 합)
    pooled_r2 = {fam: float(1.0 - sse_total[fam] / sst_total) if sst_total > 0 else float('nan')
                  for fam in fits}

    # Per-prompt mean/median + adjusted R²
    r2_per_prompt_mean   = {fam: float(np.nanmean(v))   if v else float('nan') for fam, v in per_prompt_r2.items()}
    r2_per_prompt_median = {fam: float(np.nanmedian(v)) if v else float('nan') for fam, v in per_prompt_r2.items()}
    r2_per_prompt_std    = {fam: float(np.nanstd(v))    if v else float('nan') for fam, v in per_prompt_r2.items()}
    r2_per_prompt_iqr    = {fam: float(np.nanpercentile(v, 75) - np.nanpercentile(v, 25)) if v else float('nan')
                              for fam, v in per_prompt_r2.items()}

    # Adjusted R² on task-mean fit (n = G, k = params)
    adj_r2_mean = {fam: _adj_r2(r2_mean[fam], G, _K_PARAMS[fam]) for fam in fits}
    # Adjusted Pooled R² (n = n_points)
    adj_r2_pooled = {fam: _adj_r2(pooled_r2[fam], n_points, _K_PARAMS[fam]) for fam in fits}

    # AIC / BIC on pooled SSE (lower = better)
    aic_bic = {fam: _aic_bic(sse_total[fam], n_points, _K_PARAMS[fam]) for fam in fits}
    aic = {fam: aic_bic[fam][0] for fam in fits}
    bic = {fam: aic_bic[fam][1] for fam in fits}

    return {
        'task': label,
        'N': int(N), 'G': int(G), 'chunk': chunk, 'W': W, 'n_points': int(n_points),
        'r2_task_mean':         r2_mean,
        'adj_r2_task_mean':     adj_r2_mean,
        'pooled_r2':            pooled_r2,
        'adj_pooled_r2':        adj_r2_pooled,
        'r2_per_prompt_mean':   r2_per_prompt_mean,
        'r2_per_prompt_median': r2_per_prompt_median,
        'r2_per_prompt_std':    r2_per_prompt_std,
        'r2_per_prompt_iqr':    r2_per_prompt_iqr,
        'aic': aic, 'bic': bic,
    }


def main():
    rows = []
    for path, label in TASKS:
        rows.append(analyze_task(label, path))

    families = ['sigmoid', 'exp', 'step', 'linear']

    def fmt_row(label, vals):
        return f"  {label:<18}" + ''.join(f"{vals[f]:>10.3f}" for f in families)

    print()
    print(f"=== R² on Tanimoto-optimal w_tan (4 weight families, k_params = sigmoid:2 / exp:1 / step:1 / linear:1) ===")
    print()
    print(f"{'':<18}" + ''.join(f"{f:>10}" for f in families))
    print('-' * 80)

    print('(A) Task-mean curve R² (n=G chunks):')
    for r in rows:
        print(fmt_row(r['task'], r['r2_task_mean']))

    print()
    print('(B) Task-mean ADJUSTED R² (parameter-count 보정):')
    for r in rows:
        print(fmt_row(r['task'], r['adj_r2_task_mean']))

    print()
    print('(C) POOLED R² (모든 prompt 의 SSE/SST 합으로 계산 — 통계적 표준):')
    for r in rows:
        print(fmt_row(r['task'], r['pooled_r2']))

    print()
    print('(D) Adjusted POOLED R²:')
    for r in rows:
        print(fmt_row(r['task'], r['adj_pooled_r2']))

    print()
    print('(E) Per-prompt R²  median (IQR):')
    for r in rows:
        m = r['r2_per_prompt_median']; q = r['r2_per_prompt_iqr']
        cells = [f"{m[f]:+0.3f}({q[f]:.2f})" for f in families]
        print(f"  {r['task']:<18}" + ''.join(f"{c:>14}" for c in cells))

    print()
    print('(F) Per-prompt R²  mean ± std (참고용):')
    for r in rows:
        m = r['r2_per_prompt_mean']; s = r['r2_per_prompt_std']
        cells = [f"{m[f]:+0.3f}±{s[f]:.2f}" for f in families]
        print(f"  {r['task']:<18}" + ''.join(f"{c:>14}" for c in cells))

    print()
    print('(G) AIC (lower = better):')
    for r in rows:
        print(fmt_row(r['task'], r['aic']))

    print()
    print('(H) BIC (lower = better):')
    for r in rows:
        print(fmt_row(r['task'], r['bic']))

    print()
    print("=== Summary across all 4 tasks ===")
    print(f"{'':<18}" + ''.join(f"{f:>10}" for f in families))
    print('-' * 80)
    avg_task_mean   = {f: float(np.nanmean([r['r2_task_mean'][f]         for r in rows])) for f in families}
    avg_adj_mean    = {f: float(np.nanmean([r['adj_r2_task_mean'][f]     for r in rows])) for f in families}
    avg_pooled      = {f: float(np.nanmean([r['pooled_r2'][f]            for r in rows])) for f in families}
    avg_adj_pooled  = {f: float(np.nanmean([r['adj_pooled_r2'][f]        for r in rows])) for f in families}
    avg_pp_mean     = {f: float(np.nanmean([r['r2_per_prompt_mean'][f]   for r in rows])) for f in families}
    avg_pp_median   = {f: float(np.nanmean([r['r2_per_prompt_median'][f] for r in rows])) for f in families}
    avg_aic         = {f: float(np.nanmean([r['aic'][f]                  for r in rows])) for f in families}
    avg_bic         = {f: float(np.nanmean([r['bic'][f]                  for r in rows])) for f in families}

    print('Task-mean R²:'         + ' ' * 6 + ''.join(f"{avg_task_mean[f]:>10.3f}"  for f in families))
    print('Adj task-mean R²:'     + ' ' * 2 + ''.join(f"{avg_adj_mean[f]:>10.3f}"   for f in families))
    print('Pooled R²:'            + ' ' * 9 + ''.join(f"{avg_pooled[f]:>10.3f}"     for f in families))
    print('Adj pooled R²:'        + ' ' * 5 + ''.join(f"{avg_adj_pooled[f]:>10.3f}" for f in families))
    print('Per-prompt mean R²:'  + ' ' * 0 + ''.join(f"{avg_pp_mean[f]:>10.3f}"     for f in families))
    print('Per-prompt MEDIAN R²:' + ''.join(f"{avg_pp_median[f]:>10.3f}"             for f in families))
    print('AIC (lower better):'   + ''.join(f"{avg_aic[f]:>10.1f}"                   for f in families))
    print('BIC (lower better):'   + ''.join(f"{avg_bic[f]:>10.1f}"                   for f in families))

    # Save JSON for later use
    out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fit_r2_analysis.json')
    def _cast(x):
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, (np.integer,)):  return int(x)
        if isinstance(x, dict):           return {k: _cast(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):  return [_cast(v) for v in x]
        return x
    with open(out_json, 'w') as f:
        json.dump({
            'tasks': _cast(rows),
            'avg_task_mean':     _cast(avg_task_mean),
            'avg_adj_task_mean': _cast(avg_adj_mean),
            'avg_pooled':        _cast(avg_pooled),
            'avg_adj_pooled':    _cast(avg_adj_pooled),
            'avg_per_prompt_mean':   _cast(avg_pp_mean),
            'avg_per_prompt_median': _cast(avg_pp_median),
            'avg_aic': _cast(avg_aic),
            'avg_bic': _cast(avg_bic),
        }, f, indent=2)
    print(f"\nsaved → {out_json}")


if __name__ == '__main__':
    main()
