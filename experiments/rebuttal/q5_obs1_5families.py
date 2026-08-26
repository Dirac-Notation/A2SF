"""Rebuttal Q4: obs1_sigmoid_band layout with 4 weight families fit.

Mirrors paper's obs1_sigmoid_band.pdf (mean ± std + mean line) but plots
4 fit families side by side: sigmoid / exp / linear / step.

Each family's task-mean R² and per-prompt MEDIAN R² shown in a text box.

Data source: eslab1's coord_descent.npz, key "w_tan" (Tanimoto-optimal weight).
Sigmoid choice rationale: see fit_r2_analysis.json.

Usage:
  python experiments/paper_figures/rebuttal/q4_obs1_4families.py
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit

ROOT = '/home/smp9898/A2SF'
sys.path.insert(0, ROOT)

rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.labelsize': 13, 'axes.titlesize': 14,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 13, 'axes.linewidth': 1.0,
    'figure.dpi': 150,
})

BUDGET = int(os.environ.get('BUDGET', '128'))
# Tanimoto-optimal w_tan from the obs1 pipeline (budget-independent).
DATA   = os.path.join(ROOT, 'experiments/paper_figures/observations/data')
SUFFIX = '' if BUDGET == 128 else f'_b{BUDGET}'

TASKS = [
    ('Single-doc_QA/qasper',     'Single-doc QA'),
    ('Multi-doc_QA/hotpotqa',    'Multi-doc QA'),
    ('Summarization/gov_report', 'Summarization'),
    ('Few_Shot/samsum',          'Few-Shot'),
]

# ── 4 families ────────────────────────────────────────────────────────────────
def sigmoid_fn(d, a, b):
    return 1.0 / (1.0 + np.exp(np.clip(a * (d - b), -30.0, 30.0)))

def exp_fn(d, tau):
    return np.exp(-d / max(tau, 1e-6))

def linear_fn(d, d0):
    return np.clip(1.0 - d / max(d0, 1e-6), 0.0, 1.0)

def step_fn(d, d0):
    return (d < d0).astype(np.float32)

def gaussian_fn(d, sigma):
    return np.exp(-0.5 * (d / max(sigma, 1e-6)) ** 2)

def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float('nan') if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot

def fit_sigmoid(d, y, W):
    try:
        popt, _ = curve_fit(sigmoid_fn, d, y, p0=[0.05, float(W)/4.0],
                            bounds=([0.0, 0.0], [5.0, float(W)]), maxfev=5000)
        return sigmoid_fn(d, *popt), popt, r2(y, sigmoid_fn(d, *popt))
    except Exception:
        return np.full_like(d, np.nan), None, float('nan')

def fit_exp(d, y, W):
    try:
        popt, _ = curve_fit(exp_fn, d, y, p0=[float(W)/4.0],
                            bounds=([1e-3], [10.0*float(W)]), maxfev=5000)
        return exp_fn(d, *popt), popt, r2(y, exp_fn(d, *popt))
    except Exception:
        return np.full_like(d, np.nan), None, float('nan')

def fit_linear(d, y, W):
    try:
        popt, _ = curve_fit(linear_fn, d, y, p0=[float(W)/4.0],
                            bounds=([1e-3], [10.0*float(W)]), maxfev=5000)
        return linear_fn(d, *popt), popt, r2(y, linear_fn(d, *popt))
    except Exception:
        return np.full_like(d, np.nan), None, float('nan')

def fit_step(d, y, W):
    best = (None, -float('inf'))
    for d0 in np.unique(np.concatenate([d, [d.max()+1.0]])):
        cur = r2(y, step_fn(d, d0))
        if cur > best[1]:
            best = (d0, cur)
    return step_fn(d, best[0]), [best[0]], best[1]

def fit_gaussian(d, y, W):
    try:
        popt, _ = curve_fit(gaussian_fn, d, y, p0=[float(W)/4.0],
                            bounds=([1e-3], [10.0*float(W)]), maxfev=5000)
        return gaussian_fn(d, *popt), popt, r2(y, gaussian_fn(d, *popt))
    except Exception:
        return np.full_like(d, np.nan), None, float('nan')

FAMILIES = [
    ('sigmoid', fit_sigmoid, '#1f77b4'),   # blue
    ('exp',     fit_exp,     '#ff7f0e'),   # orange
    ('linear',  fit_linear,  '#2ca02c'),   # green
    ('gauss',   fit_gaussian,'#9467bd'),   # purple
    ('step',    fit_step,    '#d62728'),   # red
]


def per_prompt_median_r2(w_tan, d, W, fit_fn):
    """Median R² across prompts (robust to outliers)."""
    rs = []
    for y in w_tan:
        if y.var() < 1e-12: continue
        _, _, r = fit_fn(d, y, W)
        rs.append(r)
    return float(np.nanmedian(rs)) if rs else float('nan')


def draw_4family(out_dir):
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8))
    band_colors = plt.get_cmap('tab10').colors

    legend_handles = []
    legend_labels  = []

    for col, ((path, label), bc) in enumerate(zip(TASKS, band_colors)):
        cd = np.load(os.path.join(DATA, path.replace('/', '__') + '.npz'))
        w = cd['w_tan']                                # (N, G)
        N, G = w.shape
        chunk = int(cd['chunk']); W = int(cd['window'])
        d = np.arange(G).astype(float) * chunk + (chunk - 1) / 2.0

        mean = w.mean(axis=0)
        std  = w.std(axis=0)
        lo   = np.clip(mean - std, 0.0, None)
        hi   = np.clip(mean + std, None, 1.0)

        ax = axes[col]

        # mean ± std band + mean line (same style as obs1_sigmoid_band.pdf)
        h_band = ax.fill_between(d, lo, hi, color=bc, alpha=0.25,
                                  linewidth=0, label='mean ± std')
        h_mean, = ax.plot(d, mean, color=bc, lw=2.0, label='mean')

        # 4 family fits + R² (task-mean + per-prompt median)
        r2_lines = []
        fit_handles = []
        for fam_name, fit_fn, fcolor in FAMILIES:
            yhat, popt, r2_mean = fit_fn(d, mean, W)
            ls = '--' if fam_name == 'sigmoid' else (':' if fam_name == 'exp'
                  else ('-.' if fam_name == 'linear' else (0, (3, 1, 1, 1))))
            h, = ax.plot(d, yhat, ls=ls, color=fcolor, lw=2.0,
                          label=f'{fam_name} fit')
            fit_handles.append((fam_name, h))
            r2_lines.append(f"{fam_name:>7s}: R²={r2_mean:+0.2f}")

        # Annotate R² (upper-left)
        ax.text(0.03, 0.97, '\n'.join(r2_lines),
                transform=ax.transAxes, fontsize=8.5, ha='left', va='top',
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor='white', edgecolor='0.8', alpha=0.9))

        ax.set_xlim(0, W - 1)
        ax.invert_xaxis()
        ax.set_ylim(-0.05, 1.10)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel(r'optimal $w_d$')
            legend_handles = [h_band, h_mean] + [h for _, h in fit_handles]
            legend_labels  = ['mean ± std', 'mean'] + [n for n, _ in fit_handles]

    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.16, wspace=0.28)
    fig.text(0.525, 0.02,
             r'query distance $d$   ($\leftarrow$ older  $\cdot$  recent $\rightarrow$)',
             ha='center', fontsize=12)
    # Legend at top of figure
    fig.legend(legend_handles, legend_labels,
                loc='upper center', bbox_to_anchor=(0.5, 0.99),
                ncol=len(legend_labels), frameon=False)

    out_pdf = os.path.join(out_dir, f'q5_obs1_5families{SUFFIX}.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_pdf.replace('.pdf', '.png'), bbox_inches='tight')
    plt.close(fig)
    print(f'saved → {out_pdf}')


if __name__ == '__main__':
    out_dir = os.path.dirname(os.path.abspath(__file__))
    draw_4family(out_dir)
