"""Observation 1 figures: optimal forgetting curve is a sigmoid.

Produces exactly two paper figures:
  obs1_sigmoid_band.pdf      per-prompt Tanimoto-optimal weight w_d (mean ± std)
                             with a sigmoid fit to the mean, 4 tasks.
  obs1_tanimoto_recovery.pdf running Tanimoto similarity to the oracle for
                             single-query / optimal / uniform / sigmoid weighting.

Data flow
---------
  generate():  LongBench prompts -> 1B prefill (windowed attention) +
               teacher-forced oracle attention -> per-prompt Tanimoto curves.
               Only the small (N, G) curves are saved to data/<task>.npz;
               the large prefill/oracle tensors are consumed and discarded.
  plot():      data/<task>.npz -> the two PDFs.

Produces the B=128 main-paper figures (window=256, chunk=4).

Usage
-----
  python obs1.py                # generate (if data missing) then plot
  python obs1.py --plot-only    # re-plot from existing data/ (no GPU)
"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rcParams.update({
    "font.family": "serif", "font.size": 12,
    "axes.labelsize": 13, "axes.titlesize": 14,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 13, "axes.linewidth": 1.0,
    "figure.dpi": 150,
})

HERE = os.path.dirname(os.path.abspath(__file__))

_SCHEME_STYLE = {
    "single":  dict(color="gray",       lw=1.5, ls="--"),
    "optimal": dict(color="black",      lw=2.0, ls="-"),
    "uniform": dict(color="steelblue",  lw=1.5, ls="-"),
    "sigmoid": dict(color="darkorange", lw=1.5, ls="--"),
}
_SCHEME_LABEL = {
    "single":  "Kronecker delta weighting",
    "optimal": "Optimal Weighting",
    "uniform": "Step Weighting",
    "sigmoid": "Sigmoid Weighting",
}
# (scheme, npz key holding the running tanimoto curve)
_TAN_CURVES = [
    ("single",  "j_single_tanimoto"),
    ("optimal", "j_tan_optimal"),
    ("uniform", "j_uniform_tanimoto"),
    ("sigmoid", "j_tan_sigmoid"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Generate (GPU) — minimal per-task curves
# ══════════════════════════════════════════════════════════════════════════════
def generate(nitems=C.NUM_ITEMS, data_suffix="", device="cuda"):
    import time
    os.makedirs(C.DATA_DIR, exist_ok=True)
    selected = C.sample_prompts(nitems)
    backup_pred = C.load_backup_preds()
    tok, model = C.load_model(device)
    collector = C.AttentionCollector(model, C.MAX_WINDOW)

    for task_path, label in C.OBS1_TASKS:
        dataset = task_path.split("/")[1]
        items = selected[dataset]
        ds_preds = backup_pred.get(dataset, [])
        print(f"\n{'='*70}\n  {label}: {dataset}  ({len(items)} samples)\n{'='*70}")

        agg = {k: [] for k in ("w_tan", "j_tan_optimal", "j_tan_sigmoid",
                                "j_uniform_tanimoto", "j_single_tanimoto", "seq_lens")}
        for si, (row_idx, prompt) in enumerate(items):
            enc = tok(f"[INST]{prompt}[/INST]", return_tensors="pt")
            input_ids = enc.input_ids.to(model.device)
            if input_ids.size(1) > C.MAX_SEQ_LEN:
                half = C.MAX_SEQ_LEN // 2
                input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)
            seq_len = input_ids.size(1)

            pred_text = ds_preds[row_idx] if row_idx < len(ds_preds) else ""
            if not pred_text:
                raise RuntimeError(
                    f"empty full-cache pred for {dataset} row {row_idx}; "
                    "the paper run teacher-forces a non-empty prediction")

            collector.reset(seq_len)
            t0 = time.time()
            with torch.no_grad():
                pf_out = model(input_ids, use_cache=True, num_logits_to_keep=1)
                past_kv = pf_out.past_key_values
                del pf_out; torch.cuda.empty_cache()
                data = collector.compute_window_data(past_kv)
            prefill_attn = data["prefill_attn"]                 # (L, H, W, S)

            answer_score = C.teacher_forcing_answer_score(
                model, tok, pred_text, past_kv, seq_len, model.device)
            del past_kv; torch.cuda.empty_cache()

            # ── reduce to kv-head space, compute the small curves, discard big tensors ──
            pf_kv = C.prefill_to_pf_kv(prefill_attn, C.CHUNK)    # (G, L, kv, S)
            oracle_norm = C.oracle_to_norm(answer_score, seq_len)
            G = pf_kv.shape[0]
            W = int(C.MAX_WINDOW)

            w_tan, j_opt, j_sig = C.tanimoto_optimal(pf_kv, oracle_norm, G, W, C.CHUNK)
            j_uni = C.tanimoto_uniform(pf_kv, oracle_norm, G)
            j_single = C.tanimoto_single(pf_kv, oracle_norm, G)

            agg["w_tan"].append(w_tan)
            agg["j_tan_optimal"].append(j_opt)
            agg["j_tan_sigmoid"].append(j_sig)
            agg["j_uniform_tanimoto"].append(j_uni)
            agg["j_single_tanimoto"].append(j_single)
            agg["seq_lens"].append(seq_len)

            del data, prefill_attn, answer_score, pf_kv, oracle_norm
            torch.cuda.empty_cache()
            print(f"  [{si+1}/{len(items)}] L={seq_len}  t={time.time()-t0:.1f}s  "
                  f"w_tan[0..2]={w_tan[:3]}  j_opt={j_opt[-1]:.3f}", flush=True)

        out = os.path.join(C.DATA_DIR, f"{C.data_stem(task_path)}{data_suffix}.npz")
        np.savez_compressed(
            out,
            w_tan=np.stack(agg["w_tan"]),
            j_tan_optimal=np.stack(agg["j_tan_optimal"]),
            j_tan_sigmoid=np.stack(agg["j_tan_sigmoid"]),
            j_uniform_tanimoto=np.stack(agg["j_uniform_tanimoto"]),
            j_single_tanimoto=np.stack(agg["j_single_tanimoto"]),
            seq_lens=np.array(agg["seq_lens"], dtype=np.int32),
            chunk=np.int32(C.CHUNK), window=np.int32(C.MAX_WINDOW),
        )
        print(f"  saved → {out}  ({os.path.getsize(out)/1024:.0f} KB)")

    collector.remove_hooks()
    print("\n>>> obs1 data generation done.")


def _load_task(task_path, data_suffix=""):
    return np.load(os.path.join(C.DATA_DIR, f"{C.data_stem(task_path)}{data_suffix}.npz"))


# ══════════════════════════════════════════════════════════════════════════════
# Plot — obs1_sigmoid_band.pdf
# ══════════════════════════════════════════════════════════════════════════════
def plot_sigmoid_band(fig_suffix="", data_suffix=""):
    from scipy.optimize import curve_fit
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharex=False, sharey=False)
    h_proxy = []

    for col, ((task_path, label), c) in enumerate(zip(C.OBS1_TASKS, colors)):
        cd = _load_task(task_path, data_suffix)
        w = cd["w_tan"]                                   # (N, G)
        G = w.shape[1]
        chunk = int(cd["chunk"]); W = int(cd["window"])
        d = np.arange(G) * chunk + (chunk - 1) / 2.0

        w_mean = w.mean(axis=0); w_std = w.std(axis=0)
        lo = np.clip(w_mean - w_std, 0.0, None)
        hi = np.clip(w_mean + w_std, None, 1.0)

        try:
            popt, _ = curve_fit(C.sigmoid, d, w_mean, p0=[0.05, float(W) / 4.0],
                                bounds=([0.0, 0.0], [5.0, float(W)]), maxfev=5000)
            a_fit, b_fit = popt
            fit_y = C.sigmoid(d, a_fit, b_fit)
            fit_label = f"$a={a_fit:.2f},\\ b={b_fit:.1f}$"
        except Exception:
            a_fit = b_fit = float("nan")
            fit_y = np.full_like(d, np.nan); fit_label = ""

        ax = axes[col]
        lb = ax.fill_between(d, lo, hi, color=c, alpha=0.30, linewidth=0, label="mean ± std")
        lm, = ax.plot(d, w_mean, color=c, lw=2.0, label="mean")
        lf, = ax.plot(d, fit_y, "--", color="black", lw=1.5, label="sigmoid fit")
        if np.isfinite(a_fit):
            ax.text(0.97, 0.95, fit_label, transform=ax.transAxes, fontsize=9,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="0.8", alpha=0.85))
        ax.set_xlim(0, W - 1); ax.invert_xaxis(); ax.set_ylim(-0.05, 1.10)
        ax.set_title(label); ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("optimal $w_d$", fontsize=11)
            h_proxy = [lb, lm, lf]

    fig.text(0.525, 0.02,
             r"query distance $d$   ($\leftarrow$ older  $\cdot$  recent $\rightarrow$)",
             ha="center", fontsize=12)
    if h_proxy:
        fig.legend(h_proxy, ["mean ± std", "mean", "sigmoid fit"],
                   loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.16, wspace=0.28)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"obs1_sigmoid_band{fig_suffix}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved → obs1_sigmoid_band{fig_suffix}.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Plot — obs1_tanimoto_recovery.pdf
# ══════════════════════════════════════════════════════════════════════════════
def plot_tanimoto_recovery(fig_suffix="", data_suffix=""):
    # global ylim from all task data (5% pad)
    all_vals = []
    for task_path, _ in C.OBS1_TASKS:
        cd = _load_task(task_path, data_suffix)
        chunk = int(cd["chunk"]); G = cd["w_tan"].shape[1]; W = G * chunk
        for _, key in _TAN_CURVES:
            all_vals.extend(np.repeat(cd[key].mean(axis=0), chunk)[:W].tolist())
    lo, hi = min(all_vals), max(all_vals)
    span = hi - lo
    ylim = (lo - 0.05 * span, hi + 0.05 * span)

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2))
    h_proxy, all_labels = [], []
    for col, (task_path, label) in enumerate(C.OBS1_TASKS):
        cd = _load_task(task_path, data_suffix)
        chunk = int(cd["chunk"]); G = cd["w_tan"].shape[1]; W = G * chunk
        ax = axes[col]; d = np.arange(W); handles_this = []
        for scheme, key in _TAN_CURVES:
            curve = np.repeat(cd[key].mean(axis=0), chunk)[:W]
            h, = ax.plot(d, curve, label=_SCHEME_LABEL[scheme], **_SCHEME_STYLE[scheme])
            handles_this.append((_SCHEME_LABEL[scheme], h))
        ax.set_xlim(0, W - 1); ax.set_ylim(*ylim); ax.invert_xaxis()
        if col == 0:
            ax.set_ylabel("Tanimoto similarity")
            h_proxy = [h for _, h in handles_this]
            all_labels = [lbl for lbl, _ in handles_this]
        ax.set_title(label); ax.grid(True, alpha=0.3)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.16, wspace=0.30)
    fig.text(0.525, 0.03,
             r"query distance $d$   ($\leftarrow$ older  $\cdot$  recent $\rightarrow$)",
             ha="center", fontsize=13)
    fig.legend(h_proxy, all_labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=min(5, len(h_proxy)), frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"obs1_tanimoto_recovery{fig_suffix}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"saved → obs1_tanimoto_recovery{fig_suffix}.pdf")


# Figure variants. w_tan is budget-independent, so the B=256 appendix figure is
# identical to B=128 (same N=40 data, reused). B=512 used only 20 prompts.
#   variant -> (nitems, fig_suffix, data_suffix)
VARIANTS = {
    "b128": (C.NUM_ITEMS, "",      ""),       # main paper figure (N=40)
    "b256": (C.NUM_ITEMS, "_b256", ""),       # appendix: identical to b128, reuses its data
    "b512": (20,          "_b512", "_b512"),  # appendix: N=20
}


def _data_exists(data_suffix=""):
    return all(os.path.isfile(os.path.join(C.DATA_DIR, f"{C.data_stem(p)}{data_suffix}.npz"))
               for p, _ in C.OBS1_TASKS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--variant", choices=list(VARIANTS), default="b128")
    args = ap.parse_args()
    nitems, fig_suffix, data_suffix = VARIANTS[args.variant]

    if not args.plot_only and not _data_exists(data_suffix):
        generate(nitems=nitems, data_suffix=data_suffix)
    elif not args.plot_only:
        print(f"data{data_suffix or '/'} already present; skipping generation.")

    plot_sigmoid_band(fig_suffix, data_suffix)
    plot_tanimoto_recovery(fig_suffix, data_suffix)


if __name__ == "__main__":
    main()
