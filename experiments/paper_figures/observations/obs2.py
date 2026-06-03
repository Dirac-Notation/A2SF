"""Observation 2 figure: token selection is governed by the observation window
(forgetting shape) more than by the prompt.

Produces one paper figure:
  obs2_window_dominance.pdf   1x3 panels:
    (a) one prompt, four windows (TOVA / SnapKV / W=128 / H2O) selection density
    (b) cross-window Tanimoto similarity (avg over prompts)
    (c) cross-prompt Tanimoto similarity at W=16

Self-contained: runs LLaMA-3.2-1B, computes per-(layer, head) top-B selection
under each hard window from all-layer attention, and bins selections into a
normalized position density.  Numerics and layout are ported verbatim from the
original fig3_window_dominance.py; only hardcoded paths were removed.

This figure IS budget-dependent (top-B selection), so --budget regenerates data.

Usage
-----
  python obs2.py                # generate (if cache missing) then plot
  python obs2.py --plot-only    # re-plot from cached data/obs2_data.npz
  python obs2.py --budget 256   # B=256 figure
"""
import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rcParams.update({
    "font.family": "serif", "font.size": 13,
    "axes.labelsize": 14, "axes.titlesize": 16,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "axes.linewidth": 1.1,
    "figure.dpi": 180,
})

HERE = os.path.dirname(os.path.abspath(__file__))

# ── obs2 config (distinct from obs1) ───────────────────────────────────────────
RECENT_BUDGET = 16
NBINS = 100
OBS2_LENGTH_MIN, OBS2_LENGTH_MAX = 3500, 4500
WINDOWS = [1, 16, 128, "full"]                 # "full" = H2O-style
WINDOW_LABELS = {1: "W=1 (TOVA)", 16: "W=16 (SnapKV)",
                 128: "W=128", "full": "W=full (H2O)"}
FIXED_W = 16
TASK_TO_DATASET = {
    "Single-doc QA": "qasper",
    "Multi-doc QA":  "hotpotqa",
    "Summarization": "gov_report",
    "Few Shot":      "samsum",
}
PER_TASK = 5
TASK_FOR_A = "Multi-doc QA"
D_PER_TASK = 1
WINDOW_STYLES = {
    1:      dict(color="tab:blue",   lw=2.5, ls="-"),
    16:     dict(color="tab:orange", lw=2.5, ls="--"),
    128:    dict(color="tab:green",  lw=2.0, ls="-."),
    "full": dict(color="tab:red",    lw=2.0, ls=":"),
}


def load_prompts():
    pools = {}
    for fname in os.listdir(C.LONGBENCH_DIR):
        ds = fname.replace(".jsonl", "")
        with open(os.path.join(C.LONGBENCH_DIR, fname)) as f:
            for line in f:
                item = json.loads(line)
                if OBS2_LENGTH_MIN <= item.get("length", 0) <= OBS2_LENGTH_MAX:
                    pools.setdefault(ds, []).append(item["input_prompt"])
    out = []
    rng = random.Random(C.SEED)
    for task, ds in TASK_TO_DATASET.items():
        if ds not in pools:
            print(f"[warn] no prompt for {task} ({ds}) in length range")
            continue
        sample = rng.sample(pools[ds], min(PER_TASK, len(pools[ds])))
        for p in sample:
            out.append((task, p))
    return out


def compute_densities(model, tokenizer, text, device, windows, budget):
    """{window -> normalized selection density (NBINS,)} for one prompt."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=4500).input_ids.to(device)
    T = input_ids.size(1)
    H  = model.config.num_attention_heads
    KV = model.config.num_key_value_heads
    G  = H // KV
    D  = model.config.hidden_size // H

    pos = torch.arange(T, device=device).unsqueeze(0)
    causal = torch.arange(T, device=device)[None] > torch.arange(T, device=device)[:, None]
    head_len = max(0, T - RECENT_BUDGET)
    select_b = max(0, budget - RECENT_BUDGET)
    counts = {w: np.zeros(T, dtype=np.int64) for w in windows}

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
        hidden_states = list(out.hidden_states)
        del out
        torch.cuda.empty_cache()

        for layer_idx, layer in enumerate(model.model.layers):
            h = hidden_states[layer_idx]
            ln_h = layer.input_layernorm(h).to(layer.self_attn.q_proj.weight.dtype)
            q = layer.self_attn.q_proj(ln_h).view(1, T, H,  D).transpose(1, 2)
            k = layer.self_attn.k_proj(ln_h).view(1, T, KV, D).transpose(1, 2)
            cos, sin = layer.self_attn.rotary_emb(k, pos)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = repeat_kv(k, G)
            scores = (q @ k.transpose(-2, -1)) / (D ** 0.5)
            scores = scores.masked_fill(causal[None, None], float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn_np = attn[0].float().cpu().numpy()         # (H, T, T)

            for w in windows:
                if w == "full":
                    weights = np.ones(T, dtype=np.float32)
                else:
                    ww = int(w)
                    weights = np.zeros(T, dtype=np.float32)
                    weights[max(0, T - ww):] = 1.0
                score = (weights[None, :, None] * attn_np).sum(axis=1)   # (H, T)
                if T <= budget:
                    counts[w] += H
                    continue
                if select_b > 0 and head_len > 0:
                    if select_b >= head_len:
                        counts[w][:head_len] += H
                    else:
                        sub = score[:, :head_len]
                        idx = np.argpartition(-sub, select_b - 1, axis=1)[:, :select_b]
                        for hh in range(H):
                            counts[w][idx[hh]] += 1
                if RECENT_BUDGET > 0:
                    counts[w][head_len:T] += H

            hidden_states[layer_idx] = None
            del scores, attn, q, k, ln_h, attn_np
            torch.cuda.empty_cache()

    edges = np.linspace(0, T, NBINS + 1).astype(int)
    densities = {}
    for w in windows:
        cnt = counts[w].astype(np.float32)
        hist = np.zeros(NBINS, dtype=np.float32)
        for i in range(NBINS):
            lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
            hist[i] = cnt[lo:hi].sum()
        s = hist.sum()
        densities[w] = hist / s if s > 0 else hist
    return densities, T


def jaccard_continuous(a, b, eps=1e-12):
    """Tanimoto-style Jaccard for non-negative real vectors: Σ min / Σ max."""
    return float(np.minimum(a, b).sum() / max(np.maximum(a, b).sum(), eps))


def plot_combined(per_prompt_densities, prompt_meta, J_w, J_p, out_dir, suffix):
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(11, 4.0))
    outer = GridSpec(1, 3, figure=fig, width_ratios=[1.4, 1.0, 1.0],
                     wspace=0.40, left=0.06, right=0.99, top=0.97, bottom=0.30)

    inner_a = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0],
                                      height_ratios=[1, 2], hspace=0.08)
    ax_a_top = fig.add_subplot(inner_a[0])
    ax_a_bot = fig.add_subplot(inner_a[1], sharex=ax_a_top)

    a_idx = next(i for i, (t, _) in enumerate(prompt_meta) if t == TASK_FOR_A)
    a_dens = per_prompt_densities[a_idx]
    for w in WINDOWS:
        ax_a_top.plot(np.arange(NBINS), a_dens[w], label=WINDOW_LABELS[w], **WINDOW_STYLES[w])
        ax_a_bot.plot(np.arange(NBINS), a_dens[w], **WINDOW_STYLES[w])

    peak = max(np.max(d) for d in a_dens.values())
    base_max = max(np.percentile(d, 99) for d in a_dens.values() if np.max(d) < peak * 0.5)
    if not np.isfinite(base_max) or base_max <= 0:
        base_max = peak * 0.18
    ax_a_top.set_ylim(peak * 0.55, peak * 1.07)
    ax_a_bot.set_ylim(0, base_max * 1.1)

    ax_a_top.spines["bottom"].set_visible(False)
    ax_a_bot.spines["top"].set_visible(False)
    ax_a_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_a_top.tick_params(axis="y", labelsize=10)
    ax_a_bot.tick_params(axis="y", labelsize=10)
    ax_a_bot.set_xlabel("token position  (older $\\leftarrow$  ·  $\\rightarrow$ recent)")
    ax_a_bot.grid(True, alpha=0.3); ax_a_top.grid(True, alpha=0.3)

    d_break = 0.015
    kw = dict(transform=ax_a_top.transAxes, color="k", clip_on=False, lw=1.0)
    ax_a_top.plot((-d_break, +d_break), (-d_break * 2, +d_break * 2), **kw)
    ax_a_top.plot((1 - d_break, 1 + d_break), (-d_break * 2, +d_break * 2), **kw)
    kw["transform"] = ax_a_bot.transAxes
    ax_a_bot.plot((-d_break, +d_break), (1 - d_break, 1 + d_break), **kw)
    ax_a_bot.plot((1 - d_break, 1 + d_break), (1 - d_break, 1 + d_break), **kw)

    ax_a_top.legend(loc="upper left", framealpha=0.95, fontsize=10)

    fig.canvas.draw()
    bbox_top = ax_a_top.get_tightbbox(fig.canvas.get_renderer())\
                       .transformed(fig.transFigure.inverted())
    bbox_bot = ax_a_bot.get_tightbbox(fig.canvas.get_renderer())\
                       .transformed(fig.transFigure.inverted())
    y_centre = (ax_a_top.get_position().y1 + ax_a_bot.get_position().y0) / 2
    x_label = min(bbox_top.x0, bbox_bot.x0) - 0.012
    fig.text(x_label, y_centre, "selection density",
             rotation=90, va="center", ha="center", fontsize=13)

    ax_b = fig.add_subplot(outer[1])
    im = ax_b.imshow(J_w, cmap="Blues", vmin=0, vmax=1)
    labels_w = [WINDOW_LABELS[w].split(" ")[0] for w in WINDOWS]
    ax_b.set_xticks(range(len(WINDOWS))); ax_b.set_yticks(range(len(WINDOWS)))
    ax_b.set_xticklabels(labels_w, rotation=30, ha="right")
    ax_b.set_yticklabels(labels_w)
    for i in range(len(WINDOWS)):
        for j in range(len(WINDOWS)):
            ax_b.text(j, i, f"{J_w[i, j]:.2f}", ha="center", va="center",
                      color="white" if J_w[i, j] > 0.55 else "black", fontsize=10)
    fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)

    ax_c = fig.add_subplot(outer[2])
    sub_idx = []
    counts = {t: 0 for t in TASK_TO_DATASET}
    for i, (t, _) in enumerate(prompt_meta):
        if counts.get(t, 0) < D_PER_TASK:
            sub_idx.append(i); counts[t] = counts.get(t, 0) + 1
    sub_idx = np.array(sub_idx)
    J_p_sub = J_p[np.ix_(sub_idx, sub_idx)]
    sub_meta = [prompt_meta[i] for i in sub_idx]
    im = ax_c.imshow(J_p_sub, cmap="Blues", vmin=0, vmax=1)
    short = {"Single-doc QA": "S-doc", "Multi-doc QA": "M-doc",
             "Summarization": "Sum.", "Few Shot": "Few-S"}
    labels_p = [short.get(t, t) for (t, _) in sub_meta]
    ax_c.set_xticks(range(len(sub_meta))); ax_c.set_yticks(range(len(sub_meta)))
    ax_c.set_xticklabels(labels_p, rotation=30, ha="right")
    ax_c.set_yticklabels(labels_p)
    for i in range(len(sub_meta)):
        for j in range(len(sub_meta)):
            ax_c.text(j, i, f"{J_p_sub[i, j]:.2f}", ha="center", va="center",
                      color="white" if J_p_sub[i, j] > 0.55 else "black", fontsize=10)
    fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)

    for txt, ax in [("(a) One prompt, four windows", ax_a_bot),
                    ("(b) Cross-window Tanimoto sim.", ax_b),
                    (f"(c) Cross-prompt Tanimoto sim. ($W={FIXED_W}$)", ax_c)]:
        bbox = ax.get_position()
        fig.text((bbox.x0 + bbox.x1) / 2, 0.04, txt, ha="center", va="bottom", fontsize=13)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"obs2_window_dominance{suffix}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"saved → obs2_window_dominance{suffix}.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--budget", type=int, default=128, choices=[128, 256, 512])
    args = ap.parse_args()
    suffix = "" if args.budget == 128 else f"_b{args.budget}"

    os.makedirs(C.DATA_DIR, exist_ok=True)
    cache_path = os.path.join(C.DATA_DIR, f"obs2_data{suffix}.npz")

    if os.path.exists(cache_path) or args.plot_only:
        print(f"loading cached data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        per_prompt_densities = list(data["per_prompt_densities"])
        prompt_meta = [tuple(p) for p in data["prompt_meta"]]
        J_w = data["J_w"]; J_p = data["J_p"]
    else:
        tok, model = C.load_model("cuda" if torch.cuda.is_available() else "cpu",
                                  dtype=torch.float16)
        device = model.device
        prompt_meta = load_prompts()
        print(f"loaded {len(prompt_meta)} prompts")

        per_prompt_densities = []
        for i, (task, p) in enumerate(prompt_meta):
            print(f"  [{i+1}/{len(prompt_meta)}] {task}", flush=True)
            densities, T = compute_densities(model, tok, p, device, WINDOWS, args.budget)
            per_prompt_densities.append(densities)
            torch.cuda.empty_cache()

        nW = len(WINDOWS)
        J_w = np.zeros((nW, nW))
        for d in per_prompt_densities:
            for i, wi in enumerate(WINDOWS):
                for j, wj in enumerate(WINDOWS):
                    J_w[i, j] += jaccard_continuous(d[wi], d[wj])
        J_w /= len(per_prompt_densities)

        b_dens = np.stack([d[FIXED_W] for d in per_prompt_densities])
        n = b_dens.shape[0]
        J_p = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                J_p[i, j] = jaccard_continuous(b_dens[i], b_dens[j])

        np.savez_compressed(
            cache_path,
            per_prompt_densities=np.array(per_prompt_densities, dtype=object),
            prompt_meta=np.array(prompt_meta, dtype=object),
            J_w=J_w, J_p=J_p,
        )
        print(f"cached → {cache_path}")

    plot_combined(per_prompt_densities, prompt_meta, J_w, J_p, HERE, suffix)

    nW = len(WINDOWS); n = J_p.shape[0]
    off_w = (J_w.sum() - np.trace(J_w)) / (nW * nW - nW)
    off_p = (J_p.sum() - np.trace(J_p)) / (n * n - n)
    print(f"[summary] mean off-diagonal Tanimoto: varying W={off_w:.3f}  "
          f"varying prompt(W={FIXED_W})={off_p:.3f}  (larger 'varying prompt' ⇒ window dominates)")


if __name__ == "__main__":
    main()
