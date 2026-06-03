"""Figure 3 (Observation 3): Token selection is governed more by the
observation window (forgetting shape) than by the prompt itself.

Setup
-----
* LLaMA-3.2-1B, **all-layer mean attention** (per head).
* Hard observation windows W ∈ {1, 16, 128, "full"}.
  - "full" = use ALL prefill queries (H2O-style).
  - W ∈ {1, 16, 128} = SnapKV-style: only the last W queries contribute.
* Per (head): score_k = Σ_{q ∈ window} mean_layer attn[q, k]; top-B(=128) keys
  selected (always-keep recent_budget=16; matches SnapPolicy).
* 20 prompts, each ~4000 tokens (LongBench, length ∈ [3500, 4500], 5 per task).
* Position binned into 100 normalized bins; per-bin "selection density" =
  fraction of (head × prompt-position) selections falling in that bin,
  normalised so each curve sums to 1.

Outputs
-------
* fig3_data.npz             intermediate data (model inference cache).
* obs2_window_dominance.pdf combined 1×4 figure for paper textwidth.
"""
import os
import json
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.append("/home/smp9898/A2SF")


rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 1.1,
    "figure.dpi": 180,
})

# ── Config ──
MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
BUDGET = int(os.environ.get("BUDGET", "128"))
RECENT_BUDGET = 16
SUFFIX = "" if BUDGET == 128 else f"_b{BUDGET}"
NBINS = 100
SEED = 42
LENGTH_MIN, LENGTH_MAX = 3500, 4500
LONGBENCH_DIR = "/home/smp9898/A2SF/datasets/longbench"

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
PER_TASK = 5                                    # 5 prompts × 4 tasks = 20
TASK_FOR_A = "Multi-doc QA"


# ─────────────────────────────────────────────────────────────────
# Prompt loading
# ─────────────────────────────────────────────────────────────────
def load_prompts():
    pools = {}
    for fname in os.listdir(LONGBENCH_DIR):
        ds = fname.replace(".jsonl", "")
        with open(os.path.join(LONGBENCH_DIR, fname)) as f:
            for line in f:
                item = json.loads(line)
                length = item.get("length", 0)
                if LENGTH_MIN <= length <= LENGTH_MAX:
                    pools.setdefault(ds, []).append(item["input_prompt"])
    out = []                                    # list of (task, prompt)
    rng = random.Random(SEED)
    for task, ds in TASK_TO_DATASET.items():
        if ds not in pools:
            print(f"[warn] no prompt for {task} ({ds}) in length range")
            continue
        sample = rng.sample(pools[ds], min(PER_TASK, len(pools[ds])))
        for p in sample:
            out.append((task, p))
    return out


# ─────────────────────────────────────────────────────────────────
# All-layer average attention computation
# ─────────────────────────────────────────────────────────────────
def compute_densities(model, tokenizer, text, device, windows):
    """For each window in `windows`, run the standard attention-based top-$B$
    selection rule \emph{per (layer, head)} on that layer's attention map and
    accumulate selection counts per key position. Returns a dict
    {window → density vector}.

    The density vector aggregates selections across all
    (layer, head, prompt-position) decisions; setting `windows = [1, 16, 128, "full"]`
    instantiates the rule as TOVA, SnapKV, mid-window, H2O respectively.
    """
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=4500).input_ids.to(device)
    T = input_ids.size(1)
    H  = model.config.num_attention_heads
    KV = model.config.num_key_value_heads
    G  = H // KV
    D  = model.config.hidden_size // H

    pos  = torch.arange(T, device=device).unsqueeze(0)
    causal = torch.arange(T, device=device)[None] > torch.arange(T, device=device)[:, None]

    head_len = max(0, T - RECENT_BUDGET)
    select_b = max(0, BUDGET - RECENT_BUDGET)

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
            attn = torch.softmax(scores, dim=-1)        # (1, H, T, T)
            attn_np = attn[0].float().cpu().numpy()     # (H, T, T)

            for w in windows:
                if w == "full":
                    weights = np.ones(T, dtype=np.float32)
                else:
                    ww = int(w)
                    weights = np.zeros(T, dtype=np.float32)
                    weights[max(0, T - ww):] = 1.0
                score = (weights[None, :, None] * attn_np).sum(axis=1)   # (H, T)

                if T <= BUDGET:
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


# ─────────────────────────────────────────────────────────────────
# [previous helpers snap_select / density_from_mask removed —
#  compute_densities now produces window→density directly per-(layer, head).]


# ─────────────────────────────────────────────────────────────────
# Position bin density (kept for back-compat; not used in main flow)
# ─────────────────────────────────────────────────────────────────
def density_from_mask(mask, nbins=NBINS):
    """mask: (T,) → length-nbins density (sums to 1)."""
    T = mask.shape[0]
    counts = mask.astype(np.float32)                    # (T,) 1 if selected
    edges = np.linspace(0, T, nbins + 1).astype(int)
    hist = np.zeros(nbins, dtype=np.float32)
    for i in range(nbins):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        hist[i] = counts[lo:hi].sum()
    s = hist.sum()
    return hist / s if s > 0 else hist


def jaccard_continuous(a, b, eps=1e-12):
    """Tanimoto-style Jaccard for non-negative real vectors:
       Σ min(a, b) / Σ max(a, b)."""
    return float(np.minimum(a, b).sum() / max(np.maximum(a, b).sum(), eps))


# ─────────────────────────────────────────────────────────────────
# Combined paper figure (1×4 row, textwidth)
# ─────────────────────────────────────────────────────────────────
D_PER_TASK = 1   # for (d): one prompt per task → 4×4 heatmap

# Distinct linestyles for (a) so TOVA(W=1) vs SnapKV(W=16) are visually separable
WINDOW_STYLES = {
    1:      dict(color="tab:blue",   lw=2.5, ls="-"),
    16:     dict(color="tab:orange", lw=2.5, ls="--"),
    128:    dict(color="tab:green",  lw=2.0, ls="-."),
    "full": dict(color="tab:red",    lw=2.0, ls=":"),
}


def plot_combined(per_prompt_densities, prompt_meta, J_w, J_p, out_dir):
    """1×3 layout. Panel (a) uses a broken y-axis to show both the high spike
    near the rightmost bins and the low-density middle range."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(11, 4.0))
    outer = GridSpec(1, 3, figure=fig, width_ratios=[1.4, 1.0, 1.0],
                      wspace=0.40,
                      left=0.06, right=0.99, top=0.97, bottom=0.30)

    # Panel (a): broken y-axis. Top sub-axis = high range, bottom = low range.
    inner_a = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0],
                                       height_ratios=[1, 2], hspace=0.08)
    ax_a_top = fig.add_subplot(inner_a[0])
    ax_a_bot = fig.add_subplot(inner_a[1], sharex=ax_a_top)

    a_idx = next(i for i, (t, _) in enumerate(prompt_meta) if t == TASK_FOR_A)
    a_dens = per_prompt_densities[a_idx]
    for w in WINDOWS:
        ax_a_top.plot(np.arange(NBINS), a_dens[w], label=WINDOW_LABELS[w],
                       **WINDOW_STYLES[w])
        ax_a_bot.plot(np.arange(NBINS), a_dens[w], **WINDOW_STYLES[w])

    # Y-limits for the split (auto-tuned against the actual peak)
    peak = max(np.max(d) for d in a_dens.values())
    base_max = max(np.percentile(d, 99) for d in a_dens.values()
                   if np.max(d) < peak * 0.5)  # exclude the spike series
    if not np.isfinite(base_max) or base_max <= 0:
        base_max = peak * 0.18
    ax_a_top.set_ylim(peak * 0.55, peak * 1.07)
    ax_a_bot.set_ylim(0, base_max * 1.1)

    # Hide the spines between the two; place the y-label spanning both.
    ax_a_top.spines["bottom"].set_visible(False)
    ax_a_bot.spines["top"].set_visible(False)
    ax_a_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_a_top.tick_params(axis="y", labelsize=10)
    ax_a_bot.tick_params(axis="y", labelsize=10)
    ax_a_bot.set_xlabel("token position  (older $\\leftarrow$  ·  $\\rightarrow$ recent)")
    ax_a_bot.grid(True, alpha=0.3)
    ax_a_top.grid(True, alpha=0.3)

    # Diagonal break marks
    d_break = 0.015
    kw = dict(transform=ax_a_top.transAxes, color="k", clip_on=False, lw=1.0)
    ax_a_top.plot((-d_break, +d_break), (-d_break * 2, +d_break * 2), **kw)
    ax_a_top.plot((1 - d_break, 1 + d_break), (-d_break * 2, +d_break * 2), **kw)
    kw["transform"] = ax_a_bot.transAxes
    ax_a_bot.plot((-d_break, +d_break), (1 - d_break, 1 + d_break), **kw)
    ax_a_bot.plot((1 - d_break, 1 + d_break), (1 - d_break, 1 + d_break), **kw)

    ax_a_top.legend(loc="upper left", framealpha=0.95, fontsize=10)

    # Place y-axis label at the vertical centre of the combined broken axes,
    # and far enough to the left of the y-tick numbers that they don't overlap.
    fig.canvas.draw()                      # need bboxes resolved before measuring
    bbox_top = ax_a_top.get_tightbbox(fig.canvas.get_renderer())\
                       .transformed(fig.transFigure.inverted())
    bbox_bot = ax_a_bot.get_tightbbox(fig.canvas.get_renderer())\
                       .transformed(fig.transFigure.inverted())
    y_centre = (ax_a_top.get_position().y1 + ax_a_bot.get_position().y0) / 2
    x_label  = min(bbox_top.x0, bbox_bot.x0) - 0.012
    fig.text(x_label, y_centre, "selection density",
             rotation=90, va="center", ha="center", fontsize=13)

    # Panel (b)
    ax_b = fig.add_subplot(outer[1])
    ax = ax_b
    im = ax.imshow(J_w, cmap="Blues", vmin=0, vmax=1)
    labels_w = [WINDOW_LABELS[w].split(" ")[0] for w in WINDOWS]
    ax.set_xticks(range(len(WINDOWS))); ax.set_yticks(range(len(WINDOWS)))
    ax.set_xticklabels(labels_w, rotation=30, ha="right")
    ax.set_yticklabels(labels_w)
    for i in range(len(WINDOWS)):
        for j in range(len(WINDOWS)):
            ax.text(j, i, f"{J_w[i, j]:.2f}", ha="center", va="center",
                    color="white" if J_w[i, j] > 0.55 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel (c): 4×4 prompt×prompt similarity (one prompt per task)
    ax_c = fig.add_subplot(outer[2])
    ax = ax_c
    sub_idx = []
    counts = {t: 0 for t in TASK_TO_DATASET}
    for i, (t, _) in enumerate(prompt_meta):
        if counts.get(t, 0) < D_PER_TASK:
            sub_idx.append(i)
            counts[t] = counts.get(t, 0) + 1
    sub_idx = np.array(sub_idx)
    J_p_sub = J_p[np.ix_(sub_idx, sub_idx)]
    sub_meta = [prompt_meta[i] for i in sub_idx]

    im = ax.imshow(J_p_sub, cmap="Blues", vmin=0, vmax=1)
    short = {"Single-doc QA": "S-doc", "Multi-doc QA": "M-doc",
             "Summarization": "Sum.", "Few Shot": "Few-S"}
    labels_p = [short.get(t, t) for (t, _) in sub_meta]
    ax.set_xticks(range(len(sub_meta))); ax.set_yticks(range(len(sub_meta)))
    ax.set_xticklabels(labels_p, rotation=30, ha="right")
    ax.set_yticklabels(labels_p)
    for i in range(len(sub_meta)):
        for j in range(len(sub_meta)):
            ax.text(j, i, f"{J_p_sub[i, j]:.2f}", ha="center", va="center",
                    color="white" if J_p_sub[i, j] > 0.55 else "black",
                    fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    captions = [
        ("(a) One prompt, four windows",              ax_a_bot),
        ("(b) Cross-window Tanimoto sim.",             ax_b),
        (f"(c) Cross-prompt Tanimoto sim. ($W={FIXED_W}$)", ax_c),
    ]
    for txt, ax in captions:
        bbox = ax.get_position()
        xc = (bbox.x0 + bbox.x1) / 2
        fig.text(xc, 0.04, txt, ha="center", va="bottom", fontsize=13)

    fig.savefig(os.path.join(out_dir, f"obs2_window_dominance{SUFFIX}.pdf"),
                 bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"obs2_window_dominance{SUFFIX}.png"),
                 bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(out_dir, f"fig3_data{SUFFIX}.npz")

    if os.path.exists(cache_path):
        print(f"loading cached data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        per_prompt_densities = list(data["per_prompt_densities"])
        prompt_meta = [tuple(p) for p in data["prompt_meta"]]
        J_w = data["J_w"]
        J_p = data["J_p"]
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"loading {MODEL_PATH} …")
        tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16,
            device_map={"": device}).eval()

        prompt_meta = load_prompts()
        print(f"loaded {len(prompt_meta)} prompts")

        per_prompt_densities = []
        for i, (task, p) in enumerate(prompt_meta):
            print(f"  [{i + 1}/{len(prompt_meta)}] {task}", flush=True)
            densities, T = compute_densities(model, tok, p, device, WINDOWS)
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

    plot_combined(per_prompt_densities, prompt_meta, J_w, J_p, out_dir)

    nW = len(WINDOWS)
    n = J_p.shape[0]
    off_w = (J_w.sum() - np.trace(J_w)) / (nW * nW - nW)
    off_p = (J_p.sum() - np.trace(J_p)) / (n * n - n)
    print(f"\n[summary] mean off-diagonal Jaccard")
    print(f"  varying W (same prompt avg): {off_w:.3f}")
    print(f"  varying prompt (W={FIXED_W}): {off_p:.3f}")
    print("  → larger 'varying prompt' value ⇒ window dominates over prompt.")
    print(f"saved → {out_dir}/obs2_window_dominance{SUFFIX}.{{pdf,png}} (budget={BUDGET})")


if __name__ == "__main__":
    main()
