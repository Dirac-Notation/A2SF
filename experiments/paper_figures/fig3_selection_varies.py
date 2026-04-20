"""Figure 3 (Observation 3): Token selection is dominated by the accumulation
coefficient α; varying the prompt at fixed α causes comparatively small shifts.

Clean 1×2 layout:
  (a) Same prompt, varying α.  We select top-B keys using only one query weight
      function per α value, then compute the Jaccard overlap between every pair of
      α-induced selections.  Low off-diagonal values ⇒ the coefficient is a
      strong driver of which keys survive.

  (b) Fixed α, varying prompts (one per task, different "important" positions).
      Jaccard between their selections.  Higher off-diagonal values ⇒ at the
      same α, the selection patterns are similar regardless of prompt.

Setup: LLaMA-3.2-1B first-layer attention, budget B = 128.
"""
import os
import json
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.append("/home/smp9898/A2SF")


rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "figure.dpi": 150,
})


MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
BUDGET = 128
ALPHA_SET = [0.0, 0.001, 0.01, 0.1, 1.0]       # 5 representative values
FIXED_ALPHA = 0.01

# ── Unified prompt selection (matches optimal.py used by fig1/fig2) ──
# Load LongBench directly, same filter (LENGTH_MIN=2000, LENGTH_MAX=6000), same
# seed (42) + random.sample(pool, 10).  We take index 0 for each task, so the
# specific prompt Fig 3 visualises is a member of the same sample pool that
# fig1/fig2 aggregate over.
SEED = 42
LENGTH_MIN, LENGTH_MAX = 2000, 6000
LONGBENCH_DIR = "/home/smp9898/A2SF/datasets/longbench"
TASK_TO_DATASET = {
    "Single-doc QA": "qasper",
    "Multi-doc QA": "hotpotqa",
    "Summarization": "gov_report",
    "Few Shot": "samsum",
}
TASK_ORDER = list(TASK_TO_DATASET.keys())


def load_one_per_task():
    """Match the prompt selection used in experiments/temporal_bias/optimal.py.

    Same LongBench source + LENGTH_MIN/MAX filter + seed=42 + random.sample(10)[0].
    """
    import random as _random
    import os as _os
    pools = {}
    for fname in _os.listdir(LONGBENCH_DIR):
        with open(_os.path.join(LONGBENCH_DIR, fname)) as f:
            for line in f:
                item = json.loads(line)
                length = item.get("length", 0)
                if LENGTH_MIN <= length <= LENGTH_MAX:
                    pools.setdefault(item["dataset"], []).append(item["input_prompt"])

    per_task = {}
    for task_name, dset in TASK_TO_DATASET.items():
        if dset not in pools:
            continue
        _random.seed(SEED)
        sample = _random.sample(pools[dset], min(10, len(pools[dset])))
        per_task[task_name] = sample[0]
    return per_task


def compute_first_layer_attention(model, tokenizer, text, device):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    first = model.model.layers[0]
    q_proj = first.self_attn.q_proj
    k_proj = first.self_attn.k_proj
    ln = first.input_layernorm
    rot = first.self_attn.rotary_emb
    emb = model.model.embed_tokens
    input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=4096).input_ids.to(device)
    T = input_ids.size(1)
    with torch.no_grad():
        h = emb(input_ids); h = ln(h).to(q_proj.weight.dtype)
        H = model.config.num_attention_heads
        KV = model.config.num_key_value_heads
        G = H // KV
        D = model.config.hidden_size // H
        q = q_proj(h).view(1, T, H, D).transpose(1, 2)
        k = k_proj(h).view(1, T, KV, D).transpose(1, 2)
        pos = torch.arange(T, device=device).unsqueeze(0)
        cos, sin = rot(k, pos)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, G)
        scores = (q @ k.transpose(-2, -1)) / (D ** 0.5)
        mask = torch.arange(T, device=device)[None] > torch.arange(T, device=device)[:, None]
        scores = scores.masked_fill(mask[None, None], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
    return attn[0].float().cpu().numpy(), T


def selection_mask_per_head(attn, alpha, budget):
    """Per-head top-k selection (matches actual A2SF behavior — each KV head
    keeps its own top-budget cache). Returns shape (H, T) binary masks."""
    H, T, _ = attn.shape
    q = np.arange(T)
    w = 1.0 / (1.0 + np.exp(-alpha * (q - (T - 1))))              # (T,)
    # score[h, k] = Σ_q w(q) · attn[h, q, k]
    score = (w[None, :, None] * attn).sum(axis=1)                  # (H, T)
    masks = np.zeros((H, T), dtype=bool)
    for h in range(H):
        idx = np.argpartition(-score[h], budget)[:budget]
        masks[h, idx] = True
    return masks                                                   # (H, T)


def jaccard_per_head_mean(masks_a, masks_b):
    """Mean Jaccard across heads between two per-head selection sets (H, T)."""
    H = masks_a.shape[0]
    js = np.empty(H)
    for h in range(H):
        inter = np.logical_and(masks_a[h], masks_b[h]).sum()
        union = np.logical_or(masks_a[h], masks_b[h]).sum()
        js[h] = inter / max(1, union)
    return float(js.mean())


def jaccard_matrix_from_perhead(masks_list):
    """masks_list: list of (H, T) per-head masks. Returns matrix of mean
    per-head Jaccard across each pair."""
    n = len(masks_list)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            J[i, j] = jaccard_per_head_mean(masks_list[i], masks_list[j])
    return J


def resample_to_length(mask, N):
    T = len(mask)
    out = np.zeros(N, dtype=bool)
    bins = np.linspace(0, T, N + 1).astype(int)
    for i in range(N):
        lo, hi = bins[i], max(bins[i] + 1, bins[i + 1])
        out[i] = mask[lo:hi].any()
    return out


def resample_per_head(masks, N):
    """masks: (H, T) → (H, N)  position-bin resampling per head."""
    H = masks.shape[0]
    out = np.zeros((H, N), dtype=bool)
    for h in range(H):
        out[h] = resample_to_length(masks[h], N)
    return out


def main():
    out_dir = "/home/smp9898/A2SF/experiments/paper_figures"
    os.makedirs(out_dir, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL_PATH} …")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16,
                                                 device_map={"": device})
    model.eval()

    prompts = load_one_per_task()
    task_for_a = "Multi-doc QA"

    # (a) alpha-sweep selections, same prompt — PER-HEAD Jaccard, mean over heads
    attn_a, T_a = compute_first_layer_attention(model, tok, prompts[task_for_a], device)
    masks_a_ph = [selection_mask_per_head(attn_a, a, BUDGET) for a in ALPHA_SET]
    J_alpha = jaccard_matrix_from_perhead(masks_a_ph)
    print(f"(a) {task_for_a}: T={T_a}; α-J mean off-diag = "
          f"{(J_alpha.sum() - np.trace(J_alpha)) / (len(ALPHA_SET)**2 - len(ALPHA_SET)):.3f}")

    # (b) prompt-sweep at fixed α — per-head masks resampled to common length
    tasks_b = [t for t in TASK_ORDER if t in prompts]
    Nres = 256
    masks_b_ph = []
    for t in tasks_b:
        attn_b, T_b = compute_first_layer_attention(model, tok, prompts[t], device)
        m_ph = selection_mask_per_head(attn_b, FIXED_ALPHA, BUDGET)       # (H, T_b)
        m_ph_resampled = resample_per_head(m_ph, Nres)                      # (H, Nres)
        masks_b_ph.append(m_ph_resampled)
    J_prompt = jaccard_matrix_from_perhead(masks_b_ph)
    print(f"(b) prompt-J mean off-diag = "
          f"{(J_prompt.sum() - np.trace(J_prompt)) / (len(tasks_b)**2 - len(tasks_b)):.3f}")

    # ── Build density curves — aggregate over heads (position histogram of union) ──
    NBINS = 30
    # (a) density of selected positions for each α (same prompt)
    density_a = []
    for mph in masks_a_ph:                                       # (H, T_a)
        # Position count = sum across heads (how many heads selected each position)
        counts = mph.sum(axis=0).astype(float)                   # (T_a,)
        positions = np.arange(T_a) / max(T_a - 1, 1)
        hist, edges = np.histogram(positions, bins=NBINS, range=(0, 1), weights=counts)
        density_a.append(hist / (hist.sum() + 1e-12))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # (b) density per prompt at fixed α (aggregate heads similarly)
    density_b = []
    raw_masks_b = []
    for t in tasks_b:
        attn_b, T_b = compute_first_layer_attention(model, tok, prompts[t], device)
        m_ph = selection_mask_per_head(attn_b, FIXED_ALPHA, BUDGET)
        raw_masks_b.append((t, m_ph, T_b))
        counts = m_ph.sum(axis=0).astype(float)
        positions = np.arange(T_b) / max(T_b - 1, 1)
        hist, _ = np.histogram(positions, bins=NBINS, range=(0, 1), weights=counts)
        density_b.append(hist / (hist.sum() + 1e-12))

    # ── Plot: 1×4 horizontal layout with EQUAL panel widths.
    fig = plt.figure(figsize=(15, 3.5))
    gs = fig.add_gridspec(1, 4, wspace=0.22, left=0.05, right=0.98,
                          bottom=0.20, top=0.88)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_ja = fig.add_subplot(gs[0, 2])
    ax_jb = fig.add_subplot(gs[0, 3])

    # (a) density vs α
    cmap_a = plt.get_cmap("plasma")
    for i, (alpha, dens) in enumerate(zip(ALPHA_SET, density_a)):
        col = cmap_a(i / max(1, len(ALPHA_SET) - 1))
        ax_a.plot(centers, dens, color=col, lw=2.2, label=rf"$\alpha={alpha:g}$")
    ax_a.set_xlabel("normalized key position")
    ax_a.set_ylabel("fraction of selected keys")
    ax_a.set_title(r"(a) varying $\alpha$")
    ax_a.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xlim(0, 1)

    # (b) density per prompt at fixed α
    cmap_b = plt.get_cmap("tab10").colors
    for i, ((t, _, _), dens) in enumerate(zip(raw_masks_b, density_b)):
        ax_b.plot(centers, dens, color=cmap_b[i], lw=2.2, label=t)
    ax_b.set_xlabel("normalized key position")
    ax_b.set_title(r"(b) varying prompt")
    ax_b.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax_b.grid(True, alpha=0.3)
    ax_b.set_xlim(0, 1)

    # (c) Jaccard α×α
    im_a = ax_ja.imshow(J_alpha, cmap="Blues", vmin=0, vmax=1)
    ax_ja.set_xticks(range(len(ALPHA_SET)))
    ax_ja.set_yticks(range(len(ALPHA_SET)))
    ax_ja.set_xticklabels([f"{a:g}" for a in ALPHA_SET])
    ax_ja.set_yticklabels([f"{a:g}" for a in ALPHA_SET])
    ax_ja.set_xlabel(r"$\alpha$")
    ax_ja.set_ylabel(r"$\alpha$")
    ax_ja.set_title(r"(c) Jaccard  (varying $\alpha$)")
    for i in range(len(ALPHA_SET)):
        for j in range(len(ALPHA_SET)):
            ax_ja.text(j, i, f"{J_alpha[i, j]:.2f}", ha="center", va="center",
                       color="white" if J_alpha[i, j] > 0.55 else "black", fontsize=10)

    # (d) Jaccard prompt×prompt
    im_b = ax_jb.imshow(J_prompt, cmap="Blues", vmin=0, vmax=1)
    ax_jb.set_xticks(range(len(tasks_b)))
    ax_jb.set_yticks(range(len(tasks_b)))
    ax_jb.set_xticklabels(tasks_b, rotation=25, ha="right")
    ax_jb.set_yticklabels(tasks_b)
    ax_jb.set_title(r"(d) Jaccard  (varying prompt)")
    for i in range(len(tasks_b)):
        for j in range(len(tasks_b)):
            ax_jb.text(j, i, f"{J_prompt[i, j]:.2f}", ha="center", va="center",
                       color="white" if J_prompt[i, j] > 0.55 else "black", fontsize=10)

    # No colorbar — text annotations carry the values.

    fig.savefig(os.path.join(out_dir, "fig3_selection_varies.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig3_selection_varies.png"), bbox_inches="tight")
    print(f"saved → {out_dir}/fig3_selection_varies.{{pdf,png}}")


if __name__ == "__main__":
    main()
