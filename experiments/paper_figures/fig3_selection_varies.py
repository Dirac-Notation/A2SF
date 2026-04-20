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
DATA_FILE = "/home/smp9898/A2SF/RL/training/1b_fix_exp_aug4/training_data_backup.jsonl"
TASK_ORDER = ["Single-doc QA", "Multi-doc QA", "Summarization", "Few Shot"]


def load_one_per_task():
    per_task = {}
    with open(DATA_FILE) as f:
        for line in f:
            r = json.loads(line)
            t = str(r.get("task_type"))
            if t not in TASK_ORDER or t in per_task:
                continue
            txt = r.get("input_prompt", "")
            if 2000 <= len(txt) <= 12000:
                per_task[t] = txt
            if len(per_task) == len(TASK_ORDER):
                break
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


def selection_mask(attn, alpha, budget):
    H, T, _ = attn.shape
    a = attn.sum(axis=0)                               # (T, T) summed across heads
    q = np.arange(T)
    w = 1.0 / (1.0 + np.exp(-alpha * (q - (T - 1))))
    score = (w[:, None] * a).sum(axis=0)               # (T,)
    idx = np.argpartition(-score, budget)[:budget]
    mask = np.zeros(T, dtype=bool); mask[idx] = True
    return mask


def jaccard_matrix(masks):
    n = len(masks)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = np.logical_and(masks[i], masks[j]).sum()
            union = np.logical_or(masks[i], masks[j]).sum()
            J[i, j] = inter / max(1, union)
    return J


def resample_to_length(mask, N):
    T = len(mask)
    out = np.zeros(N, dtype=bool)
    bins = np.linspace(0, T, N + 1).astype(int)
    for i in range(N):
        lo, hi = bins[i], max(bins[i] + 1, bins[i + 1])
        out[i] = mask[lo:hi].any()
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

    # (a) alpha-sweep selections, same prompt
    attn_a, T_a = compute_first_layer_attention(model, tok, prompts[task_for_a], device)
    masks_a = [selection_mask(attn_a, a, BUDGET) for a in ALPHA_SET]
    J_alpha = jaccard_matrix(masks_a)
    print(f"(a) {task_for_a}: T={T_a}; α-J mean off-diag = "
          f"{(J_alpha.sum() - np.trace(J_alpha)) / (len(ALPHA_SET)**2 - len(ALPHA_SET)):.3f}")

    # (b) prompt-sweep at fixed α
    tasks_b = [t for t in TASK_ORDER if t in prompts]
    Ts_b, masks_b = [], []
    Nres = 256
    for t in tasks_b:
        attn_b, T_b = compute_first_layer_attention(model, tok, prompts[t], device)
        m = selection_mask(attn_b, FIXED_ALPHA, BUDGET)
        Ts_b.append(T_b)
        masks_b.append(resample_to_length(m, Nres))
    J_prompt = jaccard_matrix(masks_b)
    print(f"(b) prompt-J mean off-diag = "
          f"{(J_prompt.sum() - np.trace(J_prompt)) / (len(tasks_b)**2 - len(tasks_b)):.3f}")

    # ── Build density curves for line plots ──
    NBINS = 30
    # (a) density of selected positions for each α (same prompt)
    density_a = []
    for m in masks_a:
        pos = np.where(m)[0] / (T_a - 1)
        hist, edges = np.histogram(pos, bins=NBINS, range=(0, 1))
        density_a.append(hist / hist.sum())
    centers = 0.5 * (edges[:-1] + edges[1:])

    # (b) density of selected positions per prompt at fixed α (already masked)
    # Re-compute from raw masks to preserve prompt-specific T (not the resampled version).
    density_b = []
    raw_masks_b = []
    for t in tasks_b:
        attn_b, T_b = compute_first_layer_attention(model, tok, prompts[t], device)
        m = selection_mask(attn_b, FIXED_ALPHA, BUDGET)
        raw_masks_b.append((t, m, T_b))
        pos = np.where(m)[0] / (T_b - 1)
        hist, _ = np.histogram(pos, bins=NBINS, range=(0, 1))
        density_b.append(hist / hist.sum())

    # ── Plot: 2×2, top row line plots, bottom row Jaccard heatmaps ──
    fig = plt.figure(figsize=(12.5, 8.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15],
                          hspace=0.45, wspace=0.25)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_ja = fig.add_subplot(gs[1, 0])
    ax_jb = fig.add_subplot(gs[1, 1])

    # (a) density vs α
    cmap_a = plt.get_cmap("plasma")
    for i, (alpha, dens) in enumerate(zip(ALPHA_SET, density_a)):
        col = cmap_a(i / max(1, len(ALPHA_SET) - 1))
        ax_a.plot(centers, dens, color=col, lw=2.2, label=rf"$\alpha={alpha:g}$")
    ax_a.set_xlabel("normalized key position  (0 = oldest, 1 = most recent)")
    ax_a.set_ylabel("fraction of selected keys")
    ax_a.set_title(rf"(a) same prompt, varying $\alpha$")
    ax_a.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xlim(0, 1)

    # (b) density per prompt at fixed α
    cmap_b = plt.get_cmap("tab10").colors
    for i, ((t, _, _), dens) in enumerate(zip(raw_masks_b, density_b)):
        ax_b.plot(centers, dens, color=cmap_b[i], lw=2.2, label=t)
    ax_b.set_xlabel("normalized key position  (0 = oldest, 1 = most recent)")
    ax_b.set_ylabel("fraction of selected keys")
    ax_b.set_title(rf"(b) fixed $\alpha={FIXED_ALPHA}$, varying prompt")
    ax_b.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax_b.grid(True, alpha=0.3)
    ax_b.set_xlim(0, 1)

    # (c) Jaccard α×α
    im_a = ax_ja.imshow(J_alpha, cmap="Blues", vmin=0, vmax=1)
    ax_ja.set_xticks(range(len(ALPHA_SET)))
    ax_ja.set_yticks(range(len(ALPHA_SET)))
    ax_ja.set_xticklabels([f"{a:g}" for a in ALPHA_SET])
    ax_ja.set_yticklabels([f"{a:g}" for a in ALPHA_SET])
    ax_ja.set_xlabel(r"coefficient  $\alpha$")
    ax_ja.set_ylabel(r"coefficient  $\alpha$")
    ax_ja.set_title(r"(c) Jaccard of selected keys across $\alpha$")
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
    ax_jb.set_title(rf"(d) Jaccard of selected keys across prompts ($\alpha={FIXED_ALPHA}$)")
    for i in range(len(tasks_b)):
        for j in range(len(tasks_b)):
            ax_jb.text(j, i, f"{J_prompt[i, j]:.2f}", ha="center", va="center",
                       color="white" if J_prompt[i, j] > 0.55 else "black", fontsize=10)

    # Shared colorbar on the right for both heatmaps
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.40])
    fig.colorbar(im_b, cax=cbar_ax, label="Jaccard")

    fig.savefig(os.path.join(out_dir, "fig3_selection_varies.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig3_selection_varies.png"), bbox_inches="tight")
    print(f"saved → {out_dir}/fig3_selection_varies.{{pdf,png}}")


if __name__ == "__main__":
    main()
