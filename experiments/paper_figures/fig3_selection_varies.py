"""Figure 3 (Observation 3): Token selection pattern depends on (coefficient, prompt).

Two panels, both directly showing that "which positions survive top-k depends on how
we weight queries via α, and on where the prompt places its important content".

  (a) Fix one prompt. Sweep α ∈ {0, 0.001, 0.01, 0.1, 0.5, 1.0}. Plot a binary heat-map
      (rows = α, columns = key position).  Cell = 1 if that position is in the top-B
      selected set.  The changing pattern across rows is the observation.

  (b) Fix α = 0.01 (representative soft decay).  Run on 4 prompts (different tasks).
      Plot same binary heat-map (rows = prompt, columns = normalized key position).
      Cells differ because each prompt's important content lives in different positions.

Data collection: load LLaMA-3.2-1B first-layer attention for chosen prompts (small
compute budget, ~1 GPU on server 18).
"""
import os
import json
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap

sys.path.append("/home/smp9898/A2SF")


rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "figure.dpi": 150,
})


MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
BUDGET = 128
# α values to sweep in panel (a) — spaced log-scale
ALPHA_SWEEP = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
FIXED_ALPHA = 0.01    # for panel (b)

DATA_FILE = "/home/smp9898/A2SF/RL/training/1b_fix_exp_aug4/training_data_backup.jsonl"
TASKS_TO_KEEP = ["Single-doc QA", "Multi-doc QA", "Summarization", "Few Shot"]


def load_prompts_one_per_task():
    """Pick a middling-length prompt from each of the target tasks."""
    per_task = {}
    with open(DATA_FILE) as f:
        for line in f:
            r = json.loads(line)
            t = str(r.get("task_type"))
            if t not in TASKS_TO_KEEP:
                continue
            if t in per_task:
                continue
            text = r.get("input_prompt", "")
            if len(text) < 2000 or len(text) > 12000:
                continue
            per_task[t] = text
            if len(per_task) == len(TASKS_TO_KEEP):
                break
    return per_task


def compute_first_layer_attention(model, tokenizer, text, device):
    """Return attn_probs shape (H, T, T) using only first-layer weights."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    first = model.model.layers[0]
    q_proj = first.self_attn.q_proj
    k_proj = first.self_attn.k_proj
    ln = first.input_layernorm
    rot = first.self_attn.rotary_emb
    emb = model.model.embed_tokens

    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(device)
    T = input_ids.size(1)
    with torch.no_grad():
        h = emb(input_ids)
        h = ln(h)
        h = h.to(dtype=q_proj.weight.dtype)

        num_heads = model.config.num_attention_heads
        num_kv = model.config.num_key_value_heads
        groups = num_heads // num_kv
        head_dim = model.config.hidden_size // num_heads

        q = q_proj(h).view(1, T, num_heads, head_dim).transpose(1, 2)     # (1, H, T, D)
        k = k_proj(h).view(1, T, num_kv, head_dim).transpose(1, 2)
        pos = torch.arange(T, device=device).unsqueeze(0)
        cos, sin = rot(k, pos)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, groups)                                          # (1, H, T, D)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        causal = torch.arange(T, device=device).unsqueeze(0) > torch.arange(T, device=device).unsqueeze(1)
        attn_scores = attn_scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = torch.softmax(attn_scores, dim=-1)                         # (1, H, T, T)
    return attn[0].float().cpu().numpy(), T                              # (H, T, T), T


def score_topk(attn, alpha, budget, local_recent=16):
    """Compute accumulative score per key position and return top-B indices.

    score[k] = Σ_q w(q; α) · sum-over-heads(attn[q, k])
    with w(q; α) = σ(α * (q − N)) using b=0 (matches A2SF sigmoid policy).
    Also pins the most-recent `local_recent` positions (mandatory keeps).
    """
    H, T, _ = attn.shape
    # Sum heads → (T, T): attn[q, k]
    a = attn.sum(axis=0)                                                  # (T, T)
    # Per-query weight
    q_idx = np.arange(T)
    w = 1.0 / (1.0 + np.exp(-alpha * (q_idx - (T - 1))))                  # σ(α(q-N))
    score = (w[:, None] * a).sum(axis=0)                                  # (T,)
    # Enforce "local recent window" (not counted in budget; mimics A2SF behavior).
    # For simplicity here: just select top-B from all positions.
    top_idx = np.argsort(-score)[:budget]
    mask = np.zeros(T, dtype=bool)
    mask[top_idx] = True
    return mask


def main():
    out_dir = "/home/smp9898/A2SF/experiments/paper_figures"
    os.makedirs(out_dir, exist_ok=True)

    # ─── Load model ───
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL_PATH} on {device} …")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16,
                                                 device_map={"": device})
    model.eval()

    # ─── Pick prompts ───
    prompts = load_prompts_one_per_task()
    print(f"selected prompts: {[(t, len(p)) for t, p in prompts.items()]}")

    # ─── Panel (a): fix one prompt, vary α ───
    panel_a_prompt_task = "Multi-doc QA"
    print(f"\nPanel (a) prompt: task={panel_a_prompt_task}")
    attn_a, T_a = compute_first_layer_attention(model, tok, prompts[panel_a_prompt_task], device)
    panel_a = np.zeros((len(ALPHA_SWEEP), T_a), dtype=bool)
    for i, alpha in enumerate(ALPHA_SWEEP):
        panel_a[i] = score_topk(attn_a, alpha, BUDGET)
        print(f"  α={alpha:<6} selected={panel_a[i].sum()}/{T_a}")

    # ─── Panel (b): fix α, vary prompt ───
    tasks_order = ["Single-doc QA", "Multi-doc QA", "Summarization", "Few Shot"]
    tasks_avail = [t for t in tasks_order if t in prompts]
    panel_b_data = []
    for t in tasks_avail:
        attn_b, T_b = compute_first_layer_attention(model, tok, prompts[t], device)
        mask = score_topk(attn_b, FIXED_ALPHA, BUDGET)
        panel_b_data.append((t, mask, T_b))
        print(f"Panel (b) task={t} T={T_b} selected={mask.sum()}")

    # ─── Plot — density curves per row, then Jaccard heatmap to drive the point home ───
    fig = plt.figure(figsize=(15, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_ja = fig.add_subplot(gs[1, 0])
    ax_jb = fig.add_subplot(gs[1, 1])

    NBINS = 40
    cmap_a = plt.get_cmap("plasma")

    # Panel (a) density of selected positions per α
    for i, alpha in enumerate(ALPHA_SWEEP):
        mask = panel_a[i]
        pos = np.where(mask)[0] / (T_a - 1)                     # normalized [0, 1]
        hist, edges = np.histogram(pos, bins=NBINS, range=(0, 1))
        centers = 0.5 * (edges[:-1] + edges[1:])
        color = cmap_a(i / max(1, len(ALPHA_SWEEP) - 1))
        ax_a.plot(centers, hist / hist.sum(), color=color, lw=2.0,
                  label=rf"$\alpha={alpha:g}$")
    ax_a.set_xlabel("normalized key position  (0 = oldest, 1 = most recent)")
    ax_a.set_ylabel("fraction of selected keys")
    ax_a.set_title(f"(a) one prompt ({panel_a_prompt_task}), varying $\\alpha$")
    ax_a.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.95)
    ax_a.grid(True, alpha=0.3)

    # Panel (b) density per prompt at fixed α
    cmap_b = plt.get_cmap("tab10")
    for i, (task, mask, T) in enumerate(panel_b_data):
        pos = np.where(mask)[0] / (T - 1)
        hist, edges = np.histogram(pos, bins=NBINS, range=(0, 1))
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax_b.plot(centers, hist / hist.sum(), color=cmap_b(i), lw=2.0,
                  label=task)
    ax_b.set_xlabel("normalized key position  (0 = oldest, 1 = most recent)")
    ax_b.set_ylabel("fraction of selected keys")
    ax_b.set_title(f"(b) fixed $\\alpha={FIXED_ALPHA}$, varying prompt")
    ax_b.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax_b.grid(True, alpha=0.3)

    # Panel (c): Jaccard similarity between α-selections (same prompt)
    n_a = len(ALPHA_SWEEP)
    jaccard_a = np.zeros((n_a, n_a))
    for i in range(n_a):
        for j in range(n_a):
            inter = np.logical_and(panel_a[i], panel_a[j]).sum()
            union = np.logical_or(panel_a[i], panel_a[j]).sum()
            jaccard_a[i, j] = inter / max(1, union)
    im = ax_ja.imshow(jaccard_a, cmap="viridis", vmin=0, vmax=1)
    ax_ja.set_xticks(range(n_a))
    ax_ja.set_xticklabels([f"{a:g}" for a in ALPHA_SWEEP], fontsize=9)
    ax_ja.set_yticks(range(n_a))
    ax_ja.set_yticklabels([f"{a:g}" for a in ALPHA_SWEEP], fontsize=9)
    ax_ja.set_xlabel(r"$\alpha$")
    ax_ja.set_ylabel(r"$\alpha$")
    ax_ja.set_title("(c) Jaccard of selections across $\\alpha$ (same prompt)")
    for i in range(n_a):
        for j in range(n_a):
            ax_ja.text(j, i, f"{jaccard_a[i, j]:.2f}", ha="center", va="center",
                       color="white" if jaccard_a[i, j] < 0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=ax_ja, fraction=0.046, pad=0.04)

    # Panel (d): Jaccard between prompts at same α — since prompt lengths differ,
    # resample each mask to a common length first.
    Nres = 256
    resampled = []
    for _, mask, T in panel_b_data:
        bins = np.linspace(0, T, Nres + 1).astype(int)
        out = np.zeros(Nres, dtype=bool)
        for j in range(Nres):
            lo, hi = bins[j], max(bins[j] + 1, bins[j + 1])
            out[j] = mask[lo:hi].any()
        resampled.append(out)
    nb = len(resampled)
    jaccard_b = np.zeros((nb, nb))
    for i in range(nb):
        for j in range(nb):
            inter = np.logical_and(resampled[i], resampled[j]).sum()
            union = np.logical_or(resampled[i], resampled[j]).sum()
            jaccard_b[i, j] = inter / max(1, union)
    im2 = ax_jb.imshow(jaccard_b, cmap="viridis", vmin=0, vmax=1)
    labels_b = [t for t, _, _ in panel_b_data]
    ax_jb.set_xticks(range(nb))
    ax_jb.set_xticklabels(labels_b, rotation=30, ha="right", fontsize=9)
    ax_jb.set_yticks(range(nb))
    ax_jb.set_yticklabels(labels_b, fontsize=9)
    ax_jb.set_title(f"(d) Jaccard of selections across prompts (same $\\alpha={FIXED_ALPHA}$)")
    for i in range(nb):
        for j in range(nb):
            ax_jb.text(j, i, f"{jaccard_b[i, j]:.2f}", ha="center", va="center",
                       color="white" if jaccard_b[i, j] < 0.5 else "black", fontsize=9)
    plt.colorbar(im2, ax=ax_jb, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_selection_varies.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig3_selection_varies.png"), bbox_inches="tight")
    print(f"\nsaved → {out_dir}/fig3_selection_varies.{{pdf,png}}")


if __name__ == "__main__":
    main()
