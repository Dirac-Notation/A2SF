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

    # ─── Plot ───
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 4.2),
                                      gridspec_kw={"width_ratios": [1.0, 1.0]})

    # Panel (a) heatmap — rows: α, cols: position (normalized)
    cmap = ListedColormap(["white", "tab:blue"])
    ax_a.imshow(panel_a, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                extent=[0, 1, len(ALPHA_SWEEP), 0])
    ax_a.set_yticks(np.arange(len(ALPHA_SWEEP)) + 0.5)
    ax_a.set_yticklabels([f"{a:g}" for a in ALPHA_SWEEP])
    ax_a.set_xlabel("normalized key position  (0 = oldest, 1 = most recent)")
    ax_a.set_ylabel(r"coefficient  $\alpha$")
    ax_a.set_title(f"(a) one prompt ({panel_a_prompt_task}), varying $\\alpha$")

    # Panel (b) heatmap — rows: prompt, cols: position (normalized)
    # Need to handle different T for each prompt → resample to common N bins
    N = 256
    panel_b_plot = np.zeros((len(panel_b_data), N), dtype=float)
    for i, (_, mask, T) in enumerate(panel_b_data):
        # bin N bins
        bins = np.linspace(0, T, N + 1).astype(int)
        for j in range(N):
            lo, hi = bins[j], max(bins[j] + 1, bins[j + 1])
            panel_b_plot[i, j] = mask[lo:hi].mean() if hi > lo else 0.0
    ax_b.imshow(panel_b_plot, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                extent=[0, 1, len(panel_b_data), 0])
    ax_b.set_yticks(np.arange(len(panel_b_data)) + 0.5)
    ax_b.set_yticklabels([t for t, _, _ in panel_b_data])
    ax_b.set_xlabel("normalized key position  (0 = oldest, 1 = most recent)")
    ax_b.set_ylabel("prompt")
    ax_b.set_title(f"(b) fixed $\\alpha={FIXED_ALPHA}$, varying prompt")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_selection_varies.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig3_selection_varies.png"), bbox_inches="tight")
    print(f"\nsaved → {out_dir}/fig3_selection_varies.{{pdf,png}}")


if __name__ == "__main__":
    main()
