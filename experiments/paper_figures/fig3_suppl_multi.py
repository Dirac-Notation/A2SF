"""Supplementary observation: panel (c) for each of 4 prompts (α×α Jaccard per task)
and panel (d) for each of several α values (prompt×prompt Jaccard per α).

Saved separately as `fig3_suppl_c_perprompt.{pdf,png}` and `fig3_suppl_d_peralpha.{pdf,png}`.
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
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})


MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
BUDGET = 128
ALPHA_SET = [0.0, 0.001, 0.01, 0.1, 1.0]
ALPHA_SWEEP_FOR_D = [0.0, 0.001, 0.01, 0.1, 1.0]    # one prompt×prompt matrix per α
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
    a = attn.sum(axis=0)
    q = np.arange(T)
    w = 1.0 / (1.0 + np.exp(-alpha * (q - (T - 1))))
    score = (w[:, None] * a).sum(axis=0)
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


def resample(mask, N):
    T = len(mask)
    out = np.zeros(N, dtype=bool)
    bins = np.linspace(0, T, N + 1).astype(int)
    for i in range(N):
        lo, hi = bins[i], max(bins[i] + 1, bins[i + 1])
        out[i] = mask[lo:hi].any()
    return out


def annotate(ax, M, fmt="{:.2f}", thresh=0.55, fontsize=9):
    n = M.shape[0]
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    color="white" if v > thresh else "black", fontsize=fontsize)


def main():
    out_dir = "/home/smp9898/A2SF/experiments/paper_figures"

    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL_PATH} …")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16,
                                                 device_map={"": device})
    model.eval()

    prompts = load_one_per_task()
    tasks_present = [t for t in TASK_ORDER if t in prompts]

    # ── Precompute attention once per prompt ──
    attn_cache = {}
    for t in tasks_present:
        attn, T = compute_first_layer_attention(model, tok, prompts[t], device)
        attn_cache[t] = (attn, T)
        print(f"attn computed: {t}  T={T}")

    # ── (c) per-prompt α×α Jaccard (one subplot per prompt) ──
    fig, axes = plt.subplots(1, len(tasks_present), figsize=(15, 3.6))
    for ax, t in zip(axes, tasks_present):
        attn, T = attn_cache[t]
        masks = [selection_mask(attn, a, BUDGET) for a in ALPHA_SET]
        J = jaccard_matrix(masks)
        im = ax.imshow(J, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(ALPHA_SET)))
        ax.set_yticks(range(len(ALPHA_SET)))
        ax.set_xticklabels([f"{a:g}" for a in ALPHA_SET])
        ax.set_yticklabels([f"{a:g}" for a in ALPHA_SET])
        ax.set_xlabel(r"$\alpha$")
        if ax is axes[0]:
            ax.set_ylabel(r"$\alpha$")
        ax.set_title(f"{t}")
        annotate(ax, J)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_suppl_c_perprompt.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig3_suppl_c_perprompt.png"), bbox_inches="tight")

    # ── (d) per-α prompt×prompt Jaccard (one subplot per α) ──
    fig, axes = plt.subplots(1, len(ALPHA_SWEEP_FOR_D), figsize=(16, 3.6))
    Nres = 256
    for ax, alpha in zip(axes, ALPHA_SWEEP_FOR_D):
        masks = []
        for t in tasks_present:
            attn, T = attn_cache[t]
            m = selection_mask(attn, alpha, BUDGET)
            masks.append(resample(m, Nres))
        J = jaccard_matrix(masks)
        im = ax.imshow(J, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(tasks_present)))
        ax.set_yticks(range(len(tasks_present)))
        short = [t.replace("Single-doc QA", "SDoc QA").replace("Multi-doc QA", "MDoc QA")
                  .replace("Summarization", "Summ").replace("Few Shot", "FewShot")
                 for t in tasks_present]
        ax.set_xticklabels(short, rotation=30, ha="right")
        ax.set_yticklabels(short if ax is axes[0] else [""] * len(short))
        ax.set_title(rf"$\alpha={alpha:g}$")
        annotate(ax, J)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_suppl_d_peralpha.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig3_suppl_d_peralpha.png"), bbox_inches="tight")
    print(f"\nsaved → {out_dir}/fig3_suppl_c_perprompt.{{pdf,png}}")
    print(f"saved → {out_dir}/fig3_suppl_d_peralpha.{{pdf,png}}")


if __name__ == "__main__":
    main()
