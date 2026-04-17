"""
Temporal Bias Analysis for KV Cache Compression

Demonstrates that recent queries in the observation window contribute more
to identifying important KV pairs than distant queries.

Analyses:
  1. Block Hit Rate  – each query's independent predictive power (decay with distance)
  2. Optimal Coefficient – greedy per-query weight search with sigmoid fit
  3. Cumulative Similarity – diminishing returns from adding distant queries

Optimised for 8× 24GB VRAM GPUs (bf16):
  - Prefill uses SDPA (flash attention): O(N) memory, never materialises the full (S, S) matrix
  - Window attention (last 128 queries) is recomputed post-hoc from cached Q inputs + KV cache
  - Decode uses output_attentions=True which falls back to eager; attention is (H, 1, S) — tiny
  - Vectorised Jaccard via one-hot scatter
"""

import math
import torch
import json
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from collections import defaultdict

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

workpath = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Reproducibility ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Config ──
MAX_WINDOW = 128
TOKEN_BUDGET = 128
LOCAL_RATIO = 0.125
NUM_ITEMS = 10
MAX_SEQ_LEN = 32768
MODEL_NAME = "llama3"

TASK_GROUP = {
    "Few Shot": ["samsum"],
    "Single-doc QA": ["qasper"],
    "Multi-doc QA": ["hotpotqa"],
    "Summarization": ["gov_report"],
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "axes.linewidth": 1.2,
})


# ══════════════════════════════════════════════════════════════
# Attention Collector  (SDPA-safe: never requires eager prefill)
# ══════════════════════════════════════════════════════════════
class AttentionCollector:
    """Memory-efficient attention collector for long sequences.

    Prefill phase (SDPA, output_attentions=False):
      - Pre-hooks capture the hidden_states INPUT to each attention module
        for the last MAX_WINDOW positions only (~1 MB/layer).
      - After the forward pass, `compute_window_attention()` recomputes
        attention weights for those 128 queries against the full KV cache,
        one layer at a time (~630 MB peak per layer on that layer's GPU).

    Decode phase (output_attentions=True → eager fallback for seq=1):
      - Post-hooks accumulate attention-to-prefill scores on-the-fly.
    """

    def __init__(self, model, max_window):
        self.model = model
        self.max_window = max_window
        cfg = model.config
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", self.num_heads)
        self.group_size = self.num_heads // self.num_kv_heads
        self.head_dim = cfg.hidden_size // self.num_heads

        self._window_inputs = {}   # layer_idx → (1, W, D) on layer's device
        self._answer = None        # (L, H, S) on CPU
        self._prefill_len = 0
        self._is_prefill = True

        self._hooks = []
        for i, layer in enumerate(model.model.layers):
            h1 = layer.self_attn.register_forward_pre_hook(self._pre_hook(i))
            h2 = layer.self_attn.register_forward_hook(self._post_hook(i))
            self._hooks.extend([h1, h2])

    # ── public ──
    def reset(self, prefill_len):
        self._window_inputs.clear()
        self._answer = torch.zeros(self.num_layers, self.num_heads, prefill_len)
        self._prefill_len = prefill_len
        self._is_prefill = True

    def set_decode(self):
        self._is_prefill = False

    @property
    def answer_score(self):
        return self._answer

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── hooks ──
    def _pre_hook(self, layer_idx):
        """Prefill: save hidden_states for last MAX_WINDOW positions."""
        def hook(module, args):
            if not self._is_prefill:
                return
            hidden = args[0]  # (B, S, D)
            if hidden.size(1) <= 1:
                return
            w = min(self.max_window, hidden.size(1))
            self._window_inputs[layer_idx] = hidden[:, -w:, :].detach()
        return hook

    def _post_hook(self, layer_idx):
        """Decode: accumulate attention to prefill tokens, then discard weights."""
        def hook(module, inp, out):
            if self._is_prefill:
                return
            attn = out[1]
            if attn is not None and attn.dim() >= 3 and attn.size(2) == 1:
                self._answer[layer_idx] += (
                    attn[0, :, 0, : self._prefill_len].detach().cpu()
                )
                return (out[0], None, out[2])
        return hook

    # ── post-hoc window attention ──
    def compute_window_attention(self, past_kv):
        """Recompute attention weights for the window queries.

        Uses q_proj + rotary on stored hidden_states, and K from KV cache.
        Processes one layer at a time → peak ~630 MB on that layer's GPU.
        Returns: (L, H, W, S) on CPU.
        """
        S = self._prefill_len
        maps = []

        for i in range(self.num_layers):
            attn_mod = self.model.model.layers[i].self_attn
            hidden = self._window_inputs[i]          # (1, W, D) on layer device
            device = hidden.device
            W = hidden.size(1)

            # ── Q projection + reshape ──
            q = attn_mod.q_proj(hidden)              # (1, W, H*D)
            q = q.view(1, W, self.num_heads, self.head_dim).transpose(1, 2)
            #   → (1, H, W, D)

            # ── Rotary embedding for Q ──
            pos_ids = torch.arange(S - W, S, device=device).unsqueeze(0)
            cos, sin = attn_mod.rotary_emb(q, pos_ids)
            q, _ = apply_rotary_pos_emb(q, q, cos, sin)   # only q matters

            # ── K from cache (already rotary-embedded) ──
            k = past_kv[i][0]                        # (1, nkv, S, D)

            # ── Attention scores (grouped matmul to avoid expanding K) ──
            q_g = q.view(1, self.num_kv_heads, self.group_size, W, self.head_dim)
            k_t = k.unsqueeze(2).transpose(-1, -2)   # (1, nkv, 1, D, S)
            scores = torch.matmul(q_g, k_t) / math.sqrt(self.head_dim)
            scores = scores.view(1, self.num_heads, W, S)

            # ── Causal mask ──
            key_pos = torch.arange(S, device=device)
            qry_pos = torch.arange(S - W, S, device=device)
            mask = key_pos.unsqueeze(0) <= qry_pos.unsqueeze(1)   # (W, S)
            scores.masked_fill_(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # ── Softmax ──
            attn_w = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
            maps.append(attn_w[0].cpu())             # (H, W, S)

            del q, q_g, k_t, scores, attn_w, hidden, cos, sin
            self._window_inputs[i] = None
            torch.cuda.empty_cache()

        return torch.stack(maps)                      # (L, H, W, S)


# ══════════════════════════════════════════════════════════════
# Vectorised helpers
# ══════════════════════════════════════════════════════════════
def jaccard(a_idx, b_idx, seq_len):
    """One-hot scatter Jaccard, mean over layers & heads."""
    L, H, _ = a_idx.shape
    a = torch.zeros(L, H, seq_len, dtype=torch.bool)
    b = torch.zeros(L, H, seq_len, dtype=torch.bool)
    a.scatter_(2, a_idx, True)
    b.scatter_(2, b_idx, True)
    inter = (a & b).sum(2).float()
    union = (a | b).sum(2).float().clamp(min=1)
    return (inter / union).mean().item()


def gqa_topk(scores, k, num_kv_heads, group_size):
    """Top-k respecting GQA grouping."""
    if group_size <= 1:
        return scores.topk(k, dim=-1).indices
    *batch, _H, S = scores.shape
    g = scores.view(*batch, num_kv_heads, group_size, S).sum(dim=-2)
    idx = g.topk(k, dim=-1).indices
    return idx.repeat_interleave(group_size, dim=-2)


# ══════════════════════════════════════════════════════════════
# Analysis functions
# ══════════════════════════════════════════════════════════════
def analyze_block_hit_rate(prefill, answer_idx, budget, kv_h, gs, seq_len):
    """Independent hit rate per query position (index 0 = most recent)."""
    W = prefill.size(2)
    rates = []
    for b in range(W):
        score = prefill[:, :, W - 1 - b, :]
        idx = gqa_topk(score, budget, kv_h, gs)
        rates.append(jaccard(answer_idx, idx, seq_len))
    return np.array(rates)


def analyze_optimal_coefficient(prefill, answer_idx, budget, kv_h, gs, seq_len):
    """Greedy per-query coefficient search. Returns (coefficients, hit_rates)."""
    L, H, W, S = prefill.shape
    local_b = int(budget * LOCAL_RATIO)
    sel_b = budget - local_b
    test_coeffs = torch.arange(0.0, 1.1, 0.1)

    accumulated = torch.zeros(L, H, S)
    opt_coeffs, hit_rates = [], []

    for b in range(W):
        block = prefill[:, :, W - 1 - b, :]
        if b == 0:
            accumulated += block
            idx = gqa_topk(accumulated, budget, kv_h, gs)
            hr = jaccard(answer_idx, idx, seq_len)
            opt_coeffs.append(1.0)
            hit_rates.append(hr)
        else:
            best_c, best_hr = 0.0, hit_rates[-1]
            for c in test_coeffs:
                tmp = accumulated + c * block
                tmp[:, :, -local_b:] = tmp.max()
                idx = gqa_topk(tmp, sel_b, kv_h, gs)
                hr = jaccard(answer_idx, idx, seq_len)
                if hr > best_hr:
                    best_hr = hr
                    best_c = c.item()
            accumulated += best_c * block
            opt_coeffs.append(best_c)
            hit_rates.append(best_hr)

    return np.array(opt_coeffs), np.array(hit_rates)


def analyze_cumulative(prefill, answer_idx, budget, kv_h, gs, seq_len):
    """Accumulate queries (coeff=1) from most-recent outward."""
    L, H, W, S = prefill.shape
    accumulated = torch.zeros(L, H, S)
    sims = []
    for b in range(W):
        accumulated += prefill[:, :, W - 1 - b, :]
        idx = gqa_topk(accumulated, budget, kv_h, gs)
        sims.append(jaccard(answer_idx, idx, seq_len))
    return np.array(sims)


# ══════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════
def _sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(a * (x - b)))


def plot_dataset(name, task, block_rates, opt_coeffs, cum_sims, save_dir):
    """Aggregate plot for one dataset: mean +/- std with individual traces."""
    n = len(block_rates)
    W = min(len(r) for r in block_rates)
    br = np.array([r[:W] for r in block_rates])
    oc = np.array([c[:W] for c in opt_coeffs])
    cs = np.array([s[:W] for s in cum_sims])
    x = np.arange(W)

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # 1 ── Block hit rate
    ax = axes[0]
    for i in range(n):
        ax.plot(x, br[i], alpha=0.15, color="steelblue", linewidth=0.8)
    m, s = br.mean(0), br.std(0)
    ax.plot(x, m, color="steelblue", linewidth=2.5)
    ax.fill_between(x, m - s, m + s, alpha=0.18, color="steelblue")
    ax.set_xlabel("Query distance from end")
    ax.set_ylabel("Hit Rate")
    ax.set_title("Per-Query Hit Rate")
    ax.grid(True, alpha=0.3)

    # 2 ── Optimal coefficient + sigmoid fit
    ax = axes[1]
    for i in range(n):
        ax.plot(x, oc[i], alpha=0.15, color="coral", linewidth=0.8)
    m_c = oc.mean(0)
    ax.plot(x, m_c, color="coral", linewidth=2.5, label="Mean")
    try:
        popt, _ = curve_fit(
            _sigmoid, x.astype(float), m_c,
            p0=[0.5, 16.0], bounds=((0, 0), (np.inf, np.inf)), maxfev=5000,
        )
        ax.plot(x, _sigmoid(x.astype(float), *popt), "k--", linewidth=2,
                label=f"Sigmoid (a={popt[0]:.2f}, b={popt[1]:.1f})")
        ax.legend(loc="upper right")
    except RuntimeError:
        pass
    ax.set_xlabel("Query distance from end")
    ax.set_ylabel("Optimal Coefficient")
    ax.set_title("Optimal Query Weight")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 3 ── Cumulative similarity
    ax = axes[2]
    for i in range(n):
        ax.plot(x, cs[i], alpha=0.15, color="forestgreen", linewidth=0.8)
    m_s, s_s = cs.mean(0), cs.std(0)
    ax.plot(x, m_s, color="forestgreen", linewidth=2.5)
    ax.fill_between(x, m_s - s_s, m_s + s_s, alpha=0.18, color="forestgreen")
    ax.set_xlabel("Number of accumulated queries")
    ax.set_ylabel("Similarity")
    ax.set_title("Cumulative Similarity")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{task}: {name} (n={n})", fontsize=24, fontweight="bold")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_summary(all_results, save_dir):
    """All datasets' mean optimal-coefficient decay on one plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_results)))

    for (name, task, oc_list), color in zip(all_results, colors):
        W = min(len(c) for c in oc_list)
        oc = np.array([c[:W] for c in oc_list])
        m = oc.mean(0)
        x = np.arange(W)
        ax.plot(x, m, linewidth=2.5, color=color, label=f"{name} ({task})")
        try:
            popt, _ = curve_fit(
                _sigmoid, x.astype(float), m,
                p0=[0.5, 16.0], bounds=((0, 0), (np.inf, np.inf)), maxfev=5000,
            )
            ax.plot(x, _sigmoid(x.astype(float), *popt), "--", color=color,
                    linewidth=1.5, alpha=0.7)
        except RuntimeError:
            pass

    ax.set_xlabel("Query distance from end")
    ax.set_ylabel("Optimal Coefficient")
    ax.set_title("Temporal Bias: Optimal Coefficient Decay", fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "summary.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved summary: {path}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    # ── Load configs ──
    with open(os.path.join(root_path, "config", "model2path.json")) as f:
        model2path = json.load(f)
    with open(os.path.join(root_path, "config", "dataset2maxlen.json")) as f:
        dataset2maxlen = json.load(f)

    model_path = model2path[MODEL_NAME]

    # ── Load & filter LongBench data ──
    longbench_dir = os.path.join(root_path, "datasets", "longbench")
    dataset_prompts = defaultdict(list)
    for fname in os.listdir(longbench_dir):
        with open(os.path.join(longbench_dir, fname)) as f:
            for line in f:
                item = json.loads(line)
                length = item.get("length", 0)
                if 2000 <= length <= 6000:
                    dataset_prompts[item["dataset"]].append(item["input_prompt"])

    dataset2task = {}
    selected = {}
    for task_name, datasets in TASK_GROUP.items():
        for d in datasets:
            dataset2task[d] = task_name
            if d in dataset_prompts:
                pool = dataset_prompts[d]
                selected[d] = random.sample(pool, min(NUM_ITEMS, len(pool)))

    # ── Load model (SDPA for memory-efficient prefill) ──
    print(f"Loading {model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()

    collector = AttentionCollector(model, MAX_WINDOW)
    plot_dir = os.path.join(workpath, "plots")
    all_results = []

    # ── Process each dataset ──
    for dataset_name, prompts in selected.items():
        task_name = dataset2task[dataset_name]
        print(f"\n{'=' * 60}")
        print(f"  {task_name}: {dataset_name}  ({len(prompts)} samples)")
        print(f"{'=' * 60}")

        all_br, all_oc, all_cs = [], [], []

        for idx, prompt in enumerate(prompts):
            print(f"\n  [{idx + 1}/{len(prompts)}] ", end="", flush=True)

            # Tokenise & truncate
            enc = tokenizer(f"[INST]{prompt}[/INST]", return_tensors="pt")
            input_ids = enc.input_ids.to(model.device)
            if input_ids.size(1) > MAX_SEQ_LEN:
                half = MAX_SEQ_LEN // 2
                input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)
            seq_len = input_ids.size(1)
            max_new = dataset2maxlen.get(dataset_name, 512)

            collector.reset(seq_len)

            with torch.no_grad():
                # ── Prefill (SDPA, no attention maps) ──
                out = model(input_ids, use_cache=True)
                past_kv = out.past_key_values
                next_tok = out.logits[:, -1:].argmax(dim=-1)
                del out
                torch.cuda.empty_cache()

                # ── Recompute window attention from stored Q inputs + KV cache ──
                prefill_attn = collector.compute_window_attention(past_kv)

                # ── Decode (output_attentions=True → eager fallback, tiny matrix) ──
                collector.set_decode()
                gen_len = 0
                for _ in range(max_new):
                    out = model(
                        next_tok,
                        past_key_values=past_kv,
                        use_cache=True,
                        output_attentions=True,
                    )
                    past_kv = out.past_key_values
                    next_tok = out.logits[:, -1:].argmax(dim=-1)
                    gen_len += 1
                    if next_tok.item() == tokenizer.eos_token_id:
                        del out
                        break
                    del out

            del past_kv
            torch.cuda.empty_cache()
            print(f"prefill={seq_len}  gen={gen_len}", end="  ", flush=True)

            # ── Ground truth: top-k from accumulated decode attention ──
            answer_idx = gqa_topk(
                collector.answer_score, TOKEN_BUDGET,
                collector.num_kv_heads, collector.group_size,
            )

            # ── Analyses ──
            kv_h, gs = collector.num_kv_heads, collector.group_size
            br = analyze_block_hit_rate(prefill_attn, answer_idx, TOKEN_BUDGET, kv_h, gs, seq_len)
            oc, hr = analyze_optimal_coefficient(prefill_attn, answer_idx, TOKEN_BUDGET, kv_h, gs, seq_len)
            cs = analyze_cumulative(prefill_attn, answer_idx, TOKEN_BUDGET, kv_h, gs, seq_len)

            all_br.append(br)
            all_oc.append(oc)
            all_cs.append(cs)
            print(f"HR[0]={br[0]:.3f}  coeff[-1]={oc[-1]:.1f}")

        # ── Per-dataset plot ──
        ds_dir = os.path.join(plot_dir, task_name.replace(" ", "_"))
        plot_dataset(dataset_name, task_name, all_br, all_oc, all_cs, ds_dir)
        all_results.append((dataset_name, task_name, all_oc))

    # ── Summary ──
    plot_summary(all_results, plot_dir)

    collector.remove_hooks()
    print("\n>>> All done.")


if __name__ == "__main__":
    main()
