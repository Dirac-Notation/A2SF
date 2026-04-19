"""
Comprehensive Temporal Bias Analysis for KV Cache Compression

Investigates *why* the optimal per-query contribution coefficient decays with
distance from the decode boundary. We test multiple hypotheses about what
makes recent queries dominant:

  H1 (Vector alignment)   : recent Q vectors are more similar to decoded Q vectors
  H2 (Pattern alignment)  : recent Qs produce attention patterns close to decode patterns
  H3 (Magnitude)          : recent Qs have larger norms, dominating accumulated scores
  H4 (Decisiveness)       : recent Qs have lower-entropy / sharper attention
  H5 (Mutual coherence)   : recent Qs form a tight cluster (high Q-Q & attention sim)
  H6 (Key decorrelation)  : K vectors decorrelate quickly so old Qs target a stale subspace
  H7 (Layer heterogeneity): some layers carry the bias more strongly than others

All metrics are aggregated per-dataset (mean ± std across 10 samples).

Hardware target: 8× 24GB GPUs, llama3-1b (~2GB) split across all GPUs to
minimise per-GPU pressure (other workloads sharing the GPUs).

Implementation notes:
  - Prefill uses SDPA (O(N) memory) — never materialises the full (S, S) attn matrix
  - The window's attention is recomputed post-hoc from cached Q-inputs + KV cache,
    one layer at a time
  - Decode uses output_attentions=True → eager fallback (tiny (H, 1, S) matrix)
  - Pre-hooks also accumulate post-rotary decoded Q vectors, enabling Q-decode similarity
"""

import math
import json
import os
import sys
import random
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
LENGTH_MIN = 2000
LENGTH_MAX = 6000
MODEL_NAME = "llama3-1b"
COEFF_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PLOT_SUBDIR = ""   # set to a non-empty subfolder name to isolate this run's outputs
PER_GPU_MAX_MEM = "2GiB"   # generous headroom for activations; we still pin layers manually below.

TASK_GROUP = {
    "Few Shot": ["samsum"],
    "Single-doc QA": ["qasper"],
    "Multi-doc QA": ["hotpotqa"],
    "Summarization": ["gov_report"],
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.2,
})


# ══════════════════════════════════════════════════════════════
# AttentionCollector  (extended: also accumulates decoded post-rotary Q)
# ══════════════════════════════════════════════════════════════
class AttentionCollector:
    """
    Prefill (SDPA, output_attentions=False):
      - pre-hook stores hidden_states[:, -W:, :] per layer (~1MB/layer)
    Decode (output_attentions=True → eager fallback):
      - pre-hook: computes & accumulates post-rotary decoded Q per layer
      - post-hook: accumulates attention scores to prefill tokens
    Post-hoc (after prefill forward):
      - compute_window_data() reconstructs window attention from stored hidden + KV cache
        and returns all raw tensors needed for downstream analyses
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
        self.hidden_size = cfg.hidden_size

        self._window_inputs = {}                 # layer_idx → (1, W, D) on layer device
        self._answer = None                      # (L, H, S) CPU
        self._decoded_q_sum = None               # (L, H, D) CPU — post-rotary
        self._decode_count = 0
        self._prefill_len = 0
        self._is_prefill = True

        self._hooks = []
        for i, layer in enumerate(model.model.layers):
            h1 = layer.self_attn.register_forward_pre_hook(
                self._pre_hook(i), with_kwargs=True
            )
            h2 = layer.self_attn.register_forward_hook(self._post_hook(i))
            self._hooks.extend([h1, h2])

    # ── public ──
    def reset(self, prefill_len):
        self._window_inputs.clear()
        self._answer = torch.zeros(self.num_layers, self.num_heads, prefill_len)
        self._decoded_q_sum = torch.zeros(self.num_layers, self.num_heads, self.head_dim)
        self._decode_count = 0
        self._prefill_len = prefill_len
        self._is_prefill = True

    def set_decode(self):
        self._is_prefill = False

    @property
    def answer_score(self):
        return self._answer

    @property
    def decoded_q_avg(self):
        if self._decode_count == 0:
            return torch.zeros(self.num_layers, self.num_heads, self.head_dim)
        return self._decoded_q_sum / self._decode_count

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── hooks ──
    def _pre_hook(self, layer_idx):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states")
            if hidden is None:
                hidden = args[0] if args else None
            if hidden is None:
                return

            if self._is_prefill:
                if hidden.size(1) > 1:
                    w = min(self.max_window, hidden.size(1))
                    self._window_inputs[layer_idx] = hidden[:, -w:, :].detach()
                return

            # decode path: compute & accumulate post-rotary Q
            if hidden.size(1) != 1:
                return
            with torch.no_grad():
                q = module.q_proj(hidden)
                q = q.view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
                # rotary: prefer pre-computed position_embeddings, else use position_ids
                pos_emb = kwargs.get("position_embeddings")
                if pos_emb is not None:
                    cos, sin = pos_emb
                else:
                    pos_ids = kwargs.get("position_ids")
                    if pos_ids is None:
                        cur = self._prefill_len + self._decode_count
                        pos_ids = torch.tensor([[cur]], device=hidden.device)
                    cos, sin = module.rotary_emb(q, pos_ids)
                q_rot, _ = apply_rotary_pos_emb(q, q, cos, sin)
                self._decoded_q_sum[layer_idx] += q_rot[0, :, 0, :].detach().float().cpu()
                if layer_idx == 0:
                    self._decode_count += 1
        return hook

    def _post_hook(self, layer_idx):
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

    # ── post-hoc reconstruction ──
    def compute_window_data(self, past_kv):
        """Returns dict with per-layer tensors needed for analyses."""
        S = self._prefill_len
        prefill_attn, window_q, k_self_sim, k_norms = [], [], [], []

        for i in range(self.num_layers):
            attn_mod = self.model.model.layers[i].self_attn
            hidden = self._window_inputs[i]
            device = hidden.device
            W = hidden.size(1)

            # Q for window
            q = attn_mod.q_proj(hidden)
            q = q.view(1, W, self.num_heads, self.head_dim).transpose(1, 2)
            pos_ids = torch.arange(S - W, S, device=device).unsqueeze(0)
            cos, sin = attn_mod.rotary_emb(q, pos_ids)
            q_rot, _ = apply_rotary_pos_emb(q, q, cos, sin)

            # K from cache
            k = past_kv[i][0]                          # (1, nkv, S, D)

            # Attention scores (grouped matmul)
            q_g = q_rot.view(1, self.num_kv_heads, self.group_size, W, self.head_dim)
            k_t = k.unsqueeze(2).transpose(-1, -2)
            scores = torch.matmul(q_g, k_t) / math.sqrt(self.head_dim)
            scores = scores.view(1, self.num_heads, W, S)

            # Causal mask
            kp = torch.arange(S, device=device)
            qp = torch.arange(S - W, S, device=device)
            mask = kp.unsqueeze(0) <= qp.unsqueeze(1)
            scores.masked_fill_(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            attn_w = torch.softmax(scores.float(), dim=-1).to(q_rot.dtype)
            prefill_attn.append(attn_w[0].cpu())       # (H, W, S)
            window_q.append(q_rot[0].cpu())            # (H, W, D)

            # K self-similarity to last K (per kv-head, then mean)
            k_n = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            last = k_n[:, :, -1:, :]
            sim = (k_n * last).sum(dim=-1)             # (1, nkv, S)
            k_self_sim.append(sim[0].mean(dim=0).cpu())          # (S,)
            k_norms.append(k.float().norm(dim=-1)[0].mean(dim=0).cpu())  # (S,)

            del q, q_rot, q_g, k_t, scores, attn_w, hidden, k, k_n, last, sim
            self._window_inputs[i] = None
            torch.cuda.empty_cache()

        return {
            "prefill_attn": torch.stack(prefill_attn),   # (L, H, W, S)
            "window_q":     torch.stack(window_q),       # (L, H, W, D)
            "k_self_sim":   torch.stack(k_self_sim),     # (L, S)
            "k_norms":      torch.stack(k_norms),        # (L, S)
        }


# ══════════════════════════════════════════════════════════════
# Vectorised helpers
# ══════════════════════════════════════════════════════════════
def jaccard(a_idx, b_idx, seq_len, per_layer=False):
    L, H, _ = a_idx.shape
    a = torch.zeros(L, H, seq_len, dtype=torch.bool)
    b = torch.zeros(L, H, seq_len, dtype=torch.bool)
    a.scatter_(2, a_idx, True)
    b.scatter_(2, b_idx, True)
    inter = (a & b).sum(2).float()
    union = (a | b).sum(2).float().clamp(min=1)
    j = inter / union
    return j.mean(dim=1).numpy() if per_layer else j.mean().item()


def gqa_topk(scores, k, num_kv_heads, group_size):
    if group_size <= 1:
        return scores.topk(k, dim=-1).indices
    *batch, _H, S = scores.shape
    g = scores.view(*batch, num_kv_heads, group_size, S).sum(dim=-2)
    idx = g.topk(k, dim=-1).indices
    return idx.repeat_interleave(group_size, dim=-2)


# ══════════════════════════════════════════════════════════════
# Existing analyses (block hit rate, optimal coefficient, cumulative)
# ══════════════════════════════════════════════════════════════
def analyze_block_hit_rate(prefill, answer_idx, budget, kv_h, gs, seq_len):
    W = prefill.size(2)
    rates = []
    for b in range(W):
        score = prefill[:, :, W - 1 - b, :]
        idx = gqa_topk(score, budget, kv_h, gs)
        rates.append(jaccard(answer_idx, idx, seq_len))
    return np.array(rates)


def analyze_optimal_coefficient(prefill, answer_idx, budget, kv_h, gs, seq_len):
    L, H, W, S = prefill.shape
    local_b = max(1, int(budget * LOCAL_RATIO))
    sel_b = budget - local_b
    test_coeffs = torch.tensor(COEFF_GRID)

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
    L, H, W, S = prefill.shape
    accumulated = torch.zeros(L, H, S)
    sims = []
    for b in range(W):
        accumulated += prefill[:, :, W - 1 - b, :]
        idx = gqa_topk(accumulated, budget, kv_h, gs)
        sims.append(jaccard(answer_idx, idx, seq_len))
    return np.array(sims)


# ══════════════════════════════════════════════════════════════
# New diagnostic analyses
# ══════════════════════════════════════════════════════════════
def _flip_recent_first(t):
    """Reverse so index 0 = most recent (closest to decode boundary)."""
    return t.flip(dims=[-1])


def analyze_q_norm(window_q):
    """H3: Q magnitude by distance from end. Returns (W,)."""
    n = window_q.float().norm(dim=-1)                 # (L, H, W)
    return _flip_recent_first(n).mean(dim=(0, 1)).numpy()


def analyze_attention_entropy(prefill_attn):
    """H4: lower entropy = sharper attention. Returns (W,)."""
    p = prefill_attn.float().clamp(min=1e-12)
    e = -(p * p.log()).sum(dim=-1)                    # (L, H, W)
    return _flip_recent_first(e).mean(dim=(0, 1)).numpy()


def analyze_attention_sharpness(prefill_attn):
    """H4 alt: max attention probability. Returns (W,)."""
    s = prefill_attn.float().max(dim=-1).values        # (L, H, W)
    return _flip_recent_first(s).mean(dim=(0, 1)).numpy()


def analyze_q_q_similarity(window_q):
    """H5: cosine similarity matrix among window queries (W × W).
       Returns (sim_matrix W×W, sim_to_recent W,) — all recent-first.
    """
    q = window_q.float()
    qn = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = torch.matmul(qn, qn.transpose(-1, -2))      # (L, H, W, W)
    sim = sim.flip(dims=[-1, -2])                      # recent-first on both axes
    M = sim.mean(dim=(0, 1)).numpy()
    return M, M[0, :]


def analyze_attention_pattern_correlation(prefill_attn):
    """H2/H5: Pearson correlation between attention vectors of pairs of window queries."""
    a = prefill_attn.float()
    ac = a - a.mean(dim=-1, keepdim=True)
    n = ac.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    an = ac / n
    corr = torch.matmul(an, an.transpose(-1, -2))
    corr = corr.flip(dims=[-1, -2])
    M = corr.mean(dim=(0, 1)).numpy()
    return M, M[0, :]


def analyze_attention_topk_overlap(prefill_attn, budget, kv_h, gs):
    """Jaccard overlap of top-k indices between window queries (W × W).

    Computed layer-by-layer to bound memory: per-layer one-hot ≈ 512 MB at S=32K.
    """
    L, H, W, S = prefill_attn.shape
    M_sum = np.zeros((W, W))

    for l in range(L):
        oh = torch.zeros(H, W, S, dtype=torch.float32)
        for w in range(W):
            score = prefill_attn[l, :, w, :].unsqueeze(0)         # (1, H, S)
            idx = gqa_topk(score, budget, kv_h, gs)[0]            # (H, k)
            oh[:, w, :].scatter_(1, idx, 1.0)
        # Pairwise Jaccard via matmul: (H, W, W)
        inter = torch.matmul(oh, oh.transpose(-1, -2))
        sums = oh.sum(-1)
        union = sums.unsqueeze(-1) + sums.unsqueeze(-2) - inter
        jacc = inter / union.clamp(min=1)
        M_sum += jacc.mean(dim=0).numpy()
        del oh, inter, sums, union, jacc

    M = M_sum / L
    M = M[::-1, ::-1]                                              # recent-first
    return M, M[0, :]


def analyze_q_decode_similarity(window_q, decoded_q):
    """H1: cosine similarity of each window Q to the average decoded Q. Returns (W,)."""
    q = window_q.float()                               # (L, H, W, D)
    dq = decoded_q.float()                             # (L, H, D)
    qn = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dqn = dq / dq.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = (qn * dqn.unsqueeze(-2)).sum(dim=-1)         # (L, H, W)
    return _flip_recent_first(sim).mean(dim=(0, 1)).numpy()


def analyze_q_decode_attn_corr(prefill_attn, answer_score):
    """H2: Pearson correlation between each window Q's attention vector and the decode
       attention vector. Returns (W,)."""
    a = prefill_attn.float()                           # (L, H, W, S)
    d = answer_score.float()                           # (L, H, S)
    # Center
    ac = a - a.mean(dim=-1, keepdim=True)
    dc = d - d.mean(dim=-1, keepdim=True)
    an = ac / ac.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dn = dc / dc.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    corr = (an * dn.unsqueeze(-2)).sum(dim=-1)         # (L, H, W)
    return _flip_recent_first(corr).mean(dim=(0, 1)).numpy()


def analyze_per_layer_block_hit(prefill, answer_idx, budget, kv_h, gs, seq_len):
    """H7: per-layer block hit rate. Returns (L, W)."""
    L, _, W, _ = prefill.shape
    rates = np.zeros((L, W))
    for b in range(W):
        score = prefill[:, :, W - 1 - b, :]
        idx = gqa_topk(score, budget, kv_h, gs)
        rates[:, b] = jaccard(answer_idx, idx, seq_len, per_layer=True)
    return rates


def analyze_per_layer_optimal(prefill, answer_idx, budget, kv_h, gs, seq_len):
    """H7: per-layer greedy optimal coefficient. Returns (L, W)."""
    L, H, W, S = prefill.shape
    local_b = max(1, int(budget * LOCAL_RATIO))
    sel_b = budget - local_b
    test_coeffs = torch.tensor(COEFF_GRID)

    accumulated = torch.zeros(L, H, S)
    coeffs = np.zeros((L, W))

    # Per-layer best-so-far hit rate
    best_hrs = np.zeros(L)

    for b in range(W):
        block = prefill[:, :, W - 1 - b, :]
        if b == 0:
            accumulated += block
            idx = gqa_topk(accumulated, budget, kv_h, gs)
            best_hrs = jaccard(answer_idx, idx, seq_len, per_layer=True)
            coeffs[:, b] = 1.0
        else:
            # For each layer, find the coefficient maximising layer-specific hit rate
            for layer in range(L):
                best_c, best_hr = 0.0, best_hrs[layer]
                for c in test_coeffs:
                    tmp = accumulated[layer:layer + 1] + c * block[layer:layer + 1]
                    tmp[:, :, -local_b:] = tmp.max()
                    idx = gqa_topk(tmp, sel_b, kv_h, gs)
                    hr_arr = jaccard(answer_idx[layer:layer + 1], idx, seq_len, per_layer=True)
                    hr = hr_arr[0]
                    if hr > best_hr:
                        best_hr = hr
                        best_c = c.item()
                accumulated[layer] += best_c * block[layer]
                coeffs[layer, b] = best_c
                best_hrs[layer] = best_hr
    return coeffs


FIXED_K_BIN_EDGES = np.array([
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384, 32768,
])


def analyze_k_self_sim(k_self_sim):
    """H6: K self-similarity to last-K curve, fixed log-bins for cross-sample aggregation.
       Returns (bin_centers, mean_sim, std_sim) — NaN for bins beyond sequence length."""
    L, S = k_self_sim.shape
    sim = k_self_sim.float().mean(dim=0).numpy()       # (S,)
    distances = np.arange(S)[::-1]                      # S-1 .. 0  (matches sim positions)

    centers, means, stds = [], [], []
    for j in range(len(FIXED_K_BIN_EDGES) - 1):
        lo, hi = FIXED_K_BIN_EDGES[j], FIXED_K_BIN_EDGES[j + 1]
        centers.append(np.sqrt(max(lo, 1) * hi))        # geometric mean for log-axis
        if lo >= S:
            means.append(np.nan); stds.append(np.nan); continue
        m = (distances >= lo) & (distances < min(hi, S))
        if m.any():
            means.append(sim[m].mean()); stds.append(sim[m].std())
        else:
            means.append(np.nan); stds.append(np.nan)
    return np.array(centers), np.array(means), np.array(stds)


# ══════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════
def _sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(a * (x - b)))


def _trace_plot(ax, x, samples, color, label=None):
    """Plot many sample traces (light) plus mean (bold) plus std band."""
    arr = np.array(samples)
    for row in arr:
        ax.plot(x, row, alpha=0.15, color=color, linewidth=0.8)
    m, s = arr.mean(0), arr.std(0)
    ax.plot(x, m, color=color, linewidth=2.5, label=label)
    ax.fill_between(x, m - s, m + s, alpha=0.18, color=color)


def plot_temporal_bias(name, task, br, oc, cs, save_dir):
    n = len(br)
    W = min(len(r) for r in br)
    x = np.arange(W)
    fig, axes = plt.subplots(1, 3, figsize=(22, 5.5))

    _trace_plot(axes[0], x, [r[:W] for r in br], "steelblue")
    axes[0].set(xlabel="Query distance from end", ylabel="Hit Rate",
                title="Per-Query Hit Rate")
    axes[0].grid(True, alpha=0.3)

    arr_oc = np.array([c[:W] for c in oc])
    for row in arr_oc:
        axes[1].plot(x, row, alpha=0.15, color="coral", linewidth=0.8)
    m_c = arr_oc.mean(0)
    axes[1].plot(x, m_c, color="coral", linewidth=2.5, label="Mean")
    try:
        popt, _ = curve_fit(_sigmoid, x.astype(float), m_c,
                            p0=[0.5, 16.0], bounds=((0, 0), (np.inf, np.inf)),
                            maxfev=5000)
        axes[1].plot(x, _sigmoid(x.astype(float), *popt), "k--", linewidth=2,
                     label=f"Sigmoid (a={popt[0]:.2f}, b={popt[1]:.1f})")
        axes[1].legend(loc="upper right")
    except RuntimeError:
        pass
    axes[1].set(xlabel="Query distance from end", ylabel="Optimal Coefficient",
                title="Optimal Query Weight", ylim=(-0.05, 1.05))
    axes[1].grid(True, alpha=0.3)

    _trace_plot(axes[2], x, [s[:W] for s in cs], "forestgreen")
    axes[2].set(xlabel="Number of accumulated queries", ylabel="Similarity",
                title="Cumulative Similarity")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"[A] Core: {task}: {name} (n={n})", fontsize=20, fontweight="bold")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    p = os.path.join(save_dir, "A_core.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p}")


def plot_query_characteristics(name, task, q_norm, ent, sharp, q_dec_sim, q_dec_corr, save_dir):
    n = len(q_norm)
    W = min(len(r) for r in q_norm)
    x = np.arange(W)
    fig, axes = plt.subplots(2, 3, figsize=(22, 11))

    _trace_plot(axes[0, 0], x, [r[:W] for r in q_norm], "navy")
    axes[0, 0].set(xlabel="Query distance from end", ylabel="‖Q‖₂",
                   title="H3: Q-vector L2 norm")
    axes[0, 0].grid(True, alpha=0.3)

    _trace_plot(axes[0, 1], x, [r[:W] for r in ent], "firebrick")
    axes[0, 1].set(xlabel="Query distance from end", ylabel="Entropy (nats)",
                   title="H4: Attention entropy (sharper ↓)")
    axes[0, 1].grid(True, alpha=0.3)

    _trace_plot(axes[0, 2], x, [r[:W] for r in sharp], "darkorange")
    axes[0, 2].set(xlabel="Query distance from end", ylabel="max attention prob",
                   title="H4 alt: Attention sharpness (peakiness)")
    axes[0, 2].grid(True, alpha=0.3)

    _trace_plot(axes[1, 0], x, [r[:W] for r in q_dec_sim], "darkgreen")
    axes[1, 0].set(xlabel="Query distance from end", ylabel="cos sim",
                   title="H1: Window-Q vs decoded-Q cosine sim",
                   ylim=(-0.1, 1.05))
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(0, color="k", linewidth=0.5)

    _trace_plot(axes[1, 1], x, [r[:W] for r in q_dec_corr], "purple")
    axes[1, 1].set(xlabel="Query distance from end", ylabel="Pearson r",
                   title="H2: window-attn vs decoded-attn correlation",
                   ylim=(-0.1, 1.05))
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(0, color="k", linewidth=0.5)

    axes[1, 2].axis("off")

    fig.suptitle(f"[B] Query characteristics: {task}: {name} (n={n})",
                 fontsize=20, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(save_dir, "B_query_characteristics.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p}")


def plot_pairwise(name, task, q_q_M, attn_corr_M, attn_jacc_M, save_dir):
    """3 heatmaps: Q-Q cos sim, attn Pearson, attn Jaccard. Mean across samples."""
    qq = np.mean(np.stack(q_q_M, axis=0), axis=0)
    ap = np.mean(np.stack(attn_corr_M, axis=0), axis=0)
    aj = np.mean(np.stack(attn_jacc_M, axis=0), axis=0)
    n = len(q_q_M)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    for ax, M, title, cmap, vmin, vmax in [
        (axes[0], qq, "H5a: Q–Q cosine similarity", "RdBu_r", -0.5, 1.0),
        (axes[1], ap, "H5b: Attention Pearson correlation", "RdBu_r", -0.5, 1.0),
        (axes[2], aj, "H5c: Top-k Jaccard overlap", "Blues", 0.0, 1.0),
    ]:
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        plt.colorbar(im, ax=ax, fraction=0.045)
        ax.set(xlabel="Query distance from end (j)",
               ylabel="Query distance from end (i)",
               title=title)

    fig.suptitle(f"[C] Pairwise structure (W×W): {task}: {name} (n={n})",
                 fontsize=20, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(save_dir, "C_pairwise.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p}")


def plot_pairwise_to_recent(name, task, qq_to_recent, ap_to_recent, aj_to_recent, save_dir):
    """1D version: similarity to most-recent query."""
    n = len(qq_to_recent)
    W = min(len(r) for r in qq_to_recent)
    x = np.arange(W)

    fig, axes = plt.subplots(1, 3, figsize=(22, 5.5))
    _trace_plot(axes[0], x, [r[:W] for r in qq_to_recent], "navy")
    axes[0].set(xlabel="Query distance from most-recent",
                ylabel="cosine similarity",
                title="H5a: Q–Q sim to most recent",
                ylim=(-0.1, 1.05))
    axes[0].axhline(0, color="k", lw=0.5); axes[0].grid(True, alpha=0.3)

    _trace_plot(axes[1], x, [r[:W] for r in ap_to_recent], "firebrick")
    axes[1].set(xlabel="Query distance from most-recent",
                ylabel="Pearson r",
                title="H5b: attn-pattern correlation to most recent",
                ylim=(-0.1, 1.05))
    axes[1].axhline(0, color="k", lw=0.5); axes[1].grid(True, alpha=0.3)

    _trace_plot(axes[2], x, [r[:W] for r in aj_to_recent], "darkorange")
    axes[2].set(xlabel="Query distance from most-recent",
                ylabel="Jaccard",
                title="H5c: top-k Jaccard with most recent",
                ylim=(-0.05, 1.05))
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"[C'] Pairwise (1-D slice to most recent): {task}: {name} (n={n})",
                 fontsize=20, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(save_dir, "C2_pairwise_1d.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p}")


def plot_key_structure(name, task, k_curves, save_dir):
    """K self-similarity to last K, fixed log bins → safe nanmean across samples."""
    fig, ax = plt.subplots(figsize=(10, 6))
    centers = k_curves[0][0]
    for _, m, _ in k_curves:
        ax.plot(centers, m, alpha=0.25, color="teal", linewidth=1)
    all_means = np.stack([m for _, m, _ in k_curves], axis=0)      # (n, n_bins)
    mean = np.nanmean(all_means, axis=0)
    std = np.nanstd(all_means, axis=0)
    ax.plot(centers, mean, color="teal", linewidth=2.5, label="Mean")
    ax.fill_between(centers, mean - std, mean + std, alpha=0.18, color="teal")
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="Distance from last key (log scale)",
           ylabel="cos(K[end-d], K[end])",
           title="H6: Key vector self-similarity vs distance")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.suptitle(f"[D] Key structure: {task}: {name}", fontsize=20, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(save_dir, "D_key_structure.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p}")


def plot_layer_breakdown(name, task, per_layer_hit, per_layer_coef, save_dir):
    """H7: per-layer block hit rate and per-layer optimal coefficient (mean across samples)."""
    hit = np.mean(np.stack(per_layer_hit, axis=0), axis=0)   # (L, W)
    coef = np.mean(np.stack(per_layer_coef, axis=0), axis=0) # (L, W)

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    for ax, M, title, vmin, vmax in [
        (axes[0], hit, "H7: Per-layer Block Hit Rate", 0.0, hit.max()),
        (axes[1], coef, "H7: Per-layer Optimal Coefficient", 0.0, 1.0),
    ]:
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.045)
        ax.set(xlabel="Query distance from end",
               ylabel="Layer index",
               title=title)

    fig.suptitle(f"[E] Layer breakdown: {task}: {name} (n={len(per_layer_hit)})",
                 fontsize=20, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(save_dir, "E_layer_breakdown.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p}")


def plot_hypothesis_correlations(name, task, agg, save_dir):
    """
    Quantitative test: which per-position metric best PREDICTS the optimal coefficient?
    Computes Pearson correlation between optimal_coefficient[pos] and each candidate
    explanatory metric[pos], averaged over samples. Bar chart of |r|.
    """
    W = min(len(c) for c in agg["oc"])
    oc_arr = np.array([c[:W] for c in agg["oc"]])  # (n, W)

    candidates = {
        "H1: Q–decode cos sim":      np.array([r[:W] for r in agg["qd_sim"]]),
        "H2: window-attn↔dec corr":  np.array([r[:W] for r in agg["qd_corr"]]),
        "H3: ‖Q‖":                   np.array([r[:W] for r in agg["qnorm"]]),
        "H4a: −entropy":             -np.array([r[:W] for r in agg["ent"]]),
        "H4b: max-prob (sharp)":     np.array([r[:W] for r in agg["sharp"]]),
        "H5a: Q–Q sim to recent":    np.array([r[:W] for r in agg["qq_recent"]]),
        "H5b: attn-pat sim recent":  np.array([r[:W] for r in agg["ap_recent"]]),
        "H5c: top-k overlap recent": np.array([r[:W] for r in agg["aj_recent"]]),
        "Block hit rate":            np.array([r[:W] for r in agg["br"]]),
    }

    rows = []
    for label, arr in candidates.items():
        # per-sample Pearson, then mean & std
        rs = []
        for i in range(arr.shape[0]):
            x, y = arr[i], oc_arr[i]
            if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                continue
            rs.append(np.corrcoef(x, y)[0, 1])
        if rs:
            rows.append((label, np.mean(rs), np.std(rs)))
        else:
            rows.append((label, np.nan, np.nan))

    rows.sort(key=lambda r: -abs(r[1]) if not np.isnan(r[1]) else 0)
    labels = [r[0] for r in rows]
    means = [r[1] for r in rows]
    stds = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(labels))
    colors = ["forestgreen" if m > 0 else "firebrick" for m in means]
    ax.barh(y, means, xerr=stds, color=colors, alpha=0.75, edgecolor="black")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Pearson r vs optimal coefficient (mean ± std across samples)")
    ax.set_xlim(-1, 1)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    fig.suptitle(f"[F] Predictors of optimal coefficient: {task}: {name}",
                 fontsize=18, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(save_dir, "F_predictors.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p}")
    return rows


def plot_summary_across_datasets(all_results, save_dir):
    """All datasets' mean optimal-coefficient decay on one plot."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_results)))
    for (name, task, oc_list), color in zip(all_results, colors):
        W = min(len(c) for c in oc_list)
        oc = np.array([c[:W] for c in oc_list])
        m = oc.mean(0)
        x = np.arange(W)
        ax.plot(x, m, linewidth=2.5, color=color, label=f"{name} ({task})")
        try:
            popt, _ = curve_fit(_sigmoid, x.astype(float), m,
                                p0=[0.5, 16.0],
                                bounds=((0, 0), (np.inf, np.inf)), maxfev=5000)
            ax.plot(x, _sigmoid(x.astype(float), *popt), "--", color=color,
                    linewidth=1.5, alpha=0.7)
        except RuntimeError:
            pass
    ax.set(xlabel="Query distance from end", ylabel="Optimal Coefficient",
           title="Temporal Bias: Optimal Coefficient Decay (all datasets)",
           ylim=(-0.05, 1.05))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, "summary.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {p}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    with open(os.path.join(root_path, "config", "model2path.json")) as f:
        model2path = json.load(f)
    with open(os.path.join(root_path, "config", "dataset2maxlen.json")) as f:
        dataset2maxlen = json.load(f)
    model_path = model2path[MODEL_NAME]

    longbench_dir = os.path.join(root_path, "datasets", "longbench")
    dataset_prompts = defaultdict(list)
    for fname in os.listdir(longbench_dir):
        with open(os.path.join(longbench_dir, fname)) as f:
            for line in f:
                item = json.loads(line)
                length = item.get("length", 0)
                if LENGTH_MIN <= length <= LENGTH_MAX:
                    dataset_prompts[item["dataset"]].append(item["input_prompt"])

    dataset2task, selected = {}, {}
    for task_name, datasets in TASK_GROUP.items():
        for d in datasets:
            dataset2task[d] = task_name
            if d in dataset_prompts:
                pool = dataset_prompts[d]
                selected[d] = random.sample(pool, min(NUM_ITEMS, len(pool)))

    # ── Force model distribution across all visible GPUs ──
    n_gpus = torch.cuda.device_count()
    print(f"Loading {model_path} across {n_gpus} GPUs (manual device map)…")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Build a manual device_map. Llama 3.2 has tied embeddings (lm_head ↔ embed_tokens),
    # so they must share a GPU. We co-locate all non-layer modules on GPU 0 and
    # distribute the transformer layers across all GPUs.
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path)
    n_layers = cfg.num_hidden_layers
    device_map = {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,
        "model.norm": 0,
        "lm_head": 0,
    }
    for i in range(n_layers):
        device_map[f"model.layers.{i}"] = (i * n_gpus) // n_layers

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa",
    ).eval()

    # Print device map for visibility
    if hasattr(model, "hf_device_map"):
        unique_devices = sorted(set(model.hf_device_map.values()))
        print(f"  model spread across devices: {unique_devices}")

    collector = AttentionCollector(model, MAX_WINDOW)
    plot_dir = os.path.join(workpath, "plots", PLOT_SUBDIR) if PLOT_SUBDIR else os.path.join(workpath, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    all_results = []

    for dataset_name, prompts in selected.items():
        task_name = dataset2task[dataset_name]
        print(f"\n{'=' * 70}\n  {task_name}: {dataset_name}  ({len(prompts)} samples)\n{'=' * 70}")

        # accumulators
        agg = defaultdict(list)

        for idx, prompt in enumerate(prompts):
            print(f"\n  [{idx + 1}/{len(prompts)}] ", end="", flush=True)

            enc = tokenizer(f"[INST]{prompt}[/INST]", return_tensors="pt")
            input_ids = enc.input_ids.to(model.device)
            if input_ids.size(1) > MAX_SEQ_LEN:
                half = MAX_SEQ_LEN // 2
                input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)
            seq_len = input_ids.size(1)
            max_new = dataset2maxlen.get(dataset_name, 512)

            collector.reset(seq_len)

            with torch.no_grad():
                # num_logits_to_keep=1 avoids materialising (S, 128K vocab) at long S
                out = model(input_ids, use_cache=True, num_logits_to_keep=1)
                past_kv = out.past_key_values
                next_tok = out.logits[:, -1:].argmax(dim=-1)
                del out
                torch.cuda.empty_cache()

                # Reconstruct window data
                data = collector.compute_window_data(past_kv)

                # Decode
                collector.set_decode()
                gen_len = 0
                for _ in range(max_new):
                    out = model(next_tok, past_key_values=past_kv,
                                use_cache=True, output_attentions=True)
                    past_kv = out.past_key_values
                    next_tok = out.logits[:, -1:].argmax(dim=-1)
                    gen_len += 1
                    if next_tok.item() == tokenizer.eos_token_id:
                        del out; break
                    del out

            del past_kv
            torch.cuda.empty_cache()
            print(f"prefill={seq_len}  gen={gen_len}", end="  ", flush=True)

            prefill_attn = data["prefill_attn"]
            window_q     = data["window_q"]
            k_self_sim   = data["k_self_sim"]

            answer_score = collector.answer_score
            decoded_q    = collector.decoded_q_avg

            # Ground truth top-k from decode attention
            answer_idx = gqa_topk(answer_score, TOKEN_BUDGET,
                                  collector.num_kv_heads, collector.group_size)
            kv_h, gs = collector.num_kv_heads, collector.group_size

            # ── A: Core ──
            br = analyze_block_hit_rate(prefill_attn, answer_idx, TOKEN_BUDGET, kv_h, gs, seq_len)
            oc, _ = analyze_optimal_coefficient(prefill_attn, answer_idx, TOKEN_BUDGET, kv_h, gs, seq_len)
            cs = analyze_cumulative(prefill_attn, answer_idx, TOKEN_BUDGET, kv_h, gs, seq_len)
            agg["br"].append(br); agg["oc"].append(oc); agg["cs"].append(cs)

            # ── B: Query characteristics ──
            agg["qnorm"].append(analyze_q_norm(window_q))
            agg["ent"].append(analyze_attention_entropy(prefill_attn))
            agg["sharp"].append(analyze_attention_sharpness(prefill_attn))
            agg["qd_sim"].append(analyze_q_decode_similarity(window_q, decoded_q))
            agg["qd_corr"].append(analyze_q_decode_attn_corr(prefill_attn, answer_score))

            # ── C: Pairwise structure ──
            qq_M, qq_recent = analyze_q_q_similarity(window_q)
            ap_M, ap_recent = analyze_attention_pattern_correlation(prefill_attn)
            aj_M, aj_recent = analyze_attention_topk_overlap(prefill_attn, TOKEN_BUDGET, kv_h, gs)
            agg["qq_M"].append(qq_M);   agg["qq_recent"].append(qq_recent)
            agg["ap_M"].append(ap_M);   agg["ap_recent"].append(ap_recent)
            agg["aj_M"].append(aj_M);   agg["aj_recent"].append(aj_recent)

            # ── D: K structure ──
            agg["k_curve"].append(analyze_k_self_sim(k_self_sim))

            # ── E: Layer breakdown ──
            agg["pl_hit"].append(analyze_per_layer_block_hit(
                prefill_attn, answer_idx, TOKEN_BUDGET, kv_h, gs, seq_len))
            agg["pl_coef"].append(analyze_per_layer_optimal(
                prefill_attn, answer_idx, TOKEN_BUDGET, kv_h, gs, seq_len))

            print(f"HR[0]={br[0]:.3f}  coeff[-1]={oc[-1]:.2f}  "
                  f"qd_sim[0]={agg['qd_sim'][-1][0]:.2f}", flush=True)

            # free per-sample tensors
            del data, prefill_attn, window_q, k_self_sim, answer_score, decoded_q
            torch.cuda.empty_cache()

        # ── Plots per dataset ──
        ds_dir = os.path.join(plot_dir, task_name.replace(" ", "_"), dataset_name)
        plot_temporal_bias(dataset_name, task_name,
                           agg["br"], agg["oc"], agg["cs"], ds_dir)
        plot_query_characteristics(dataset_name, task_name,
                                   agg["qnorm"], agg["ent"], agg["sharp"],
                                   agg["qd_sim"], agg["qd_corr"], ds_dir)
        plot_pairwise(dataset_name, task_name,
                      agg["qq_M"], agg["ap_M"], agg["aj_M"], ds_dir)
        plot_pairwise_to_recent(dataset_name, task_name,
                                agg["qq_recent"], agg["ap_recent"],
                                agg["aj_recent"], ds_dir)
        plot_key_structure(dataset_name, task_name, agg["k_curve"], ds_dir)
        plot_layer_breakdown(dataset_name, task_name,
                             agg["pl_hit"], agg["pl_coef"], ds_dir)
        rows = plot_hypothesis_correlations(dataset_name, task_name, agg, ds_dir)

        # ── Print quantitative summary to stdout ──
        print(f"\n  ── Predictors of optimal coefficient (sorted by |r|) ──")
        for label, m, s in rows:
            print(f"    {label:<32s}  r = {m:+.3f} ± {s:.3f}")

        # ── Save raw aggregated metrics ──
        save_path = os.path.join(ds_dir, "metrics.npz")
        np.savez_compressed(
            save_path,
            br=np.array([r for r in agg["br"]]),
            oc=np.array([r for r in agg["oc"]]),
            cs=np.array([r for r in agg["cs"]]),
            qnorm=np.array([r for r in agg["qnorm"]]),
            ent=np.array([r for r in agg["ent"]]),
            sharp=np.array([r for r in agg["sharp"]]),
            qd_sim=np.array([r for r in agg["qd_sim"]]),
            qd_corr=np.array([r for r in agg["qd_corr"]]),
            qq_recent=np.array([r for r in agg["qq_recent"]]),
            ap_recent=np.array([r for r in agg["ap_recent"]]),
            aj_recent=np.array([r for r in agg["aj_recent"]]),
            qq_M=np.array(agg["qq_M"]),
            ap_M=np.array(agg["ap_M"]),
            aj_M=np.array(agg["aj_M"]),
            pl_hit=np.array(agg["pl_hit"]),
            pl_coef=np.array(agg["pl_coef"]),
        )
        print(f"  saved metrics → {save_path}")

        all_results.append((dataset_name, task_name, agg["oc"]))

    plot_summary_across_datasets(all_results, plot_dir)

    collector.remove_hooks()
    print("\n>>> All done.")


if __name__ == "__main__":
    main()
