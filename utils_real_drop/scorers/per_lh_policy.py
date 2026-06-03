"""Per-(L, h) policy scorer: at prefill time, computes layer's snap from last W
queries × all keys, runs that layer's 8 per-head policies, sets per-head (a, b)
on a SigmoidScorer.

Adds NO 2-pass forward: extracts snap via a small extra (W × L) attention
computation in `prepare_prefill`, before the main score-accumulating attention
loop runs.
"""
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sigmoid import SigmoidScorer


# Action grid (must match training)
A_VALUES = torch.tensor([0.0, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 10.0, 10.0, 10.0, 10.0])
B_VALUES = torch.tensor([1, 1, 16, 32, 128, 1, 16, 32, 128, 1, 16, 32, 128], dtype=torch.float32)


class SimplePolicy(nn.Module):
    """Embedding + 2 FFN-Act + head, matches script/train_per_lh_simple_reg.py."""
    def __init__(self, in_dim, out_dim=13, hidden=128, dropout=0.0):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.ffn1 = nn.Linear(hidden, hidden)
        self.ffn2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, out_dim)
        self.act = nn.GELU()
    def forward(self, x):
        h = self.act(self.embed(x))
        h = self.act(self.ffn1(h))
        h = self.act(self.ffn2(h))
        return self.head(h)


def stats_top_k(score: torch.Tensor, topk: int = 64) -> torch.Tensor:
    """score: (G, L) → (G, 2k+3) — top-K positions + values + entropy/mean_pos/std_pos."""
    G, L = score.shape
    k = min(topk, L)
    if L < topk:
        pad = torch.zeros(G, topk - L, device=score.device, dtype=score.dtype)
        s = torch.cat([score, pad], dim=-1)
    else:
        s = score
    L_eff = s.size(-1)
    topk_vals, topk_idx = s.topk(k, dim=-1)
    denom = float(max(1, L_eff - 1))
    pos_from_end = (L_eff - 1 - topk_idx).float() / denom
    pos = s.clamp(min=0.0)
    sm = pos.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    p = pos / sm
    log_T = float(max(1e-6, math.log(L_eff)))
    ent = -(p * (p.clamp(min=1e-12)).log()).sum(dim=-1) / log_T
    positions_from_end = (L_eff - 1 - torch.arange(L_eff, device=s.device).float()) / denom
    mean_pos = (p * positions_from_end.unsqueeze(0)).sum(dim=-1)
    diff = positions_from_end.unsqueeze(0) - mean_pos.unsqueeze(-1)
    var_pos = (p * diff * diff).sum(dim=-1).clamp(min=0.0)
    std_pos = var_pos.sqrt()
    return torch.cat([
        pos_from_end, topk_vals.float(),
        ent.unsqueeze(-1).float(), mean_pos.unsqueeze(-1).float(),
        std_pos.unsqueeze(-1).float(),
    ], dim=-1)


_POLICY_BANK_CACHE = {}


def load_policies(policies_dir: str, n_layers: int, n_heads: int, in_dim: int, hidden: int):
    """Load all (n_layers × n_heads) SimplePolicy weights from a directory.
    Returns dict {(L, h): nn.Module}.
    Caches per directory.
    """
    key = (str(policies_dir), n_layers, n_heads, in_dim, hidden)
    if key in _POLICY_BANK_CACHE:
        return _POLICY_BANK_CACHE[key]
    pdir = Path(policies_dir)
    bank = {}
    for L in range(n_layers):
        for h in range(n_heads):
            ckpt = torch.load(pdir / f"L{L}h{h}.pt", map_location="cpu", weights_only=False)
            m = SimplePolicy(ckpt["in_dim"], 13, hidden=ckpt["hidden"])
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            for p in m.parameters(): p.requires_grad_(False)
            bank[(L, h)] = m
    _POLICY_BANK_CACHE[key] = bank
    return bank


class PerLHPolicyScorer(SigmoidScorer):
    """Per-layer scorer: runs head policies during prepare_prefill.

    After this scorer's prepare_prefill, self.a and self.b are length-num_kv
    (per-head), and the SigmoidScorer per-head window is built.
    """

    def __init__(self, num_kv_heads, layer_idx, policies_per_head,
                 a_values=None, b_values=None,
                 query_window=16, topk=64, sink_mask=4):
        # Init SigmoidScorer with placeholder per-head a, b (will be overwritten)
        a_init = torch.zeros(num_kv_heads, dtype=torch.float32)
        b_init = torch.ones(num_kv_heads, dtype=torch.float32)
        super().__init__(num_kv_heads, a=a_init, b=b_init)
        self.layer_idx = layer_idx
        self.policies_per_head = policies_per_head  # list/dict of 8 nn.Module
        self.a_values = a_values if a_values is not None else A_VALUES.clone()
        self.b_values = b_values if b_values is not None else B_VALUES.clone()
        self.query_window = int(query_window)
        self.topk = int(topk)
        self.sink_mask = int(sink_mask)

    def prepare_prefill(self, seq_len_q, device, dtype, query=None, key=None, num_kv=None):
        if query is None or key is None:
            super().prepare_prefill(seq_len_q, device, dtype)
            return
        with torch.no_grad():
            B, num_q_heads, _, head_dim = query.shape
            nk = int(num_kv if num_kv is not None else self.num_key_value_heads)
            group = num_q_heads // nk
            W = min(self.query_window, seq_len_q)
            q_last = query[:, :, -W:, :].view(B, nk, group, W, head_dim).mean(dim=2)
            # key shape: (B, num_kv, L, head_dim) — already in KV-head space (post repeat_kv? no)
            # Looking at attention.py: key is in num_kv space if num_heads % num_kv == 0; in
            # this codebase the cache stores K/V at num_kv heads. Assume key shape (B, num_kv, L, D).
            # If key has more heads (e.g., already repeat_kv'd), reduce.
            if key.size(1) != nk:
                # repeat_kv'd: average over groups (since same K is repeated per group)
                k_g = key.view(B, nk, -1, key.size(2), head_dim)[:, :, 0]
            else:
                k_g = key
            scores = torch.einsum("bgwd,bgld->bgwl", q_last, k_g) / math.sqrt(head_dim)
            qpos = torch.arange(seq_len_q - W, seq_len_q, device=device)
            kpos = torch.arange(seq_len_q, device=device)
            causal = kpos.unsqueeze(0) > qpos.unsqueeze(1)
            scores = scores.masked_fill(causal[None, None, :, :], float("-inf"))
            attn = F.softmax(scores.float(), dim=-1)
            snap_per_head = attn.sum(dim=2)[0]  # (num_kv, L)
            if self.sink_mask > 0:
                snap_per_head = snap_per_head.clone()
                snap_per_head[:, :self.sink_mask] = 0.0
            stats = stats_top_k(snap_per_head, self.topk)  # (num_kv, 2k+3)
            new_a = torch.zeros(nk, dtype=torch.float32)
            new_b = torch.zeros(nk, dtype=torch.float32)
            for h in range(nk):
                if h not in self.policies_per_head and (self.layer_idx, h) not in self.policies_per_head:
                    continue
                pol = self.policies_per_head.get(h, None)
                if pol is None:
                    pol = self.policies_per_head[(self.layer_idx, h)]
                pol_dev = next(pol.parameters()).device
                pol_in = stats[h:h+1].to(pol_dev)
                logits = pol(pol_in)
                ai = int(logits.argmax(dim=-1).item())
                new_a[h] = self.a_values[ai]
                new_b[h] = self.b_values[ai]
            # Update SigmoidScorer's a, b → per-head mode
            self.a = new_a
            self.b = new_b
            self.per_head = True
        # Build window with per-head (a, b)
        super().prepare_prefill(seq_len_q, device, dtype)

    def is_per_head(self) -> bool:
        return True
