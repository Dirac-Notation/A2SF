"""Ada-KV (Feng et al., NeurIPS 2025): head-wise ADAPTIVE budget allocation.

Orthogonal to the scorer: takes the same per-(kv-head) scores and, instead of a uniform
per-head budget, redistributes the layer's total budget (num_kv * budget) across heads via a
flattened top-k within the layer. Heads whose tokens score high keep more; others keep fewer.
Sink + recent tokens are always kept per head (a floor).

ACCURACY-SIMULATION implementation (pad-to-max + valid mask): real deployment needs the
flash_attn_varlen ragged kernel; here each head keeps its b_h tokens, the gathered cache is
padded to max_b (uniform) and the padding is masked out in attention -> identical OUTPUT, no
memory/speed benefit. `self.last_valid_mask` [B, num_kv, max_b] is read by the v5 attention fn.
"""
from typing import Optional

import torch

from .base import Selector


class AdaSelector(Selector):
    def __init__(self, budgets, recent_budget: int = 16, n_sink: int = 0):
        super().__init__(budgets, recent_budget, n_sink)
        self.last_valid_mask = None

    def _select_impl(self, layer_idx, scores, seq_len_k, total_budget):
        B, H, Sk = scores.shape                      # H = num_kv
        dev = scores.device
        n_sink = min(self.n_sink, max(0, Sk - self.recent_budget))
        head_len = Sk - self.recent_budget
        recent = self.recent_budget
        M = head_len - n_sink                        # scorable region width
        adaptive_total = max(0, total_budget - recent - n_sink) * H

        sink_idx = torch.arange(n_sink, device=dev)
        rec_idx = torch.arange(head_len, Sk, device=dev)

        if adaptive_total <= 0 or M <= 0:            # nothing adaptive -> uniform sink+recent
            kept = torch.cat([sink_idx, rec_idx])
            self.last_valid_mask = torch.ones(B, H, kept.numel(), dtype=torch.bool, device=dev)
            return kept.view(1, 1, -1).expand(B, H, -1).contiguous()

        seg = scores[:, :, n_sink:head_len].reshape(B, H * M)     # flatten heads
        k = min(adaptive_total, H * M)
        flat = seg.topk(k, dim=-1).indices                       # [B, k]
        head_of = flat // M
        pos_of = (flat % M) + n_sink

        kept_bh = [[None] * H for _ in range(B)]
        max_b = 0
        for b in range(B):
            for h in range(H):
                ph = pos_of[b][head_of[b] == h]
                kept = torch.cat([sink_idx, ph.sort().values, rec_idx])
                kept_bh[b][h] = kept
                max_b = max(max_b, int(kept.numel()))

        idx = torch.zeros(B, H, max_b, dtype=torch.long, device=dev)
        valid = torch.zeros(B, H, max_b, dtype=torch.bool, device=dev)
        for b in range(B):
            for h in range(H):
                kept = kept_bh[b][h]; n = int(kept.numel())
                idx[b, h, :n] = kept
                valid[b, h, :n] = True               # padding (>=n) stays index 0, masked
        self.last_valid_mask = valid
        return idx
