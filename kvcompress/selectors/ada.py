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
    """`floor_alpha` is the paper's safeguard: each head is guaranteed alpha of the uniform share,
    and only the remaining (1 - alpha) is allocated by the flattened top-k. The paper ships 0.2;
    with alpha = 0 a head can come out of the global top-k with no adaptive tokens at all, which
    the paper's own ablation reports as worse."""

    def __init__(self, budgets, recent_budget: int = 16, n_sink: int = 0, floor_alpha: float = 0.2):
        super().__init__(budgets, recent_budget, n_sink)
        self.floor_alpha = float(floor_alpha)
        self.last_valid_mask = None

    def select(self, layer_idx, scores, seq_len_k):
        # A layer short enough to skip compression must not inherit the previous layer's mask:
        # compress.py copies whatever is here into _ada_mask, and a stale one would -inf out real
        # tokens at decode.
        self.last_valid_mask = None
        return super().select(layer_idx, scores, seq_len_k)

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

        seg_h = scores[:, :, n_sink:head_len]                     # [B, H, M]
        seg = seg_h.reshape(B, H * M)                             # flatten heads
        k = min(adaptive_total, H * M)
        flat = seg.topk(k, dim=-1).indices                        # [B, k]
        head_of = flat // M

        counts = torch.zeros(B, H, dtype=torch.long, device=dev)  # f_i in the paper's Alg. 1
        counts.scatter_add_(1, head_of, torch.ones_like(head_of))
        share = k / H
        a = self.floor_alpha
        quota = ((1.0 - a) * counts.float() + a * share).round().long().clamp(0, M)
        for b in range(B):                                        # rounding residue back onto the total
            d = int(k - int(quota[b].sum()))
            while d != 0:
                j = int(quota[b].argmin() if d > 0 else quota[b].argmax())
                step = 1 if d > 0 else -1
                if 0 <= int(quota[b, j]) + step <= M:
                    quota[b, j] += step
                    d -= step
                else:
                    break

        kept_bh = [[None] * H for _ in range(B)]
        max_b = 0
        for b in range(B):
            for h in range(H):
                q = int(quota[b, h])
                ph = seg_h[b, h].topk(q, dim=-1).indices + n_sink if q > 0 else \
                    torch.empty(0, dtype=torch.long, device=dev)
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
