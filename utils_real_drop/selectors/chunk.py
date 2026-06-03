from typing import Dict, Optional

import torch

from .base import Selector


class ChunkSelector(Selector):
    """ChunkKV: group head-region tokens into consecutive chunks of `chunk_size`,
    pick top chunks by mean per-token score, then expand back to tokens. Always
    keeps the most recent `recent_budget` tokens.

    Optional layer-group index sharing (LIR): when `layer_group_size > 1`, the
    first layer of each group computes selection from its scores and caches
    the resulting kept-token indices; the remaining layers in the same group
    reuse those indices and skip score accumulation entirely (their attention
    runs the fast SDPA path because `needs_scores(layer_idx)` returns False).
    """

    def __init__(
        self,
        budgets,
        recent_budget: int = 16,
        chunk_size: int = 1,
        layer_group_size: int = 1,
    ):
        super().__init__(budgets, recent_budget=recent_budget)
        self.chunk_size = max(1, int(chunk_size))
        self.layer_group_size = max(1, int(layer_group_size))
        self._lir_cache: Dict[int, Optional[torch.Tensor]] = {}

    def needs_scores(self, layer_idx: int) -> bool:
        if self.layer_group_size <= 1:
            return True
        return (layer_idx % self.layer_group_size) == 0

    def _leader_of(self, layer_idx: int) -> int:
        return (layer_idx // self.layer_group_size) * self.layer_group_size

    def select(
        self, layer_idx: int, scores: Optional[torch.Tensor], seq_len_k: int
    ) -> Optional[torch.Tensor]:
        if self.layer_group_size <= 1:
            return super().select(layer_idx, scores, seq_len_k)

        leader = self._leader_of(layer_idx)
        if layer_idx == leader:
            indices = super().select(layer_idx, scores, seq_len_k)
            self._lir_cache[leader] = indices
            return indices
        return self._lir_cache.get(leader)

    def _select_impl(
        self, layer_idx: int, scores: torch.Tensor, seq_len_k: int, total_budget: int
    ) -> Optional[torch.Tensor]:
        batch, heads, _ = scores.shape
        device = scores.device
        cs = self.chunk_size
        head_len = seq_len_k - self.recent_budget
        select_budget = max(0, total_budget - self.recent_budget)

        if head_len <= 0 or select_budget <= 0:
            head_idx = torch.empty((batch, heads, 0), dtype=torch.long, device=device)
            return self._attach_recent_tail(head_idx, head_len, seq_len_k, self.recent_budget)

        num_chunks = head_len // cs
        if num_chunks == 0:
            head_idx = torch.arange(0, head_len, device=device, dtype=torch.long)
            head_idx = head_idx.expand(batch, heads, head_len)
            return self._attach_recent_tail(head_idx, head_len, seq_len_k, self.recent_budget)

        trimmed = scores[:, :, : num_chunks * cs]
        chunk_scores = trimmed.view(batch, heads, num_chunks, cs).mean(dim=-1)
        # Ceil rounding: any leftover budget rounds up to one extra chunk so
        # the budget never goes unused. Total kept tokens may exceed
        # select_budget by up to (cs - 1).
        k_chunks = min(num_chunks, (select_budget + cs - 1) // cs)
        if k_chunks == 0:
            head_idx = torch.empty((batch, heads, 0), dtype=torch.long, device=device)
            return self._attach_recent_tail(head_idx, head_len, seq_len_k, self.recent_budget)

        topk = chunk_scores.topk(k_chunks, dim=-1)
        chunk_indices = topk.indices.sort(dim=-1).values  # (B, H, k_chunks)
        offsets = torch.arange(cs, device=device, dtype=torch.long)
        head_idx = chunk_indices.unsqueeze(-1) * cs + offsets   # (B, H, k_chunks, cs)
        head_idx = head_idx.reshape(batch, heads, k_chunks * cs)
        return self._attach_recent_tail(head_idx, head_len, seq_len_k, self.recent_budget)
