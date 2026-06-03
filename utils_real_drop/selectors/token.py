from typing import Optional

import torch

from .base import Selector


class TokenSelector(Selector):
    """Default top-k selection: pick top `select_budget` tokens by score, plus
    always-keep the most recent `recent_budget` tokens.
    """

    def _select_impl(
        self, layer_idx: int, scores: torch.Tensor, seq_len_k: int, total_budget: int
    ) -> Optional[torch.Tensor]:
        batch, heads, _ = scores.shape
        head_len = seq_len_k - self.recent_budget
        select_budget = max(0, total_budget - self.recent_budget)
        if select_budget > 0 and head_len > 0:
            topk = scores[:, :, :head_len].topk(select_budget, dim=-1)
            head_idx = topk.indices.sort(dim=-1).values
        else:
            head_idx = torch.empty(
                (batch, heads, 0), dtype=torch.long, device=scores.device
            )
        return self._attach_recent_tail(head_idx, head_len, seq_len_k, self.recent_budget)
