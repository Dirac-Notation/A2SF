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
        n_sink = min(self.n_sink, max(0, seq_len_k - self.recent_budget))
        head_len = seq_len_k - self.recent_budget
        select_budget = max(0, total_budget - self.recent_budget - n_sink)
        avail = head_len - n_sink                       # scorable middle [n_sink:head_len]
        if select_budget > 0 and avail > 0:
            seg = scores[:, :, n_sink:head_len]
            topk = seg.topk(min(select_budget, avail), dim=-1)
            head_idx = (topk.indices + n_sink).sort(dim=-1).values
        else:
            head_idx = torch.empty(
                (batch, heads, 0), dtype=torch.long, device=scores.device
            )
        if n_sink > 0:                                   # always keep attention-sink tokens
            sink_idx = torch.arange(n_sink, device=scores.device, dtype=torch.long).expand(
                batch, heads, n_sink)
            head_idx = torch.cat([sink_idx, head_idx], dim=-1)
        return self._attach_recent_tail(head_idx, head_len, seq_len_k, self.recent_budget)
