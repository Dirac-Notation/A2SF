"""Base Selector + per-layer budget helpers.

A Selector owns the layer-aware budget and converts (B, num_kv, seq_len_k)
score tensors into kept-token indices [B, num_kv, total_budget_l]. It is the
single place where ChunkKV, PyramidKV, LIR, and similar selection-side
modifiers live, all orthogonal to the scoring math.
"""
from typing import List, Optional

import torch


def uniform_budgets(num_layers: int, base_budget: int) -> List[int]:
    return [int(base_budget)] * num_layers


def pyramid_budgets(num_layers: int, base_budget: int, ratio: float = 4.0) -> List[int]:
    """Linearly decreasing per-layer budgets summing to num_layers * base_budget.

    layer 0   : base_budget * (2*ratio / (ratio + 1))
    layer L-1 : base_budget * (2 / (ratio + 1))
    Average   : base_budget (preserves total cache size).
    """
    if num_layers <= 1 or ratio <= 1.0:
        return [int(base_budget)] * num_layers
    bottom = 2.0 * ratio / (ratio + 1.0) * base_budget
    top = 2.0 / (ratio + 1.0) * base_budget
    step = (bottom - top) / (num_layers - 1)
    raw = [int(round(bottom - i * step)) for i in range(num_layers)]
    target = num_layers * int(base_budget)
    diff = target - sum(raw)
    if diff != 0:
        raw[0] += diff
    return [max(4, b) for b in raw]


class Selector:
    """Score → kept-indices for every layer of the model."""

    def __init__(self, budgets: List[int], recent_budget: int = 16, n_sink: int = 0):
        self.budgets = [max(int(b), 2) for b in budgets]
        self.recent_budget = int(recent_budget)
        self.n_sink = int(n_sink)            # always-keep first n_sink (attention-sink) tokens
        self.num_layers = len(self.budgets)

    def budget_for(self, layer_idx: int) -> int:
        return self.budgets[layer_idx]

    def select_budget_for(self, layer_idx: int) -> int:
        return max(0, self.budgets[layer_idx] - self.recent_budget)

    def needs_scores(self, layer_idx: int) -> bool:
        """Whether this layer must run the score-accumulating attention path.

        Selectors that share a selection across layers (e.g. ChunkSelector's
        layer-group sharing / LIR) return False on follower layers so attention
        can take the fast SDPA path there.
        """
        return True

    def select(
        self, layer_idx: int, scores: torch.Tensor, seq_len_k: int
    ) -> Optional[torch.Tensor]:
        """Return kept indices [B, num_kv, k] or None to keep all positions."""
        budget = self.budgets[layer_idx]
        if seq_len_k <= budget:
            return None
        return self._select_impl(layer_idx, scores, seq_len_k, budget)

    # ---- to override ----
    def _select_impl(
        self, layer_idx: int, scores: torch.Tensor, seq_len_k: int, total_budget: int
    ) -> Optional[torch.Tensor]:
        raise NotImplementedError

    # ---- shared helper ----
    @staticmethod
    def _attach_recent_tail(
        head_idx: torch.Tensor, head_len: int, seq_len_k: int, recent_budget: int
    ) -> torch.Tensor:
        if recent_budget <= 0:
            return head_idx
        batch, heads, _ = head_idx.shape
        tail_idx = torch.arange(
            head_len, seq_len_k, device=head_idx.device, dtype=torch.long
        ).expand(batch, heads, recent_budget)
        return torch.cat([head_idx, tail_idx], dim=-1)
