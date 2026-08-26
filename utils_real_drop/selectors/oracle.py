"""OracleSelector: drops the score-accumulation path entirely and uses
precomputed kept-token indices supplied by the caller.

The intended use is the two-pass oracle evaluation: first run full-cache
generation while capturing per-decode-step attention to derive the
"post-generation reference set", then run a second pass with a
CompressedCache whose selector is this OracleSelector preloaded with
those indices. The framework's standard prefill→compress→decode flow
takes care of position_ids / RoPE without any monkey-patching.
"""
from typing import List, Optional

import torch

from .base import Selector


class OracleSelector(Selector):
    """Selector that returns precomputed indices and skips score accumulation.

    `oracle_indices` is a list of length num_layers, each entry
        (B=1, num_kv_heads, sel_budget) of int64
    holding the kept-token positions in the head region (i.e. positions
    in [0, seq_len_k - recent_budget)).  The local recent tail is appended
    by the base class helper.
    """

    def __init__(
        self,
        budgets: List[int],
        recent_budget: int = 16,
        oracle_indices: Optional[List[torch.Tensor]] = None,
    ):
        super().__init__(budgets, recent_budget=recent_budget)
        self.oracle_indices: Optional[List[torch.Tensor]] = oracle_indices

    def set_indices(self, oracle_indices: List[torch.Tensor]) -> None:
        self.oracle_indices = oracle_indices

    def needs_scores(self, layer_idx: int) -> bool:  # noqa: D401
        return False

    def _select_impl(
        self,
        layer_idx: int,
        scores: Optional[torch.Tensor],
        seq_len_k: int,
        total_budget: int,
    ) -> Optional[torch.Tensor]:
        if self.oracle_indices is None:
            return None
        head_idx = self.oracle_indices[layer_idx]
        if head_idx is None:
            return None
        head_len = seq_len_k - self.recent_budget
        # Clip indices to valid head region (defensive; should already be in range)
        head_idx = head_idx.clamp_(min=0, max=max(0, head_len - 1))
        return self._attach_recent_tail(
            head_idx, head_len, seq_len_k, self.recent_budget
        )
