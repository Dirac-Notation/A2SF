from typing import Optional

import torch


class CompressionPolicy:
    """Per-layer KV compression policy.

    Owns scheduling state (is the prefill done? what's the budget?) and the
    forgetting/scoring math, but does NOT own K/V tensors. The cache stores
    those; the attention kernel calls into this object to get per-query
    weights and final selection indices.

    Subclass contract:
      - prepare_prefill(seq_len_q, device, dtype): build per-prefill state
        (e.g. forgetting window). Called once at the start of the prefill pass.
      - get_query_weights(q_start, q_end, device, dtype) -> Tensor[qb] | None:
        per-query forgetting weights for the current q-block. None disables
        score accumulation (rare).
      - select(scores, seq_len_k) -> LongTensor[B, num_kv, total_budget] | None:
        decide which token positions to keep. None means "keep everything".
    """

    def __init__(self, num_key_value_heads: int, total_budget: int, recent_budget: int = 16):
        self.num_key_value_heads = num_key_value_heads
        self.total_budget = max(int(total_budget), 2)
        self.recent_budget = recent_budget
        self.select_budget = self.total_budget - self.recent_budget
        self.is_prefilled = False
        self.selected_indices: Optional[torch.Tensor] = None  # cached for external inspection

    # ---- lifecycle ----
    def reset(self) -> None:
        self.is_prefilled = False
        self.selected_indices = None

    def needs_scores(self) -> bool:
        """Whether the attention kernel should run the score-accumulating path."""
        return not self.is_prefilled

    def finalize_prefill(self) -> None:
        self.is_prefilled = True

    # ---- to override ----
    def prepare_prefill(self, seq_len_q: int, device: torch.device, dtype: torch.dtype) -> None:
        return

    def get_query_weights(
        self, q_start: int, q_end: int, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        return None

    def select(self, scores: torch.Tensor, seq_len_k: int) -> Optional[torch.Tensor]:
        return None

    # ---- shared helper for "topk + always-keep recent" pattern ----
    def _topk_with_recent(self, scores: torch.Tensor, seq_len_k: int) -> Optional[torch.Tensor]:
        if seq_len_k <= self.total_budget:
            return None
        scores = scores.clone()
        scores[:, :, -self.recent_budget:] = scores.max()
        indices = scores.topk(self.total_budget, dim=-1).indices.sort().values
        self.selected_indices = indices
        return indices
