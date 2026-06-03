from typing import Optional

import torch


class Scorer:
    """Per-layer scorer.

    Owns the per-query weighting curve used by score-accumulating attention.
    Knows nothing about budget, recency, or how the resulting scores are turned
    into kept indices. Selection lives in `selectors/`.

    Subclass contract:
      - prepare_prefill(seq_len_q, device, dtype): build per-prefill state
        (e.g. forgetting window). Called once at the start of the prefill pass.
      - get_query_weights(q_start, q_end, device, dtype) -> Tensor[qb] | Tensor[N, qb] | None:
        per-query weights for the current q-block. None disables score
        accumulation (rare).
    """

    def __init__(self, num_key_value_heads: int):
        self.num_key_value_heads = num_key_value_heads
        self.is_prefilled = False

    def reset(self) -> None:
        self.is_prefilled = False

    def needs_scores(self) -> bool:
        return not self.is_prefilled

    def finalize_prefill(self) -> None:
        self.is_prefilled = True

    def prepare_prefill(self, seq_len_q: int, device: torch.device, dtype: torch.dtype, **kwargs) -> None:
        return

    def get_query_weights(
        self, q_start: int, q_end: int, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        return None
