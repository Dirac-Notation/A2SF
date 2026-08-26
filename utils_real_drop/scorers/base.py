from typing import Optional

import torch

# Query weights below this are treated as negligible: queries whose forgetting
# weight is < SCORE_WEIGHT_EPS are skipped during score accumulation. Smaller =
# more exact, larger window. See `score_query_start`.
SCORE_WEIGHT_EPS = 1e-5


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

    def score_keys(self, query, key, num_kv):
        """ATTENTION-FREE direct scoring: return per-key importance [B, num_kv, Sk]
        computed from key/value vectors or position WITHOUT attention scores
        (StreamingLLM, L2-norm, KeyDiff, TriAttention). Default None -> the caller
        falls back to attention-based `_accumulate_scores`. key: [B, num_kv, Sk, hd]."""
        return None

    def score_query_start(self, seq_len_q: int) -> int:
        """Smallest prefill query index whose weight is non-negligible.

        Queries in [0, score_query_start) have weight < SCORE_WEIGHT_EPS and are
        skipped during score accumulation, bounding scoring to the recent window.
        Default 0 = unbounded support (all queries needed, e.g. H2O / a=0).
        """
        return 0
