import math

import torch

from .base import Scorer, SCORE_WEIGHT_EPS


class A2SFScorer(Scorer):
    """A2SF: exponential forgetting on accumulated attention scores."""

    def __init__(self, num_key_value_heads: int, forgetting_factor: float):
        super().__init__(num_key_value_heads)
        self.forgetting_factor = float(forgetting_factor)
        self._window: torch.Tensor = None  # [seq_len_q] fp32

    def prepare_prefill(self, seq_len_q, device, dtype, **kwargs):
        rev = torch.arange(seq_len_q - 1, -1, -1, device=device, dtype=torch.float32)
        self._window = self.forgetting_factor ** rev

    def get_query_weights(self, q_start, q_end, device, dtype):
        return self._window[q_start:q_end].to(device=device, dtype=torch.float32)

    def score_query_start(self, seq_len_q: int) -> int:
        """Earliest query whose forgetting weight reaches SCORE_WEIGHT_EPS.

        w[q] = ff^(N-1-q) >= eps  <=>  (N-1-q) <= log(eps)/log(ff)   (ff < 1)
          <=> q >= (N-1) - log(eps)/log(ff). ff >= 1 -> unbounded support -> 0.
        """
        ff = self.forgetting_factor
        if ff >= 1.0 or ff <= 0.0:
            return 0
        dmax = math.log(SCORE_WEIGHT_EPS) / math.log(ff)   # >= 0
        return int(max(0, min(seq_len_q, math.ceil((seq_len_q - 1) - dmax))))
