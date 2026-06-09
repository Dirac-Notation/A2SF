import math

import torch

from .base import Scorer, SCORE_WEIGHT_EPS


class SigmoidScorer(Scorer):
    """Sigmoid forgetting window with midpoint shifted between tokens.

    w[q] = sigmoid(a * (q - (N - b - 0.5)))

    Action-batch mode: `a`/`b` are scalar or 1-D length 1 (single action) or
    N_actions (batched data generation). Window shape: (N, seq_len_q); attention
    broadcasts across heads.
    """

    def __init__(self, num_key_value_heads, a, b):
        super().__init__(num_key_value_heads)
        self.a = self._as_1d(a)
        self.b = self._as_1d(b)
        self._window: torch.Tensor = None

    @staticmethod
    def _as_1d(value):
        if isinstance(value, torch.Tensor):
            return value.detach().to(torch.float32).reshape(-1)
        return torch.tensor([float(value)], dtype=torch.float32)

    def prepare_prefill(self, seq_len_q, device, dtype, **kwargs):
        # w[q] = sigmoid(a * (q - (N - b - 0.5))), midpoint between tokens
        # N-b-1 and N-b. Matches paper §4.1.
        exponents = torch.arange(seq_len_q, device=device, dtype=torch.float32)
        a = self.a.to(device=device, dtype=torch.float32).view(-1, 1)
        b = self.b.to(device=device, dtype=torch.float32).view(-1, 1)
        self._window = 1.0 / (1.0 + torch.exp(-a * (exponents - (seq_len_q - b - 0.5))))

    def get_query_weights(self, q_start, q_end, device, dtype):
        return self._window[:, q_start:q_end].to(device=device, dtype=torch.float32)

    def score_query_start(self, seq_len_q: int) -> int:
        """Earliest query whose sigmoid weight reaches SCORE_WEIGHT_EPS.

        w[q] = sigmoid(a*(q - (N-b-0.5))) >= eps
          <=> a*(q - (N-b-0.5)) >= logit(eps)
          <=> q >= (N-b-0.5) + logit(eps)/a       (a > 0; logit(eps) < 0)
        a <= 0 (flat / H2O) -> unbounded support -> 0. Across a batch of actions,
        take the widest window (smallest start).
        """
        logit_eps = math.log(SCORE_WEIGHT_EPS / (1.0 - SCORE_WEIGHT_EPS))  # < 0
        a = self.a.tolist()
        b = self.b.tolist()
        n_act = max(len(a), len(b))
        start = seq_len_q
        for i in range(n_act):
            ai = a[i] if i < len(a) else a[0]
            bi = b[i] if i < len(b) else b[0]
            if ai <= 0.0:
                return 0
            qmin = (seq_len_q - bi - 0.5) + logit_eps / ai
            start = min(start, qmin)
        return int(max(0, min(seq_len_q, math.floor(start))))
