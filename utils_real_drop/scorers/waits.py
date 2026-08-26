import math

import torch

from .base import Scorer, SCORE_WEIGHT_EPS


class WaitsScorer(Scorer):
    """WAITS scorer: sigmoid forgetting window with midpoint shifted between tokens.

    This is the single, unified compression scorer for the method (an earlier
    exponential-forgetting variant was folded into this sigmoid form). The (a, b)
    action grid is interpreted with this formula in both training-data generation
    and inference.

    w[q] = sigmoid(a * (q - (N - b - 0.5)))

    Action-batch mode: `a`/`b` are scalar or 1-D length 1 (single action) or
    N_actions (batched data generation). Window shape: (N, seq_len_q); attention
    broadcasts across heads.
    """

    def __init__(self, num_key_value_heads, a, b, a_heads=None, b_heads=None,
                 curve="sigmoid"):
        super().__init__(num_key_value_heads)
        self.a = self._as_1d(a)
        self.b = self._as_1d(b)
        # Rebuttal alt-function families: curve in {sigmoid, exp, linear, gauss}.
        # Non-sigmoid curves use b as the scale parameter (tau / W / sigma) over the
        # distance d = (N-1) - q from the prompt end; a is ignored.
        self.curve = str(curve)
        # N3 head-portfolio: per-kv-head curves (lists of len num_kv). When set, the
        # window is [num_kv, Sq] and overrides scalar/batched a/b.
        self.a_heads = list(a_heads) if a_heads is not None else None
        self.b_heads = list(b_heads) if b_heads is not None else None
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
        if self.a_heads is not None:
            a = torch.tensor(self.a_heads, device=device, dtype=torch.float32).view(-1, 1)
            b = torch.tensor(self.b_heads, device=device, dtype=torch.float32).view(-1, 1)
            self._window = torch.where(
                a > 0, 1.0 / (1.0 + torch.exp(-a * (exponents - (seq_len_q - b - 0.5)))),
                torch.ones_like(a * exponents))          # a=0 -> uniform (H2O)
            self._per_head = True
            return
        self._per_head = False
        a = self.a.to(device=device, dtype=torch.float32).view(-1, 1)
        b = self.b.to(device=device, dtype=torch.float32).view(-1, 1)
        if self.curve != "sigmoid":
            d = (seq_len_q - 1) - exponents                      # distance from prompt end
            bs = b.clamp(min=1e-6)
            if self.curve == "exp":
                self._window = torch.exp(-d / bs)
            elif self.curve == "linear":
                self._window = (1.0 - d / bs).clamp(0.0, 1.0)
            elif self.curve == "gauss":
                self._window = torch.exp(-0.5 * (d / bs) ** 2)
            else:
                raise ValueError(f"unknown curve {self.curve!r}")
            return
        self._window = 1.0 / (1.0 + torch.exp(-a * (exponents - (seq_len_q - b - 0.5))))

    def get_query_weights(self, q_start, q_end, device, dtype):
        return self._window[:, q_start:q_end].to(device=device, dtype=torch.float32)

    def is_per_head(self):
        return getattr(self, "_per_head", False)

    def score_query_start(self, seq_len_q: int) -> int:
        """Earliest query whose sigmoid weight reaches SCORE_WEIGHT_EPS.

        w[q] = sigmoid(a*(q - (N-b-0.5))) >= eps
          <=> a*(q - (N-b-0.5)) >= logit(eps)
          <=> q >= (N-b-0.5) + logit(eps)/a       (a > 0; logit(eps) < 0)
        a <= 0 (flat / H2O) -> unbounded support -> 0. Across a batch of actions,
        take the widest window (smallest start).
        """
        if self.curve != "sigmoid":
            ln_inv_eps = -math.log(SCORE_WEIGHT_EPS)             # ~11.5
            start = seq_len_q
            for bi in self.b.tolist():
                if self.curve == "exp":
                    dmax = bi * ln_inv_eps
                elif self.curve == "linear":
                    dmax = bi
                else:                                            # gauss
                    dmax = bi * math.sqrt(2.0 * ln_inv_eps)
                start = min(start, (seq_len_q - 1) - dmax)
            return int(max(0, min(seq_len_q, math.floor(start))))
        logit_eps = math.log(SCORE_WEIGHT_EPS / (1.0 - SCORE_WEIGHT_EPS))  # < 0
        a = self.a_heads if self.a_heads is not None else self.a.tolist()
        b = self.b_heads if self.b_heads is not None else self.b.tolist()
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


