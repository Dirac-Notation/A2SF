import torch

from .base import Scorer


class SigmoidScorer(Scorer):
    """Sigmoid forgetting window with midpoint shifted between tokens.

    w[q] = sigmoid(a * (q - (N - b - 0.5)))

    Two modes (selected by shape of `a`/`b`):
      - **Action-batch mode** (default): `a`/`b` are scalar or 1-D length 1
        (single action) or N_actions (batched data generation). Window shape:
        (N, seq_len_q). Attention broadcasts across heads.
      - **Per-head mode**: `a`/`b` are length-`num_key_value_heads`. Each KV
        head gets its own (a, b). Window shape: (num_kv, seq_len_q). Attention
        kernel applies the head-h weight to head-h's score accumulation.
    """

    def __init__(self, num_key_value_heads, a, b):
        super().__init__(num_key_value_heads)
        self.a = self._as_1d(a)
        self.b = self._as_1d(b)
        # Per-head mode if both a and b have length == num_kv_heads (and >1)
        self.per_head = (
            num_key_value_heads > 1
            and self.a.numel() == num_key_value_heads
            and self.b.numel() == num_key_value_heads
        )
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

    def is_per_head(self) -> bool:
        return self.per_head
