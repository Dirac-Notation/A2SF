import torch

from .base import CompressionPolicy


class SigmoidPolicy(CompressionPolicy):
    """Sigmoid-shaped forgetting window over query positions."""

    def __init__(self, num_key_value_heads, total_budget, a, b, recent_budget=16):
        super().__init__(num_key_value_heads, total_budget, recent_budget)
        self.a = self._as_1d(a)
        self.b = self._as_1d(b)
        self._window: torch.Tensor = None  # [seq_len_q] fp32

    @staticmethod
    def _as_1d(value):
        if isinstance(value, torch.Tensor):
            return value.detach().to(torch.float32).reshape(-1)
        return torch.tensor([float(value)], dtype=torch.float32)

    def prepare_prefill(self, seq_len_q, device, dtype):
        exponents = torch.arange(seq_len_q, device=device, dtype=torch.float32)
        a = self.a.to(device=device, dtype=torch.float32).view(-1, 1)
        b = self.b.to(device=device, dtype=torch.float32).view(-1, 1)
        window = 1 / torch.exp(-a * (exponents - (seq_len_q - b - 1.0)))
        self._window = window[0]  # take the first parameter set; matches old behavior

    def get_query_weights(self, q_start, q_end, device, dtype):
        return self._window[q_start:q_end].to(device=device, dtype=torch.float32)

    def select(self, scores, seq_len_k):
        return self._topk_with_recent(scores, seq_len_k)
