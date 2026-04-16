import torch

from .base import CompressionPolicy


class A2SFPolicy(CompressionPolicy):
    """A2SF: exponential forgetting on accumulated attention scores."""

    def __init__(self, num_key_value_heads, total_budget, forgetting_factor, recent_budget=16):
        super().__init__(num_key_value_heads, total_budget, recent_budget)
        self.forgetting_factor = float(forgetting_factor)
        self._window: torch.Tensor = None  # [seq_len_q] fp32

    def prepare_prefill(self, seq_len_q, device, dtype):
        # window[q] = forgetting_factor ** (seq_len_q - 1 - q); newest token has weight 1.
        # Use a constant upper bound to avoid a device->host sync from exponents.max().
        rev = torch.arange(seq_len_q - 1, -1, -1, device=device, dtype=torch.float32)
        self._window = self.forgetting_factor ** rev

    def get_query_weights(self, q_start, q_end, device, dtype):
        return self._window[q_start:q_end].to(device=device, dtype=torch.float32)

    def select(self, scores, seq_len_k):
        return self._topk_with_recent(scores, seq_len_k)
