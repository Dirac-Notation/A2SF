import torch

from .base import Scorer


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
