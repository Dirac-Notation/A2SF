import torch

from .base import CompressionPolicy


class SnapPolicy(CompressionPolicy):
    """SnapKV: only queries inside the observation window contribute to scores."""

    def __init__(self, num_key_value_heads, total_budget, observation_window, recent_budget=16):
        super().__init__(num_key_value_heads, total_budget, recent_budget)
        self.observation_window = int(observation_window)
        self._observation_start = 0

    def prepare_prefill(self, seq_len_q, device, dtype):
        self._observation_start = max(0, seq_len_q - self.observation_window)

    def get_query_weights(self, q_start, q_end, device, dtype):
        qb = q_end - q_start
        w = torch.zeros(qb, device=device, dtype=torch.float32)
        local_start = max(0, self._observation_start - q_start)
        if local_start < qb:
            w[local_start:] = 1.0
        return w

    def select(self, scores, seq_len_k):
        return self._topk_with_recent(scores, seq_len_k)
