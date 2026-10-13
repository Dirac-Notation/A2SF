import torch

from .base import Scorer


class SnapScorer(Scorer):
    """SnapKV: only queries inside the observation window contribute to scores."""

    def __init__(self, num_key_value_heads: int, observation_window: int):
        super().__init__(num_key_value_heads)
        self.observation_window = int(observation_window)
        self._observation_start = 0

    def prepare_prefill(self, seq_len_q, device, dtype, **kwargs):
        self._observation_start = max(0, seq_len_q - self.observation_window)

    def get_query_weights(self, q_start, q_end, device, dtype):
        qb = q_end - q_start
        w = torch.zeros(qb, device=device, dtype=torch.float32)
        local_start = max(0, self._observation_start - q_start)
        if local_start < qb:
            w[local_start:] = 1.0
        return w

    def score_query_start(self, seq_len_q: int) -> int:
        # Only the last `observation_window` queries have weight 1; the rest 0.
        return max(0, seq_len_q - self.observation_window)
