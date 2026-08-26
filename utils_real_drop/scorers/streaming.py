import torch

from .base import Scorer


class StreamingLLMScorer(Scorer):
    """StreamingLLM (Xiao et al., ICLR 2024): keep attention-sink tokens (first
    `n_sink`) + the recent window. Pure position, NO attention scores, static.
    Score = recency (later positions higher) + a large boost on the first n_sink
    sinks, so top-budget = sinks + most-recent."""

    def __init__(self, num_key_value_heads, n_sink: int = 4):
        super().__init__(num_key_value_heads)
        self.n_sink = int(n_sink)

    def score_keys(self, query, key, num_kv):
        B, Sk = key.shape[0], key.shape[2]
        pos = torch.arange(Sk, device=key.device, dtype=torch.float32)  # recency
        if self.n_sink > 0:
            pos = pos.clone()
            pos[: min(self.n_sink, Sk)] += 1e9                          # sinks always kept
        return pos.view(1, 1, Sk).expand(B, num_kv, Sk).contiguous()
