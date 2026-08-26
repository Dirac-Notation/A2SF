import torch

from .base import Scorer


class L2NormScorer(Scorer):
    """L2-norm strategy (Devoto et al., EMNLP 2024): a low L2 norm of a key
    embedding correlates with high attention, so KEEP low-norm keys. Attention-free
    (uses only key vectors), static. key: [B, num_kv, Sk, hd]."""

    def score_keys(self, query, key, num_kv):
        return (-key.float().norm(dim=-1)).contiguous()      # low norm -> high score
