import torch
import torch.nn.functional as F

from .base import Scorer


class KeyDiffScorer(Scorer):
    """KeyDiff (Park et al., NeurIPS 2025): evict KEYS that are redundant (high
    cosine similarity to the average key) and keep geometrically diverse keys.
    Attention-free (uses only key vectors), static. key: [B, num_kv, Sk, hd]."""

    def score_keys(self, query, key, num_kv):
        k = key.float()
        anchor = k.mean(dim=2, keepdim=True)                 # [B, num_kv, 1, hd]
        cos = F.cosine_similarity(k, anchor, dim=-1)         # [B, num_kv, Sk]
        return (-cos).contiguous()                           # diverse (low sim) -> high score
