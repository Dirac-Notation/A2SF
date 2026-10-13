"""Q-Trust scores with the Triton kernels (same result as `qtrust.reference.qtrust_scores`).

The kernels need the row log-sum-exp (LSE) of the prefill attention, which FlashAttention-2 already
computes. In PyTorch it is returned by `torch.ops.aten._scaled_dot_product_flash_attention`
(see `qtrust.hf.qtrust_attention`), so the extra cost is only the two score passes.
"""
import torch

from .kernels import window_score, hop2_score
from .reference import query_weights


def qtrust_scores_fast(q, k, lse, scale, window=16, tau=1.0, window_weight="one"):
    """q [1, H, N, D], k [1, KV, N, D] (after RoPE, k NOT repeated over the GQA group),
    lse [1, H, N] fp32 from the FlashAttention-2 forward. Returns [KV, N] fp32."""
    s = window_score(q, k, lse, scale, window=window)[0]                 # first hop, [KV, N]
    w, keep = query_weights(s, window, tau, window_weight)
    # rows to visit = queries kept by at least one KV head (None = dense second hop)
    qidx = keep.any(dim=0).nonzero().flatten() if tau > 0 else None
    return hop2_score(q, k, lse, w, scale, qidx=qidx)[0]                 # second hop, [KV, N]
