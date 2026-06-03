"""Memory-efficient attention with optional KV-compression score accumulation.

This module is model-agnostic: it knows nothing about Llama, Qwen, etc. It takes
already-projected (and already-RoPE'd, already-repeat_kv'd) Q/K/V plus an
optional `Scorer`, and returns the attention output along with the accumulated
per-key scores (or `None` if no compression). Selection from those scores is
the cache + selector's responsibility, not attention's.

Two paths:
  * Fast path (no scorer or scorer already prefilled): SDPA with `is_causal`.
  * Score-accumulating path: Q-tiled single-pass. Only the Q dimension is
    chunked; K/V are kept whole so softmax is exact per block and both output
    and compression scores are computed in a single pass over K.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .scorers import Scorer


def compressed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scorer: Optional[Scorer],
    attn_mask: Optional[torch.Tensor],
    head_dim: int,
    q_block_size: int = 128,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run attention. Returns (output, scores).

    `scores` is the accumulated importance vector in KV-head space, shape
    [B, num_kv, seq_len_k] in fp32, or None if no compression should be applied
    for this call. The cache turns scores into kept indices via its Selector.
    """
    batch_size, num_heads, seq_len_q, _ = query.shape
    _, _, seq_len_k, _ = key.shape
    sm_scale = 1.0 / math.sqrt(head_dim)
    device = query.device

    if scorer is None or not scorer.needs_scores():
        is_causal = attn_mask is None and seq_len_q > 1
        out = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=None if is_causal else attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        return out, None

    num_kv = scorer.num_key_value_heads
    if num_kv > 0 and num_heads % num_kv == 0:
        group = num_heads // num_kv
    else:
        num_kv = num_heads
        group = 1

    # Pass query/key to prepare_prefill so policy-driven scorers can compute snap.
    scorer.prepare_prefill(seq_len_q, device, query.dtype, query=query, key=key, num_kv=num_kv)

    # Fast path for scorers that pre-compute scores without the Q-tiled loop
    # (e.g. TriAttentionScorer). Uses FlashAttention for the attention output.
    precomp = getattr(scorer, "_precomputed_scores", None)
    if precomp is not None:
        is_causal = attn_mask is None and seq_len_q > 1
        out = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=None if is_causal else attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        scorer.finalize_prefill()
        return out, precomp

    acc_scores = torch.zeros(
        (batch_size, num_kv, seq_len_k), dtype=torch.float32, device=device
    )
    output = torch.empty_like(query)

    k_pos = torch.arange(seq_len_k, device=device)
    q_offset = seq_len_k - seq_len_q

    for q_start in range(0, seq_len_q, q_block_size):
        q_end = min(q_start + q_block_size, seq_len_q)
        qb = q_end - q_start
        q_chunk = query[:, :, q_start:q_end, :]
        q_pos = torch.arange(q_start + q_offset, q_end + q_offset, device=device)

        s = torch.matmul(q_chunk, key.transpose(2, 3)) * sm_scale

        causal = k_pos.view(1, seq_len_k) > q_pos.view(qb, 1)
        s.masked_fill_(causal.view(1, 1, qb, seq_len_k), float("-inf"))

        if attn_mask is not None:
            s = s + attn_mask[:, :, q_start:q_end, :].to(s.dtype)

        probs = F.softmax(s, dim=-1)

        output[:, :, q_start:q_end, :] = torch.matmul(
            probs.to(value.dtype), value
        )

        if hasattr(scorer, "observe_probs"):
            scorer.observe_probs(q_start, q_end, probs)

        q_weights = scorer.get_query_weights(
            q_start=q_start, q_end=q_end, device=device, dtype=probs.dtype,
        )
        if q_weights is None:
            continue

        # Per-head mode: scorer returns (num_kv, qb) where each KV head has its
        # own weight curve. Otherwise treat as (N_actions, qb) action-batch.
        per_head = bool(getattr(scorer, "is_per_head", lambda: False)())
        if q_weights.ndim == 1:
            qw_view = q_weights.view(1, 1, qb, 1)
        elif q_weights.ndim == 2:
            if per_head and q_weights.size(0) == num_kv:
                # Expand (num_kv, qb) → (num_q_heads, qb) by broadcasting within group
                qw_full = q_weights.unsqueeze(1).expand(num_kv, group, qb).reshape(num_heads, qb)
                qw_view = qw_full.view(1, num_heads, qb, 1)
            else:
                N_actions = q_weights.size(0)
                if N_actions not in (1, batch_size):
                    raise ValueError(
                        f"q_weights leading dim {N_actions} must be 1 or batch_size {batch_size}"
                    )
                qw_view = q_weights.view(N_actions, 1, qb, 1)
        else:
            raise ValueError(f"q_weights.ndim must be 1 or 2, got {q_weights.ndim}")

        weighted = probs * qw_view
        if group > 1:
            contrib = weighted.view(
                batch_size, num_kv, group, qb, seq_len_k
            ).sum(dim=(2, 3))
        else:
            contrib = weighted.sum(dim=2)
        acc_scores.add_(contrib.to(torch.float32))

    scorer.finalize_prefill()
    return output, acc_scores
