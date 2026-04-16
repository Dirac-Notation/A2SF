"""Memory-efficient attention with optional KV-compression score accumulation.

This module is model-agnostic: it knows nothing about Llama, Qwen, etc. It takes
already-projected (and already-RoPE'd, already-repeat_kv'd) Q/K/V plus an
optional `CompressionPolicy`, and returns the attention output along with the
selection indices the policy decided to keep (or `None` if no compression).

Two paths:
  * Fast path (no policy or policy already prefilled): SDPA with `is_causal`.
  * Score-accumulating path: Q-tiled single-pass. Only the Q dimension is
    chunked; K/V are kept whole so softmax is exact per block and both output
    and compression scores are computed in a single pass over K.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .policies.base import CompressionPolicy


def compressed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    policy: Optional[CompressionPolicy],
    attn_mask: Optional[torch.Tensor],
    head_dim: int,
    q_block_size: int = 128,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run attention. Returns (output, selected_indices).

    `selected_indices` is in KV-head space, shape [B, num_kv, total_budget],
    or None if no compression should be applied for this call.
    """
    batch_size, num_heads, seq_len_q, _ = query.shape
    _, _, seq_len_k, _ = key.shape
    sm_scale = 1.0 / math.sqrt(head_dim)
    device = query.device

    # Fast path: no policy, or policy is past its prefill score-accumulation step.
    if policy is None or not policy.needs_scores():
        is_causal = attn_mask is None and seq_len_q > 1
        out = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=None if is_causal else attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        return out, None

    # Score-accumulating path (single-pass, Q-tiled only).
    policy.prepare_prefill(seq_len_q, device, query.dtype)

    num_kv = policy.num_key_value_heads
    if num_kv > 0 and num_heads % num_kv == 0:
        group = num_heads // num_kv
    else:
        num_kv = num_heads
        group = 1

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

        # [B, H, qb, Sk] — qb is small, so this is memory-efficient.
        # Keep s in q_chunk's dtype; F.softmax upcasts internally for accumulation.
        s = torch.matmul(q_chunk, key.transpose(2, 3)) * sm_scale

        # Causal mask
        causal = k_pos.view(1, seq_len_k) > q_pos.view(qb, 1)
        s.masked_fill_(causal.view(1, 1, qb, seq_len_k), float("-inf"))

        if attn_mask is not None:
            s = s + attn_mask[:, :, q_start:q_end, :].to(s.dtype)

        # Softmax is exact since K is not chunked.
        probs = F.softmax(s, dim=-1)  # [B, H, qb, Sk] in s.dtype

        # Attention output
        output[:, :, q_start:q_end, :] = torch.matmul(
            probs.to(value.dtype), value
        )

        # Score accumulation
        q_weights = policy.get_query_weights(
            q_start=q_start, q_end=q_end, device=device, dtype=probs.dtype,
        )
        if q_weights is None:
            continue

        # Fuse q-weighting with group reduction: (probs * q_w) then sum over
        # the group and qb dims in one shot, avoiding a [B, num_kv, qb, Sk]
        # intermediate. Cast to fp32 only at the final (smaller) contrib.
        weighted = probs * q_weights.view(1, 1, qb, 1)
        if group > 1:
            contrib = weighted.view(
                batch_size, num_kv, group, qb, seq_len_k
            ).sum(dim=(2, 3))
        else:
            contrib = weighted.sum(dim=2)
        acc_scores.add_(contrib.to(torch.float32))

    selected = policy.select(acc_scores, seq_len_k=seq_len_k)
    policy.finalize_prefill()
    return output, selected
