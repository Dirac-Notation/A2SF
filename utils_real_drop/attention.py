"""Memory-efficient attention with optional KV-compression score accumulation.

This module is model-agnostic: it knows nothing about Llama, Qwen, etc. It takes
already-projected (and already-RoPE'd, already-repeat_kv'd) Q/K/V plus an
optional `CompressionPolicy`, and returns the attention output along with the
selection indices the policy decided to keep (or `None` if no compression).

Two paths:
  * Fast path (no policy or policy already prefilled): SDPA with `is_causal`.
  * Score-accumulating path: K-tiled online softmax (true flash-attention style),
    so the [B,H,qb,Sk] score tensor is never materialized. A second K-tiled pass
    reconstructs probabilities from the saved (m, l) state to feed the policy's
    per-query weights, accumulating directly in KV-head space.
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
    k_block_size: int = 1024,
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

    # Score-accumulating path.
    policy.prepare_prefill(seq_len_q, device, query.dtype)

    num_kv = policy.num_key_value_heads
    if num_kv > 0 and num_heads % num_kv == 0:
        group = num_heads // num_kv
    else:
        num_kv = num_heads
        group = 1

    # Issue #2: accumulate directly in KV-head space.
    acc_scores = torch.zeros(
        (batch_size, num_kv, seq_len_k), dtype=torch.float32, device=device
    )
    output = torch.empty_like(query)

    k_pos_full = torch.arange(seq_len_k, device=device)
    q_offset = seq_len_k - seq_len_q  # new tokens occupy the last seq_len_q slots

    for q_start in range(0, seq_len_q, q_block_size):
        q_end = min(q_start + q_block_size, seq_len_q)
        qb = q_end - q_start
        q_chunk = query[:, :, q_start:q_end, :]
        q_pos = torch.arange(q_start + q_offset, q_end + q_offset, device=device)

        # Online-softmax running state. Only [B,H,qb,*] — never grows with Sk.
        m_i = torch.full(
            (batch_size, num_heads, qb, 1), float("-inf"),
            device=device, dtype=torch.float32,
        )
        l_i = torch.zeros(
            (batch_size, num_heads, qb, 1), device=device, dtype=torch.float32
        )
        o_i = torch.zeros(
            (batch_size, num_heads, qb, head_dim), device=device, dtype=torch.float32
        )

        # ---- Pass 1: streaming online softmax over K blocks. ----
        for k_start in range(0, seq_len_k, k_block_size):
            k_end = min(k_start + k_block_size, seq_len_k)
            kb = k_end - k_start
            k_chunk = key[:, :, k_start:k_end, :]
            v_chunk = value[:, :, k_start:k_end, :]

            s = torch.matmul(q_chunk, k_chunk.transpose(2, 3)) * sm_scale
            s = s.to(torch.float32)

            # Issue #6: causal mask built on the fly, only [qb, kb].
            kp = k_pos_full[k_start:k_end]
            causal = kp.view(1, kb) > q_pos.view(qb, 1)
            s.masked_fill_(causal.view(1, 1, qb, kb), float("-inf"))

            if attn_mask is not None:
                s = s + attn_mask[:, :, q_start:q_end, k_start:k_end].to(torch.float32)

            m_block = s.amax(dim=-1, keepdim=True)
            m_new = torch.maximum(m_i, m_block)
            alpha = torch.exp(m_i - m_new)
            p = torch.exp(s - m_new)
            l_i = l_i * alpha + p.sum(dim=-1, keepdim=True)
            o_i = o_i * alpha + torch.matmul(p, v_chunk.to(torch.float32))
            m_i = m_new

        output[:, :, q_start:q_end, :] = (o_i / l_i).to(query.dtype)

        # Per-query forgetting weights for this q-block (issue #3: no temporaries).
        q_weights = policy.get_query_weights(
            q_start=q_start, q_end=q_end, device=device, dtype=torch.float32,
        )
        if q_weights is None:
            continue

        # ---- Pass 2: re-stream K blocks, materialize probs only block-by-block. ----
        for k_start in range(0, seq_len_k, k_block_size):
            k_end = min(k_start + k_block_size, seq_len_k)
            kb = k_end - k_start
            k_chunk = key[:, :, k_start:k_end, :]

            s = torch.matmul(q_chunk, k_chunk.transpose(2, 3)) * sm_scale
            s = s.to(torch.float32)
            kp = k_pos_full[k_start:k_end]
            causal = kp.view(1, kb) > q_pos.view(qb, 1)
            s.masked_fill_(causal.view(1, 1, qb, kb), float("-inf"))
            if attn_mask is not None:
                s = s + attn_mask[:, :, q_start:q_end, k_start:k_end].to(torch.float32)

            probs = torch.exp(s - m_i) / l_i  # [B, H, qb, kb]

            if group > 1:
                probs = probs.view(batch_size, num_kv, group, qb, kb).sum(dim=2)

            contrib = torch.einsum("q,bhqk->bhk", q_weights, probs)
            acc_scores[:, :, k_start:k_end].add_(contrib)

    selected = policy.select(acc_scores, seq_len_k=seq_len_k)
    policy.finalize_prefill()
    return output, selected
