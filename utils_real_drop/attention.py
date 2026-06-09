"""Memory-efficient attention with optional KV-compression score accumulation.

Model-agnostic: takes already-projected (RoPE'd, repeat_kv'd) Q/K/V plus an
optional `Scorer`, returns the attention output and the accumulated per-key
scores (or `None`). Selection from those scores is the cache + selector's job.

Decoupled design (mirrors SnapKV's official flow):
  1. **Output** is always computed with FlashAttention-class
     `F.scaled_dot_product_attention` on the FULL K/V (exact causal attention).
  2. **Scoring** for compression is a SEPARATE pass that only revisits the
     recent query window the forgetting curve actually weights. `Scorer.
     score_query_start(seq_len_q)` returns the earliest query whose weight is
     >= SCORE_WEIGHT_EPS; earlier queries contribute negligibly and are skipped.
     The window is processed in Q-tiles (`q_block_size`): a single matmul when it
     fits one tile, memory-efficiently tiled when larger (e.g. a=0 / H2O, where
     the window spans all queries). FlashAttention cannot expose the softmax
     probabilities, hence this small extra scoring matmul.

Three paths:
  * No scorer (or already prefilled): SDPA only, no scores.
  * Precomputed-score scorer (e.g. TriAttentionScorer): SDPA output + ready scores.
  * Score-accumulating: SDPA output + windowed Q-tiled scoring.
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

    # ── Fast path: no compression scoring needed ─────────────────────────────
    if scorer is None or not scorer.needs_scores():
        is_causal = attn_mask is None and seq_len_q > 1
        out = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=None if is_causal else attn_mask,
            dropout_p=0.0, is_causal=is_causal,
        )
        return out, None

    num_kv = scorer.num_key_value_heads
    if num_kv > 0 and num_heads % num_kv == 0:
        group = num_heads // num_kv
    else:
        num_kv = num_heads
        group = 1

    # Let policy-driven scorers precompute (e.g. snap from query/key).
    scorer.prepare_prefill(seq_len_q, device, query.dtype, query=query, key=key, num_kv=num_kv)

    # ── Step 1: attention OUTPUT via FlashAttention on full K/V ───────────────
    is_causal = attn_mask is None and seq_len_q > 1
    output = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=None if is_causal else attn_mask,
        dropout_p=0.0, is_causal=is_causal,
    )

    # Precomputed-score scorers (e.g. TriAttentionScorer): scores already ready.
    precomp = getattr(scorer, "_precomputed_scores", None)
    if precomp is not None:
        scorer.finalize_prefill()
        return output, precomp

    # ── Step 2: SCORING over the bounded recent query window ─────────────────
    acc_scores = torch.zeros(
        (batch_size, num_kv, seq_len_k), dtype=torch.float32, device=device
    )
    k_pos = torch.arange(seq_len_k, device=device)
    q_offset = seq_len_k - seq_len_q

    # Skip old queries whose forgetting weight is negligible. Single matmul if the
    # window fits one tile; Q-tiled (memory-efficient) otherwise (a=0 -> all).
    q_score_start = scorer.score_query_start(seq_len_q)
    q_score_start = max(0, min(q_score_start, seq_len_q - 1))   # always >= 1 query

    for q_start in range(q_score_start, seq_len_q, q_block_size):
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

        q_weights = scorer.get_query_weights(
            q_start=q_start, q_end=q_end, device=device, dtype=probs.dtype,
        )
        if q_weights is None:
            continue

        # (qb,) single curve, or (N_actions, qb) action-batch (data generation).
        if q_weights.ndim == 1:
            qw_view = q_weights.view(1, 1, qb, 1)
        elif q_weights.ndim == 2:
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
