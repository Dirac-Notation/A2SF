"""MiniCrossAttn — small bidirectional self-attn + cross-attn encoder.

Reconstructed from checkpoint state_dict (the original training script was
removed in a cleanup; only weight shapes survived). Architecture matches
weight shapes of runs/mini_attn_v5/mini_attn_best.pt:

  config = {hidden:128, n_heads:4, query_window:16, max_len:32768, causal_mode:'bidir-causal'}

  down_proj : Linear(d_model=2048, hidden=128, bias=False)
  norm_in   : LayerNorm(hidden)
  self_attn_layers[0]:
    norm1 : LayerNorm(hidden)
    q/k/v : Linear(hidden, hidden, bias=False)
    o     : Linear(hidden, hidden, bias=False)
    norm2 : LayerNorm(hidden)
    ff    : Sequential(Linear(hidden, 2*hidden), GELU, Linear(2*hidden, hidden))
  distance_bias : (n_heads, 64)   — per-query learned bias added to attn logits
                  (only first query_window=16 buckets are used)
  q_proj, k_proj : Linear(hidden, hidden, bias=False)   — cross-attn projections
                   used on the last `query_window` tokens (Q) vs all tokens (K)

Forward path:
  embeds (L, d_model)
   → down_proj → norm_in
   → self-attn block (bidir) + FFN block
   → take last W=query_window tokens as Q, all tokens as K
   → cross-attn → per-token score
   → mean over n_heads → (L,) score vector
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttnBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.hidden = hidden
        self.norm1 = nn.LayerNorm(hidden)
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, hidden) → (B, L, hidden)"""
        B, L, H = x.shape
        h = self.norm1(x)
        q = self.q(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        sa = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # bidir
        sa = sa.transpose(1, 2).reshape(B, L, H)
        x = x + self.dropout(self.o(sa))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class MiniCrossAttn(nn.Module):
    """Small bidirectional encoder + cross-attention readout.

    Returns a tuple where index [0] is the (B, L) per-token importance score.
    """
    def __init__(
        self,
        embed_dim: int       = 2048,
        hidden: int          = 128,
        n_heads: int         = 4,
        query_window: int    = 16,
        dropout: float       = 0.0,
        self_attn_layers: int= 1,
        max_len: int         = 32768,
        use_distance_bias: bool = True,
        causal_mode: str     = "bidir-causal",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.query_window = query_window
        self.max_len = max_len
        self.use_distance_bias = use_distance_bias
        self.causal_mode = causal_mode

        self.down_proj = nn.Linear(embed_dim, hidden, bias=False)
        self.norm_in = nn.LayerNorm(hidden)
        self.self_attn_layers = nn.ModuleList([
            SelfAttnBlock(hidden, n_heads, dropout) for _ in range(self_attn_layers)
        ])

        # Per-query distance bias (shape (n_heads, 64); only first query_window used).
        # Indexed by query position within the last `query_window` tokens.
        if use_distance_bias:
            self.distance_bias = nn.Parameter(torch.zeros(n_heads, 64))
        else:
            self.register_parameter("distance_bias", None)

        # Cross-attention projections (W queries → all keys)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, embeds: torch.Tensor):
        """
        embeds: (B, L, embed_dim) or (L, embed_dim)
        returns: tuple (scores,) where scores is (B, L) or (L,) per-token importance
        """
        squeeze = False
        if embeds.dim() == 2:
            embeds = embeds.unsqueeze(0)
            squeeze = True
        B, L, _ = embeds.shape

        h = self.down_proj(embeds)
        h = self.norm_in(h)
        for blk in self.self_attn_layers:
            h = blk(h)                                  # (B, L, hidden)

        # Cross-attention: last W tokens as Q, all as K
        W = min(self.query_window, L)
        q = self.q_proj(h[:, -W:, :])                   # (B, W, hidden)
        k = self.k_proj(h)                              # (B, L, hidden)

        q = q.view(B, W, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, W, D)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)

        scale  = math.sqrt(self.head_dim)
        logits = (q @ k.transpose(-2, -1)) / scale       # (B, H, W, L)

        if self.use_distance_bias and self.distance_bias is not None:
            # bias[h, i] is added to all logits from query i (i ∈ [0, W))
            # Take first W slots of the 64-bucket bias
            bias = self.distance_bias[:, :W]             # (H, W)
            logits = logits + bias.view(1, self.n_heads, W, 1)

        if self.causal_mode in ("causal-causal", "bidir-causal"):
            # Query at position (L - W + i) cannot attend to keys > its own position
            # but here we ALLOW bidir (no mask) for "bidir-*" mode and only mask
            # future keys for the cross-attn queries. In practice the last W
            # tokens look at all earlier tokens — so a single causal-style mask
            # is applied only when causal_mode == "causal-causal".
            if self.causal_mode == "causal-causal":
                # mask: query i (position L - W + i) attends to keys 0..L-W+i
                q_pos = torch.arange(L - W, L, device=embeds.device)
                k_pos = torch.arange(L, device=embeds.device)
                mask  = (k_pos.unsqueeze(0) > q_pos.unsqueeze(1))   # (W, L)
                logits = logits.masked_fill(mask.view(1, 1, W, L), float("-inf"))

        attn = F.softmax(logits, dim=-1)                 # (B, H, W, L)

        # Per-token score: sum over W queries, then mean over heads
        scores = attn.sum(dim=2).mean(dim=1)             # (B, L)
        if squeeze or B == 1:
            scores = scores.squeeze(0)                   # (L,)
        return (scores,)
