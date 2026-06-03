"""ChunkAttnEncoder — jointly trainable encoder for RL agent.

Architecture (per user spec):
  1. embed_tokens + input_layernorm  (frozen target model)  [L, d_model]
  2. Q/K/V proj → RoPE → Bidir Self-Attention → O proj     [L, hidden]
     RMSNorm (post)
  3. FFN (hidden → hidden*2 → hidden) + residual, RMSNorm   [L, hidden]
  4. End-aligned mean pooling, chunk_size=16                 [n_chunks, hidden]
  5. Cross-attn: last chunk as Q, all as K, RoPE             (n_chunks,) weights
  6. Sliced-Linear (end-aligned): weight[:, -n:] → (hidden,)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE cos/sin cache (non-trainable)."""

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self._cached_len = 0
        self._extend(max_seq_len)

    @torch.no_grad()
    def _extend(self, seq_len: int):
        if seq_len <= self._cached_len:
            return
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)          # (seq_len, head_dim)
        self.register_buffer("_cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("_sin", emb.sin()[None, None], persistent=False)
        self._cached_len = seq_len

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """x: (B, H, L, head_dim) | positions: (L,) int64"""
        self._extend(int(positions.max().item()) + 1)
        cos = self._cos[:, :, positions]   # (1, 1, L, head_dim)
        sin = self._sin[:, :, positions]
        return _apply_rope(x, cos, sin)


# ──────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────

class ChunkAttnEncoder(nn.Module):
    """
    Parameters
    ----------
    target_model      : frozen LLaMA model (used only for embed_tokens + layernorm)
    target_tokenizer  : corresponding tokenizer
    d_model           : target model embedding dim (2048 for 1B)
    hidden            : encoder hidden dim (128)
    n_heads           : self-attn heads (4); head_dim = hidden // n_heads = 32
    chunk_size        : mean-pool window (16 tokens)
    max_chunks        : max number of chunks = ceil(max_input_length / chunk_size) (2048)
    max_input_length  : max token length before truncation (32768)
    """

    def __init__(
        self,
        target_model,
        target_tokenizer,
        d_model: int = 2048,
        hidden: int = 128,
        n_heads: int = 4,
        chunk_size: int = 16,
        max_chunks: int = 2048,
        max_input_length: int = 32768,
        ffn_mult: int = 2,
    ):
        super().__init__()
        assert hidden % n_heads == 0
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.max_input_length = max_input_length

        # frozen references from target model
        object.__setattr__(self, "_embed_tokens",    target_model.model.embed_tokens)
        object.__setattr__(self, "_input_layernorm", target_model.model.layers[0].input_layernorm)
        object.__setattr__(self, "_tokenizer",       target_tokenizer)

        # ── Step 2: Self-attention ───────────────────────────────────
        self.q_proj = nn.Linear(d_model, hidden, bias=False)
        self.k_proj = nn.Linear(d_model, hidden, bias=False)
        self.v_proj = nn.Linear(d_model, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.sa_norm = RMSNorm(hidden)

        # ── Step 3: FFN ─────────────────────────────────────────────
        self.ff1 = nn.Linear(hidden, hidden * ffn_mult, bias=False)
        self.ff2 = nn.Linear(hidden * ffn_mult, hidden, bias=False)
        self.ff_norm = RMSNorm(hidden)

        # ── Step 5: Cross-attention ──────────────────────────────────
        self.ca_norm = RMSNorm(hidden)
        self.ca_q    = nn.Linear(hidden, hidden, bias=False)
        self.ca_k    = nn.Linear(hidden, hidden, bias=False)

        # ── Step 6: Sliced-Linear ────────────────────────────────────
        # weight[:, -n_chunks:] applied to (1, n_chunks) attn weights
        self.sliced_w = nn.Parameter(torch.empty(hidden, max_chunks))
        self.sliced_b = nn.Parameter(torch.zeros(hidden))
        nn.init.kaiming_uniform_(self.sliced_w, a=math.sqrt(5))

        # ── RoPE ─────────────────────────────────────────────────────
        # Self-attn: head_dim = hidden // n_heads
        self.rope_sa = RotaryEmbedding(self.head_dim, max_seq_len=max_input_length)
        # Cross-attn: single-head, head_dim = hidden
        self.rope_ca = RotaryEmbedding(hidden, max_seq_len=max_chunks + 1)

        # output_dim exposed for compatibility with encoder interface
        self.output_dim = hidden

    @property
    def _device(self) -> torch.device:
        return self.q_proj.weight.device

    # ── Frozen embedding extraction ──────────────────────────────────

    def _get_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns (L, d_model) float32, detached (no grad into LLaMA)."""
        with torch.no_grad():
            e = self._embed_tokens(input_ids.to(self._device))
            e = self._input_layernorm(e.to(self._input_layernorm.weight.dtype))
        return e.detach().float()   # (L, d_model)

    # ── Core forward (gradients flow here) ───────────────────────────

    def forward_embeds(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        embeds : (L, d_model)  float32, detached from target model
        returns: (hidden,)
        """
        L, device = embeds.size(0), embeds.device

        # ── Step 2: Bidir self-attention ─────────────────────────────
        q = self.q_proj(embeds)   # (L, hidden)
        k = self.k_proj(embeds)
        v = self.v_proj(embeds)

        # → (1, n_heads, L, head_dim)
        def to_heads(x):
            return x.view(L, self.n_heads, self.head_dim).permute(1, 0, 2).unsqueeze(0)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)
        pos = torch.arange(L, device=device)
        q = self.rope_sa(q, pos)
        k = self.rope_sa(k, pos)

        # Flash attention (O(L) memory, bidirectional = no causal mask)
        sa_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # (1, n_heads, L, head_dim)
        sa_out = sa_out.squeeze(0).permute(1, 0, 2).reshape(L, self.hidden)
        sa_out = self.sa_norm(self.o_proj(sa_out))              # (L, hidden), post-norm

        # ── Step 3: FFN with residual ─────────────────────────────────
        h = self.ff_norm(sa_out + self.ff2(F.gelu(self.ff1(sa_out))))  # (L, hidden)

        # ── Step 4: End-aligned mean pooling ─────────────────────────
        n_chunks = math.ceil(L / self.chunk_size)
        chunk_list = []
        for j in range(n_chunks):
            end_j   = L - j * self.chunk_size
            start_j = max(0, end_j - self.chunk_size)
            chunk_list.append(h[start_j:end_j].mean(0))        # (hidden,)
        # chunk_list[0]=most_recent → reversed → chunk_emb[0]=oldest
        chunk_emb = torch.stack(list(reversed(chunk_list)), dim=0)  # (n_chunks, hidden)

        # ── Step 5: Cross-attention (last chunk as Q) ─────────────────
        c  = self.ca_norm(chunk_emb)                            # (n_chunks, hidden)
        cq = self.ca_q(c[-1:])                                  # (1, hidden)
        ck = self.ca_k(c)                                       # (n_chunks, hidden)

        # RoPE for cross-attn (single-head: batch=1, heads=1)
        chunk_pos = torch.arange(n_chunks, device=device)
        cq_r = self.rope_ca(
            cq.unsqueeze(0).unsqueeze(0),
            torch.tensor([n_chunks - 1], device=device)
        ).squeeze(0).squeeze(0)                                 # (1, hidden)
        ck_r = self.rope_ca(
            ck.unsqueeze(0).unsqueeze(0),
            chunk_pos
        ).squeeze(0).squeeze(0)                                 # (n_chunks, hidden)

        scale_c = math.sqrt(self.hidden)
        weights = F.softmax(cq_r @ ck_r.T / scale_c, dim=-1)   # (1, n_chunks)

        # ── Step 6: Sliced-Linear (end-aligned) ───────────────────────
        w_slice = self.sliced_w[:, -n_chunks:]                  # (hidden, n_chunks)
        out = F.linear(weights, w_slice, self.sliced_b)         # (1, hidden)

        return out.squeeze(0)                                   # (hidden,)

    # ── Public API ────────────────────────────────────────────────────

    def encode_context(
        self,
        text: str,
        generation_length: int = 0,
        token_budget: int = 128,
        metric_type: Optional[str] = None,
        task_type: Optional[str] = None,
        dataset: Optional[str] = None,
        detach: bool = True,
    ) -> torch.Tensor:
        """Tokenize → embed → encode. Returns (hidden,) on CPU if detach=True."""
        ids = self._tokenizer(
            text, return_tensors="pt", padding=False, truncation=True,
            max_length=self.max_input_length, add_special_tokens=False,
        ).input_ids.squeeze(0)                                  # (L,)

        embeds = self._get_embeds(ids)                          # (L, d_model)

        if detach:
            with torch.no_grad():
                out = self.forward_embeds(embeds)
            return out.cpu()
        else:
            return self.forward_embeds(embeds)
