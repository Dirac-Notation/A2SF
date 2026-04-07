import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache


class BaseCompressor:
    """
    압축 정책 훅만 담당하는 레이어 상태/전략 객체.
    key/value 텐서 저장은 DynamicCustomCache가 직접 관리한다.
    """

    def __init__(self, num_key_value_heads: int, device: torch.device):
        self.num_key_value_heads = num_key_value_heads
        self.device = device

    def init_cache(self, compression_config, layer_idx: int):
        return

    def should_accumulate_scores(self, seq_len_q: int, seq_len_k: int) -> bool:
        return False

    def prepare_prefill(self, seq_len_q: int, seq_len_k: int, device: torch.device, dtype: torch.dtype):
        return

    def get_query_weights(
        self,
        q_start: int,
        q_end: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Return per-query forgetting weights, shape [q_end - q_start], or None to skip."""
        return None

    def select(self, scores: torch.Tensor, seq_len_k: int) -> Optional[torch.Tensor]:
        return None


def _build_layer_compressor(
    compression_method: str,
    num_key_value_heads: int,
    device: torch.device,
) -> BaseCompressor:
    if compression_method in (None, "full"):
        return BaseCompressor(num_key_value_heads=num_key_value_heads, device=device)
    if compression_method == "h2o":
        from .h2o_cache import H2OCompressor

        return H2OCompressor(num_key_value_heads=num_key_value_heads, device=device)
    if compression_method == "a2sf":
        from .a2sf_cache import A2SFCompressor

        return A2SFCompressor(num_key_value_heads=num_key_value_heads, device=device)
    if compression_method == "snap":
        from .snap_cache import SnapCompressor

        return SnapCompressor(num_key_value_heads=num_key_value_heads, device=device)
    if compression_method == "sigmoid":
        from .sigmoid_cache import SigmoidCompressor

        return SigmoidCompressor(num_key_value_heads=num_key_value_heads, device=device)
    raise ValueError(f"Unsupported compression method: {compression_method}")


class DynamicCustomCache(Cache):
    """
    transformers.DynamicCache와 동일한 key/value 관리 방식 + 정책 기반 압축 훅.
    """

    def __init__(
        self,
        layer_compressors: Optional[List[BaseCompressor]] = None,
        layer_caches: Optional[List[BaseCompressor]] = None,
        *,
        config=None,
        compression_config=None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self._seen_tokens = 0
        if layer_compressors is None and layer_caches is not None:
            layer_compressors = layer_caches
        self.layer_compressors: List[BaseCompressor] = (
            layer_compressors if layer_compressors is not None else []
        )
        # Backward compatibility alias
        self.layer_caches = self.layer_compressors
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        if not self.layer_compressors and config is not None:
            method = None if compression_config is None else compression_config.compression_method
            build_device = device if device is not None else torch.device("cpu")
            num_kv_heads = (
                config.num_attention_heads
                if getattr(config, "num_key_value_heads", None) is None
                else config.num_key_value_heads
            )
            for layer_idx in range(config.num_hidden_layers):
                layer_compressor = _build_layer_compressor(
                    compression_method=method,
                    num_key_value_heads=num_kv_heads,
                    device=build_device,
                )
                if compression_config is not None:
                    layer_compressor.init_cache(compression_config, layer_idx)
                self.layer_compressors.append(layer_compressor)

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        config=None,
        compression_config=None,
        device: Optional[torch.device] = None,
    ) -> "DynamicCustomCache":
        cache = cls(config=config, compression_config=compression_config, device=device)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append([])
                self.value_cache.append([])
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif len(self.key_cache[layer_idx]) == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if len(self.layer_compressors) > layer_idx:
            # Keep logical sequence length (absolute tokens seen), not compressed KV length.
            self.layer_compressors[layer_idx].seq_length = self._seen_tokens
            self.layer_compressors[layer_idx].device = self.key_cache[layer_idx].device

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _apply_selection(self, layer_idx: int, selected_indices: torch.Tensor):
        if selected_indices is None:
            return
        if len(self.key_cache) <= layer_idx or len(self.key_cache[layer_idx]) == 0:
            return

        key_states = self.key_cache[layer_idx]
        value_states = self.value_cache[layer_idx]
        gather_idx = selected_indices.to(key_states.device)
        gather_idx = gather_idx.unsqueeze(-1).expand(-1, -1, -1, key_states.size(-1))
        self.key_cache[layer_idx] = key_states.gather(dim=2, index=gather_idx)
        self.value_cache[layer_idx] = value_states.gather(dim=2, index=gather_idx)

        if len(self.layer_compressors) > layer_idx:
            # Selection may shrink KV tensors, but logical position index must keep increasing.
            self.layer_compressors[layer_idx].seq_length = self._seen_tokens

    def flash_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        head_dim: int,
        q_block_size: int = 128,
        k_block_size: int = 1024,
    ) -> torch.Tensor:
        """
        Memory-efficient attention with optional score accumulation for KV compression.

        - Decode / non-accumulating path: delegates to fused SDPA.
        - Score-accumulating path: K-tiled online softmax (true flash-attention style),
          so the [B,H,qb,Sk] score/probs tensor is never materialized. A second
          K-tiled pass reconstructs probabilities (using the saved m, l) to feed the
          compressor's per-query weights, accumulating directly in KV-head space.
        """
        batch_size, num_heads, seq_len_q, _ = query.shape
        _, _, seq_len_k, _ = key.shape
        sm_scale = 1.0 / math.sqrt(head_dim)

        layer_compressor = (
            self.layer_compressors[layer_idx] if len(self.layer_compressors) > layer_idx else None
        )

        need_scores = (
            layer_compressor is not None
            and layer_compressor.should_accumulate_scores(seq_len_q, seq_len_k)
        )

        if not need_scores:
            # Decode / non-accumulating path. With no padding mask and Sq>1 we can
            # let SDPA handle causal masking internally; otherwise pass the mask
            # through (Sq=1 decode is fine with attn_mask=None too).
            is_causal = attn_mask is None and seq_len_q > 1
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None if is_causal else attn_mask,
                dropout_p=0.0,
                is_causal=is_causal,
            )

        layer_compressor.prepare_prefill(seq_len_q, seq_len_k, query.device, query.dtype)

        num_kv = layer_compressor.num_key_value_heads
        if num_kv > 0 and num_heads % num_kv == 0:
            group = num_heads // num_kv
        else:
            num_kv = num_heads
            group = 1

        # Issue #2: accumulate directly in KV-head space (4x smaller for Llama3-8B).
        acc_scores = torch.zeros(
            (batch_size, num_kv, seq_len_k), dtype=torch.float32, device=query.device
        )
        output = torch.empty_like(query)

        # Precompute key positions; query positions are offset because the new
        # tokens correspond to the last seq_len_q slots of the key sequence.
        device = query.device
        k_pos_full = torch.arange(seq_len_k, device=device)
        q_offset = seq_len_k - seq_len_q

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
                (batch_size, num_heads, qb, 1), device=device, dtype=torch.float32,
            )
            o_i = torch.zeros(
                (batch_size, num_heads, qb, head_dim), device=device, dtype=torch.float32,
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
            q_weights = layer_compressor.get_query_weights(
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

                # Reduce query heads to KV heads (issue #2) before weighting.
                if group > 1:
                    probs = probs.view(batch_size, num_kv, group, qb, kb).sum(dim=2)

                # Issue #3: einsum collapses q with forgetting weights, no temp tensor.
                contrib = torch.einsum("q,bhqk->bhk", q_weights, probs)
                acc_scores[:, :, k_start:k_end].add_(contrib)

        selected = layer_compressor.select(acc_scores, seq_len_k=seq_len_k)
        self._apply_selection(layer_idx=layer_idx, selected_indices=selected)

        return output



    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        # Return logical sequence length used for position progression.
        # This must not decrease when KV tensors are compressed.
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            if len(self.key_cache[layer_idx]) != 0:
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            if len(self.value_cache[layer_idx]) != 0:
                device = self.value_cache[layer_idx].device
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor]]:
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @property
    def seen_tokens(self):
        return self._seen_tokens

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        return self.get_seq_length(layer_idx)


# Backward compatibility aliases
FullCache = BaseCompressor
_build_layer_cache = _build_layer_compressor

__all__ = [
    "DynamicCustomCache",
    "BaseCompressor",
    "FullCache",
]