import math
from typing import Any, Dict, List, Optional, Tuple

import torch
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

    def accumulate_scores(
        self,
        attn_probs: torch.Tensor,
        q_start: int,
        q_end: int,
        acc_scores: torch.Tensor,
    ):
        return

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

    @staticmethod
    def _reduce_to_kv_heads(acc_scores: torch.Tensor, num_key_value_heads: int) -> torch.Tensor:
        # acc_scores: [batch, num_heads, seq_len]
        if acc_scores.shape[1] == num_key_value_heads:
            return acc_scores
        if acc_scores.shape[1] % num_key_value_heads != 0:
            return acc_scores
        group_size = acc_scores.shape[1] // num_key_value_heads
        return acc_scores.view(
            acc_scores.shape[0], num_key_value_heads, group_size, acc_scores.shape[-1]
        ).sum(dim=2)

    def flash_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        head_dim: int,
        block_size: int = 128,
    ) -> torch.Tensor:
        batch_size, num_heads, seq_len_q, _ = query.shape
        _, _, seq_len_k, _ = key.shape
        sm_scale = 1.0 / math.sqrt(head_dim)

        output = torch.zeros_like(query)
        layer_compressor = (
            self.layer_compressors[layer_idx] if len(self.layer_compressors) > layer_idx else None
        )

        acc_scores = None
        if layer_compressor is not None and layer_compressor.should_accumulate_scores(seq_len_q, seq_len_k):
            layer_compressor.prepare_prefill(seq_len_q, seq_len_k, query.device, query.dtype)
            acc_scores = torch.zeros((batch_size, num_heads, seq_len_k), dtype=torch.float32, device=query.device)

        for q_start in range(0, seq_len_q, block_size):
            q_end = min(q_start + block_size, seq_len_q)
            q_chunk = query[:, :, q_start:q_end, :]

            scores = torch.matmul(q_chunk.to(torch.float32), key.transpose(2, 3).to(torch.float32)) * sm_scale
            if attn_mask is not None:
                scores = scores + attn_mask[:, :, q_start:q_end, :].to(torch.float32)

            probs = torch.softmax(scores, dim=-1).to(q_chunk.dtype)
            output[:, :, q_start:q_end] = torch.matmul(probs, value)

            if acc_scores is not None:
                layer_compressor.accumulate_scores(
                    attn_probs=probs,
                    q_start=q_start,
                    q_end=q_end,
                    acc_scores=acc_scores,
                )

        if acc_scores is not None:
            reduced_scores = self._reduce_to_kv_heads(
                acc_scores=acc_scores,
                num_key_value_heads=layer_compressor.num_key_value_heads,
            )
            selected = layer_compressor.select(reduced_scores, seq_len_k=seq_len_k)
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