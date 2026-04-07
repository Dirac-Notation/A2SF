"""KV storage with optional compression hooks.

This class is a thin HF `Cache` implementation. It owns the K/V tensors and the
per-layer compression policies, but it does NOT run attention. The model's
attention layer calls `cache.update()` to append new K/V, then runs
`compressed_attention(...)` from `attention.py`, then optionally calls
`cache.compress(layer_idx, indices)` to drop unselected tokens.

The cache reports a *logical* sequence length (the absolute number of tokens
seen) so position ids keep advancing even after physical KV tensors shrink.
"""
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache

from .policies import build_policies
from .policies.base import CompressionPolicy


class CompressedKVCache(Cache):
    def __init__(
        self,
        *,
        config,
        compression_config=None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        num_kv_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )
        self.policies: Optional[List[CompressionPolicy]] = build_policies(
            compression_config=compression_config,
            num_layers=config.num_hidden_layers,
            num_kv_heads=num_kv_heads,
        )

    # ---- HF Cache interface ----
    def __len__(self) -> int:
        return len(self.key_cache)

    def __iter__(self):
        for i in range(len(self)):
            yield self.key_cache[i], self.value_cache[i]

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self):
            raise KeyError(layer_idx)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        # Logical (un-compressed) length, used for position ids.
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        return self.get_seq_length(layer_idx)

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    # ---- mutation ----
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def compress(self, layer_idx: int, selected_indices: Optional[torch.Tensor]) -> None:
        if selected_indices is None:
            return
        if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
            return
        key = self.key_cache[layer_idx]
        value = self.value_cache[layer_idx]
        gather_idx = selected_indices.to(key.device)
        gather_idx = gather_idx.unsqueeze(-1).expand(-1, -1, -1, key.size(-1))
        self.key_cache[layer_idx] = key.gather(dim=2, index=gather_idx)
        self.value_cache[layer_idx] = value.gather(dim=2, index=gather_idx)

    # ---- policy access ----
    def get_policy(self, layer_idx: int) -> Optional[CompressionPolicy]:
        if self.policies is None:
            return None
        return self.policies[layer_idx]


__all__ = ["CompressedKVCache"]
