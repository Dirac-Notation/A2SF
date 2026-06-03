"""KV storage with optional compression hooks.

Owns the K/V tensors, the per-layer Scorer list, and a single Selector. The
model's attention layer calls `cache.update()` to append new K/V, then runs
`compressed_attention(...)` from `attention.py`, then optionally calls
`cache.compress(layer_idx, scores, seq_len_k)` to drop unselected tokens.

The cache reports a *logical* sequence length (the absolute number of tokens
seen) so position ids keep advancing even after physical KV tensors shrink.
"""
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache

from .scorers import build_scorers, Scorer
from .selectors import build_selector, Selector


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
        self.scorers: Optional[List[Scorer]] = build_scorers(
            compression_config=compression_config,
            num_layers=config.num_hidden_layers,
            num_kv_heads=num_kv_heads,
        )
        self.selector: Optional[Selector] = build_selector(
            compression_config=compression_config,
            num_layers=config.num_hidden_layers,
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

    def compress(
        self,
        layer_idx: int,
        scores: Optional[torch.Tensor],
        seq_len_k: Optional[int] = None,
    ) -> None:
        """Run selection on the given accumulated scores and gather KV tensors.

        scores: (B, num_kv, seq_len_k) fp32 or None. Score-free selectors
            (e.g. OracleSelector or LIR follower layers) are still invoked
            when scores is None so they can apply their own selection logic.
        seq_len_k: defaults to scores.shape[-1] when not given.
        """
        if self.selector is None:
            return
        if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
            return
        if seq_len_k is None:
            seq_len_k = (
                scores.shape[-1]
                if scores is not None
                else self.key_cache[layer_idx].shape[-2]
            )
        indices = self.selector.select(layer_idx, scores, seq_len_k)
        if indices is None:
            return
        key = self.key_cache[layer_idx]
        value = self.value_cache[layer_idx]
        gather_idx = indices.to(key.device)
        gather_idx = gather_idx.unsqueeze(-1).expand(-1, -1, -1, key.size(-1))
        self.key_cache[layer_idx] = key.gather(dim=2, index=gather_idx)
        self.value_cache[layer_idx] = value.gather(dim=2, index=gather_idx)

    # ---- access ----
    def get_scorer(self, layer_idx: int) -> Optional[Scorer]:
        if self.scorers is None:
            return None
        return self.scorers[layer_idx]


__all__ = ["CompressedKVCache"]
