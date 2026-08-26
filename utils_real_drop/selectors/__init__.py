"""Selectors: turn (B, num_kv, seq_len_k) scores into kept indices.

A Selector owns the per-layer budget. It is the single place where ChunkKV,
PyramidKV, and similar selection-side modifiers live, all orthogonal to the
scoring math.

Selector type is chosen by `chunk_size` modifier:
  - chunk_size <= 1 : TokenSelector (default)
  - chunk_size  > 1 : ChunkSelector

Per-layer budget shape is chosen by `pyramid_kv` modifier:
  - pyramid_kv = False : uniform budget across layers
  - pyramid_kv = True  : linearly decreasing per-layer budget (PyramidKV)

ChunkKV-only modifier:
  - chunk_group_size > 1 : layer-group index sharing (LIR). The first layer of
    each group runs selection; the rest reuse those indices and skip the score
    path. Only applied when chunk_size > 1.
"""
from typing import Optional

from .base import Selector, uniform_budgets, pyramid_budgets
from .token import TokenSelector
from .chunk import ChunkSelector
from .oracle import OracleSelector


def build_selector(compression_config, num_layers: int) -> Optional[Selector]:
    """Pick budget shape + selector class from the compression config."""
    if compression_config is None:
        return None
    method = getattr(compression_config, "compression_method", None)
    if method in (None, "full"):
        return None

    chunk_size = int(getattr(compression_config, "chunk_size", 0) or 0)
    chunk_group_size = int(getattr(compression_config, "chunk_group_size", 1) or 1)
    pyramid_kv = bool(getattr(compression_config, "pyramid_kv", False))
    pyramid_ratio = float(getattr(compression_config, "pyramid_ratio", 4.0) or 4.0)
    base_budget = int(compression_config.total_budget)
    recent_budget = int(getattr(compression_config, "recent_budget", 16) or 16)
    n_sink = int(getattr(compression_config, "n_sink", 0) or 0)
    ada_kv = bool(getattr(compression_config, "ada_kv", False))

    if pyramid_kv:
        budgets = pyramid_budgets(num_layers, base_budget, ratio=pyramid_ratio)
    else:
        budgets = uniform_budgets(num_layers, base_budget)

    if method == "oracle":
        oracle_indices = getattr(compression_config, "oracle_indices", None)
        return OracleSelector(
            budgets,
            recent_budget=recent_budget,
            oracle_indices=oracle_indices,
        )

    if chunk_size > 1:
        return ChunkSelector(
            budgets,
            recent_budget=recent_budget,
            chunk_size=chunk_size,
            layer_group_size=chunk_group_size,
        )
    if ada_kv:
        from .ada import AdaSelector
        return AdaSelector(budgets, recent_budget=recent_budget, n_sink=n_sink)
    return TokenSelector(budgets, recent_budget=recent_budget, n_sink=n_sink)


__all__ = [
    "Selector",
    "TokenSelector",
    "ChunkSelector",
    "OracleSelector",
    "uniform_budgets",
    "pyramid_budgets",
    "build_selector",
]
