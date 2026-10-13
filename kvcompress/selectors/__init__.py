"""Selectors: turn (B, num_kv, seq_len_k) scores into kept indices.

A Selector owns the per-layer budget (uniform across layers):
  - TokenSelector (default): per KV head, top-k by score + always keep the most recent tokens
  - AdaSelector (ada_kv=True): Ada-KV head-adaptive budgets within each layer
"""
from typing import Optional

from .base import Selector, uniform_budgets
from .token import TokenSelector


def build_selector(compression_config, num_layers: int) -> Optional[Selector]:
    if compression_config is None:
        return None
    method = getattr(compression_config, "compression_method", None)
    if method in (None, "full"):
        return None
    base_budget = int(compression_config.total_budget)
    recent_budget = int(getattr(compression_config, "recent_budget", 16) or 16)
    n_sink = int(getattr(compression_config, "n_sink", 0) or 0)
    budgets = uniform_budgets(num_layers, base_budget)
    if bool(getattr(compression_config, "ada_kv", False)):
        from .ada import AdaSelector
        return AdaSelector(budgets, recent_budget=recent_budget, n_sink=n_sink)
    return TokenSelector(budgets, recent_budget=recent_budget, n_sink=n_sink)


__all__ = ["Selector", "TokenSelector", "uniform_budgets", "build_selector"]
