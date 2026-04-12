"""KV cache compression policies.

Adding a new method:
  1. Subclass `CompressionPolicy` in a new file under this package.
  2. Implement `prepare_prefill`, `get_query_weights`, `select`.
  3. Register it in `_REGISTRY` below.

`build_policies(compression_config, num_layers, num_kv_heads)` returns a per-layer
list of policies, or `None` for the no-compression / "full" case.
"""
from typing import List, Optional

from .base import CompressionPolicy
from .a2sf import A2SFPolicy
from .snap import SnapPolicy
from .sigmoid import SigmoidPolicy


def _build_a2sf(cfg, num_kv):
    return A2SFPolicy(num_kv, cfg.total_budget, cfg.forgetting_factor)


def _build_snap(cfg, num_kv):
    return SnapPolicy(num_kv, cfg.total_budget, cfg.observation_window)


def _build_sigmoid(cfg, num_kv):
    recent_budget = int(0.125 * cfg.total_budget)
    return SigmoidPolicy(num_kv, cfg.total_budget, cfg.a, cfg.b, recent_budget=recent_budget)


_REGISTRY = {
    "a2sf": _build_a2sf,
    "snap": _build_snap,
    "sigmoid": _build_sigmoid,
}


def build_policies(
    compression_config, num_layers: int, num_kv_heads: int
) -> Optional[List[CompressionPolicy]]:
    if compression_config is None:
        return None
    method = getattr(compression_config, "compression_method", None)
    if method in (None, "full"):
        return None
    if method not in _REGISTRY:
        raise ValueError(
            "Unsupported compression method: {!r}. Available: {}".format(
                method, sorted(_REGISTRY.keys())
            )
        )
    builder = _REGISTRY[method]
    return [builder(compression_config, num_kv_heads) for _ in range(num_layers)]


__all__ = [
    "CompressionPolicy",
    "A2SFPolicy",
    "SnapPolicy",
    "SigmoidPolicy",
    "build_policies",
]
