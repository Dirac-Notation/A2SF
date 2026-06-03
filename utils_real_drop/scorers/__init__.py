"""Per-token scorers.

A Scorer produces per-query weights used to accumulate attention into per-key
importance scores. It is intentionally budget-free and selection-free; turning
scores into kept indices is the Selector's job (see `selectors/`).

Adding a new scorer:
  1. Subclass `Scorer` in a new file.
  2. Implement `prepare_prefill` and `get_query_weights`.
  3. Register a builder in `_REGISTRY` below.
"""
from typing import List, Optional

from .base import Scorer
from .a2sf import A2SFScorer
from .snap import SnapScorer
from .sigmoid import SigmoidScorer
from .per_lh_policy import PerLHPolicyScorer, load_policies, A_VALUES, B_VALUES
from .triattention import TriAttentionScorer


def _build_a2sf(cfg, num_kv, layer_idx=None):
    return A2SFScorer(num_kv, forgetting_factor=cfg.a)


def _build_snap(cfg, num_kv, layer_idx=None):
    return SnapScorer(num_kv, observation_window=cfg.observation_window)


def _build_sigmoid(cfg, num_kv, layer_idx=None):
    # Per-(L, h) policy mode: cfg.per_lh_policy_bank present.
    bank = getattr(cfg, "per_lh_policy_bank", None)
    if bank is not None and layer_idx is not None:
        # Build dict {h: policy} for this layer
        layer_policies = {h: bank[(layer_idx, h)] for h in range(num_kv)
                            if (layer_idx, h) in bank}
        if layer_policies:
            return PerLHPolicyScorer(
                num_kv, layer_idx=layer_idx,
                policies_per_head=layer_policies,
                a_values=A_VALUES, b_values=B_VALUES,
                query_window=int(getattr(cfg, "per_lh_query_window", 16)),
                topk=int(getattr(cfg, "per_lh_topk", 64)),
                sink_mask=int(getattr(cfg, "per_lh_sink_mask", 4)),
            )
    # Per-(L, h) static lookup mode.
    per_lh_a = getattr(cfg, "per_lh_a", None)
    per_lh_b = getattr(cfg, "per_lh_b", None)
    if per_lh_a is not None and per_lh_b is not None and layer_idx is not None:
        return SigmoidScorer(num_kv, a=per_lh_a[layer_idx], b=per_lh_b[layer_idx])
    return SigmoidScorer(num_kv, a=cfg.a, b=cfg.b)


def _build_triattention(cfg, num_kv, layer_idx=None):
    stats = getattr(cfg, "triattention_stats", None)
    if stats is None:
        raise ValueError("triattention method requires cfg.triattention_stats (dict loaded from .pt file)")
    return TriAttentionScorer(num_kv, layer_idx=layer_idx, stats=stats)


_REGISTRY = {
    "a2sf": _build_a2sf,
    "snap": _build_snap,
    "sigmoid": _build_sigmoid,
    "triattention": _build_triattention,
}


def build_scorers(
    compression_config, num_layers: int, num_kv_heads: int
) -> Optional[List[Scorer]]:
    """One Scorer instance per layer. Returns None if no compression."""
    if compression_config is None:
        return None
    method = getattr(compression_config, "compression_method", None)
    if method in (None, "full"):
        return None
    if method == "oracle":
        # Oracle uses precomputed indices and skips the score path entirely.
        return None
    if method not in _REGISTRY:
        raise ValueError(
            "Unsupported compression method: {!r}. Available: {}".format(
                method, sorted(_REGISTRY.keys())
            )
        )
    builder = _REGISTRY[method]
    return [builder(compression_config, num_kv_heads, layer_idx=L)
            for L in range(num_layers)]


__all__ = [
    "Scorer",
    "A2SFScorer",
    "SnapScorer",
    "SigmoidScorer",
    "TriAttentionScorer",
    "build_scorers",
]
