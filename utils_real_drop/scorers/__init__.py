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
from .snap import SnapScorer
from .waits import WaitsScorer
from .triattention import TriAttentionScorer
from .streaming import StreamingLLMScorer
from .keydiff import KeyDiffScorer
from .l2norm import L2NormScorer


def _build_streamingllm(cfg, num_kv, layer_idx=None):
    n = getattr(cfg, "n_sink", None)
    return StreamingLLMScorer(num_kv, n_sink=4 if n is None else int(n))


def _build_keydiff(cfg, num_kv, layer_idx=None):
    return KeyDiffScorer(num_kv)


def _build_l2norm(cfg, num_kv, layer_idx=None):
    return L2NormScorer(num_kv)


def _build_snap(cfg, num_kv, layer_idx=None):
    return SnapScorer(num_kv, observation_window=cfg.observation_window)


def _build_waits(cfg, num_kv, layer_idx=None):
    # per-(layer, head) table: cfg.a_heads_by_layer/b_heads_by_layer =
    # list[num_layers][num_kv]. Takes precedence over the flat variants below.
    a_lh = getattr(cfg, "a_heads_by_layer", None)
    b_lh = getattr(cfg, "b_heads_by_layer", None)
    if a_lh is not None and b_lh is not None and layer_idx is not None:
        row_a, row_b = a_lh[layer_idx], b_lh[layer_idx]
        return WaitsScorer(num_kv, a=row_a[0], b=row_b[0],
                           a_heads=row_a, b_heads=row_b)
    # N3 head-portfolio: cfg.a_heads/b_heads = per-kv-head curve lists (len num_kv).
    a_heads = getattr(cfg, "a_heads", None)
    b_heads = getattr(cfg, "b_heads", None)
    if a_heads is not None and b_heads is not None:
        return WaitsScorer(num_kv, a=a_heads[0], b=b_heads[0],
                           a_heads=a_heads, b_heads=b_heads)
    # per-layer schedule: cfg.a_schedule/b_schedule are length-num_layers lists.
    # If present, layer ℓ uses (a_schedule[ℓ], b_schedule[ℓ]); else uniform cfg.a/cfg.b.
    a_sched = getattr(cfg, "a_schedule", None)
    b_sched = getattr(cfg, "b_schedule", None)
    if a_sched is not None and b_sched is not None and layer_idx is not None:
        return WaitsScorer(num_kv, a=a_sched[layer_idx], b=b_sched[layer_idx])
    return WaitsScorer(num_kv, a=cfg.a, b=cfg.b, curve=(getattr(cfg, "curve", None) or "sigmoid"))


def _build_triattention(cfg, num_kv, layer_idx=None):
    stats = getattr(cfg, "triattention_stats", None)
    if stats is None:
        raise ValueError("triattention method requires cfg.triattention_stats (dict loaded from .pt file)")
    return TriAttentionScorer(num_kv, layer_idx=layer_idx, stats=stats)


_REGISTRY = {
    "waits": _build_waits,        # canonical: the unified sigmoid-based WAITS scorer
    "snap": _build_snap,
    "triattention": _build_triattention,
    "streamingllm": _build_streamingllm,
    "keydiff": _build_keydiff,
    "l2norm": _build_l2norm,
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
    "SnapScorer",
    "WaitsScorer",
    "TriAttentionScorer",
    "build_scorers",
]
