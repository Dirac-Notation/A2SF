"""Per-key scorers. A Scorer decides how prefill attention is turned into per-key importance; turning scores
into kept indices is the Selector's job (see `selectors/`).

    qtrust        Q-Trust (two hops: the request's attention weights the prefill queries)
    snap          accumulated attention of the last W queries: SnapKV (W=16), TOVA (W=1), H2O (W >= prompt length)
    streamingllm  attention sinks + recent window
    keydiff       keys far from the mean key (attention-free)
    l2norm        low-norm keys (attention-free)
    triattention  calibrated trigonometric estimate of future attention (attention-free)
KVzip is not a per-key scorer of the prefill pass and lives in kvcompress/kvzip.py.
"""
from typing import List, Optional

from .base import Scorer
from .snap import SnapScorer
from .twohop import QTrustScorer
from .triattention import TriAttentionScorer
from .streaming import StreamingLLMScorer
from .keydiff import KeyDiffScorer
from .l2norm import L2NormScorer


def _build_qtrust(cfg, num_kv, layer_idx=None):
    return QTrustScorer(num_kv, observation_window=cfg.observation_window,
                        temperature=float(cfg.get("temperature") or 1.0), skip=float(cfg.get("tau") or 0.0),
                        window_one=bool(cfg.get("window_one") or False))


def _build_snap(cfg, num_kv, layer_idx=None):
    return SnapScorer(num_kv, observation_window=cfg.observation_window)


def _build_streamingllm(cfg, num_kv, layer_idx=None):
    n = getattr(cfg, "n_sink", None)
    return StreamingLLMScorer(num_kv, n_sink=4 if n is None else int(n))


def _build_keydiff(cfg, num_kv, layer_idx=None):
    return KeyDiffScorer(num_kv)


def _build_l2norm(cfg, num_kv, layer_idx=None):
    return L2NormScorer(num_kv)


def _build_triattention(cfg, num_kv, layer_idx=None):
    stats = getattr(cfg, "triattention_stats", None)
    if stats is None:
        raise ValueError("triattention needs cfg.triattention_stats (dict loaded from the calibration .pt file)")
    return TriAttentionScorer(num_kv, layer_idx=layer_idx, stats=stats)


_REGISTRY = {
    "qtrust": _build_qtrust,
    "snap": _build_snap,
    "streamingllm": _build_streamingllm,
    "keydiff": _build_keydiff,
    "l2norm": _build_l2norm,
    "triattention": _build_triattention,
}


def build_scorers(compression_config, num_layers: int, num_kv_heads: int) -> Optional[List[Scorer]]:
    """One Scorer instance per layer. Returns None if no compression."""
    if compression_config is None:
        return None
    method = getattr(compression_config, "compression_method", None)
    if method in (None, "full", "kvzip"):
        return None
    if method not in _REGISTRY:
        raise ValueError(f"Unsupported compression method: {method!r}. Available: {sorted(_REGISTRY)}")
    return [_REGISTRY[method](compression_config, num_kv_heads, layer_idx=L) for L in range(num_layers)]


__all__ = ["Scorer", "SnapScorer", "QTrustScorer", "build_scorers"]
