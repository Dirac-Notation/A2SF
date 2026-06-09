"""KV-cache compression for transformers v5 via the AttentionInterface plugin.

Model-agnostic: instead of reimplementing each model (kv_llama/kv_qwen/...), we
register ONE custom attention function ("a2sf") into `ALL_ATTENTION_FUNCTIONS`.
v5's `LlamaAttention.forward` (and every other model's) does
`past_key_values.update(k, v, layer_idx)` then calls the registered attention
function with `(module, q, k, v, attention_mask, scaling, ...)`. Our function:
  1. computes the attention OUTPUT with SDPA (flash) on the full K/V, and
  2. on prefill, accumulates per-key compression scores (reusing `scorers/`,
     windowed via `score_query_start`) and gathers the kept KV in the cache
     (`selectors/`), exactly like the 4.46.2 path — but with zero per-model code.

The scorers/ and selectors/ packages are pure torch and shared unchanged.

Usage (v5 env):
    from utils_real_drop.v5_compress import load_compressed_model, CompressionConfig
    model, tok = load_compressed_model("meta-llama/Llama-3.2-1B-Instruct")
    cfg = CompressionConfig(compression_method="snap", observation_window=16,
                            total_budget=128, recent_budget=16)
    init_cache(model, cfg)            # None for no compression
    out = model.generate(input_ids, past_key_values=CompressedCache(model.config, cfg), ...)
"""
import math
from typing import Optional

import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import DynamicCache

from .scorers import build_scorers
from .selectors import build_selector

try:
    from transformers.models.llama.modeling_llama import repeat_kv
except Exception:  # pragma: no cover
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        b, nkv, slen, hd = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hs = hidden_states[:, :, None, :, :].expand(b, nkv, n_rep, slen, hd)
        return hs.reshape(b, nkv * n_rep, slen, hd)


class CompressionConfig(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# ── compressed cache: tracks true seen-tokens separately from cached length ────
class CompressedCache(DynamicCache):
    """DynamicCache whose `get_seq_length` reports the TRUE number of seen tokens
    (for RoPE positions), independent of how much KV was evicted by compression.
    `compress(layer_idx, indices)` shrinks that layer's keys/values in place.
    """

    def __init__(self, config=None, compression_config=None):
        super().__init__(config=config)
        self._seen = 0
        num_kv = (config.num_attention_heads
                  if getattr(config, "num_key_value_heads", None) is None
                  else config.num_key_value_heads)
        self.scorers = build_scorers(compression_config, config.num_hidden_layers, num_kv)
        self.selector = build_selector(compression_config, config.num_hidden_layers)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen += key_states.shape[-2]
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen

    def get_scorer(self, layer_idx):
        return None if self.scorers is None else self.scorers[layer_idx]

    def reset_scorers(self):
        if self.scorers is not None:
            for s in self.scorers:
                s.reset()

    def compress(self, layer_idx, indices):
        if indices is None:
            return
        layer = self.layers[layer_idx]
        keys, values = layer.keys, layer.values
        gidx = indices.to(keys.device).unsqueeze(-1).expand(-1, -1, -1, keys.size(-1))
        layer.keys = keys.gather(dim=2, index=gidx)
        layer.values = values.gather(dim=2, index=gidx)


# ── score accumulation (windowed; reuses scorer curves) ───────────────────────
def _accumulate_scores(query, key, scorer, num_kv, head_dim, attn_mask, q_block_size=128):
    """query [B, H, Sq, hd], key [B, num_kv, Sk, hd] -> scores [B, num_kv, Sk] fp32."""
    B, num_heads, Sq, _ = query.shape
    Sk = key.shape[2]
    group = num_heads // num_kv
    sm_scale = 1.0 / math.sqrt(head_dim)
    device = query.device
    key_rep = repeat_kv(key, group)
    acc = torch.zeros(B, num_kv, Sk, dtype=torch.float32, device=device)
    k_pos = torch.arange(Sk, device=device)
    q_offset = Sk - Sq
    q_start = max(0, min(scorer.score_query_start(Sq), Sq - 1))
    for qs in range(q_start, Sq, q_block_size):
        qe = min(qs + q_block_size, Sq)
        qb = qe - qs
        s = torch.matmul(query[:, :, qs:qe, :], key_rep.transpose(2, 3)) * sm_scale
        causal = k_pos.view(1, Sk) > torch.arange(qs + q_offset, qe + q_offset, device=device).view(qb, 1)
        s.masked_fill_(causal.view(1, 1, qb, Sk), float("-inf"))
        if attn_mask is not None:
            s = s + attn_mask[:, :, qs:qe, :].to(s.dtype)
        probs = F.softmax(s, dim=-1)
        w = scorer.get_query_weights(qs, qe, device, probs.dtype)
        if w is None:
            continue
        qw = w.view(1, 1, qb, 1) if w.ndim == 1 else w.view(w.size(0), 1, qb, 1)
        weighted = probs * qw
        contrib = (weighted.view(B, num_kv, group, qb, Sk).sum(dim=(2, 3))
                   if group > 1 else weighted.sum(dim=2))
        acc.add_(contrib.to(torch.float32))
    return acc


# ── the registered attention function ─────────────────────────────────────────
def a2sf_attention_forward(module, query, key, value, attention_mask,
                           scaling=None, dropout=0.0, **kwargs):
    num_kv = key.shape[1]
    num_heads = query.shape[1]
    group = num_heads // num_kv
    Sq = query.shape[2]
    head_dim = query.shape[-1]
    if scaling is None:
        scaling = head_dim ** -0.5

    key_rep = repeat_kv(key, group)
    value_rep = repeat_kv(value, group)
    is_causal = attention_mask is None and Sq > 1
    out = F.scaled_dot_product_attention(
        query, key_rep, value_rep,
        attn_mask=None if is_causal else attention_mask,
        dropout_p=0.0, is_causal=is_causal, scale=scaling,
    )
    out = out.transpose(1, 2).contiguous()   # [B, Sq, num_heads, hd]

    # compression on prefill
    cache = getattr(module, "_a2sf_cache", None)
    if cache is not None and Sq > 1 and getattr(cache, "selector", None) is not None:
        layer_idx = module.layer_idx
        scorer = cache.get_scorer(layer_idx)
        selector = cache.selector
        if selector.needs_scores(layer_idx) and scorer is not None and scorer.needs_scores():
            scorer.prepare_prefill(Sq, query.device, query.dtype, query=query, key=key, num_kv=num_kv)
            scores = _accumulate_scores(query, key, scorer, num_kv, head_dim, attention_mask)
            scorer.finalize_prefill()
            indices = selector.select(layer_idx, scores, key.shape[2])
            cache.compress(layer_idx, indices)
        elif not selector.needs_scores(layer_idx):
            indices = selector.select(layer_idx, None, key.shape[2])
            cache.compress(layer_idx, indices)
    return out, None


ALL_ATTENTION_FUNCTIONS.register("a2sf", a2sf_attention_forward)


# ── wiring ────────────────────────────────────────────────────────────────────
def init_cache(model, compression_config):
    """Switch the model to the 'a2sf' attention impl and stash a pre-hook on each
    attention module that captures the cache (so the attention fn can compress it).
    Pass compression_config=None to disable compression (plain attention)."""
    model.config._attn_implementation = "a2sf"
    model._a2sf_compression_config = compression_config
    for layer in model.model.layers:
        attn = layer.self_attn
        if getattr(attn, "_a2sf_hooked", False):
            continue
        def _pre_hook(mod, args, kwargs):
            mod._a2sf_cache = kwargs.get("past_key_values", None)
            return None
        attn.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        attn._a2sf_hooked = True


def make_cache(model, compression_config):
    return CompressedCache(model.config, compression_config)


def load_compressed_model(model_path, dtype=torch.bfloat16, device_map=None):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device_map,
        attn_implementation="sdpa",
    ).eval()
    return model, tok
