"""KV-cache compression for transformers v5 via the AttentionInterface plugin.

Model-agnostic: instead of one KV-compression subclass per model, we register ONE
custom attention function ("waits") into `ALL_ATTENTION_FUNCTIONS`.
v5's `LlamaAttention.forward` (and every other model's) does
`past_key_values.update(k, v, layer_idx)` then calls the registered attention
function with `(module, q, k, v, attention_mask, scaling, ...)`. Our function:
  1. computes the attention OUTPUT with SDPA (flash) on the full K/V, and
  2. on prefill, accumulates per-key compression scores (reusing `scorers/`,
     windowed via `score_query_start`) and gathers the kept KV in the cache
     (`selectors/`) — with zero per-model code.

The scorers/ and selectors/ packages are pure torch and shared unchanged.

Usage (v5 env):
    from utils_real_drop.compress import load_compressed_model, CompressionConfig
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
        # optional key-position prior curve multiplying accumulated scores before
        # selection: "band:gamma:lo:hi" or "sink:gamma:d:c" (None = off, default)
        self.key_prior = (compression_config or {}).get("key_prior") or None
        # value-aware scoring exponent p: score *= ||v_k||^p (0/None = off, default)
        self.value_weight = (compression_config or {}).get("value_weight") or None
        self._ada_mask = {}      # Ada-KV: per-layer valid mask over the padded (pad-to-max) cache

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


# ── key-position prior curve (2nd curve of the 2-curve WAITS variant) ─────────
def _key_prior_curve(spec: str, seq_len_k: int, device):
    """Parse "band:gamma:lo:hi" (boost keys in [lo*S, hi*S) by 1+gamma) or
    "sink:gamma:d:c" (early-key sigmoid boost 1 + gamma*sigmoid(-c*(k-d)))."""
    parts = spec.split(":")
    k = torch.arange(seq_len_k, dtype=torch.float32, device=device)
    if parts[0] == "band":
        gamma, lo, hi = float(parts[1]), float(parts[2]), float(parts[3])
        g = torch.ones(seq_len_k, dtype=torch.float32, device=device)
        g[int(lo * seq_len_k): int(hi * seq_len_k)] += gamma
        return g
    if parts[0] == "sink":
        gamma, d, c = float(parts[1]), float(parts[2]), float(parts[3])
        return 1.0 + gamma * torch.sigmoid(-c * (k - d))
    raise ValueError(f"unknown key_prior spec: {spec}")


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
        if w.ndim == 1:
            qw = w.view(1, 1, qb, 1)
        elif getattr(scorer, "is_per_head", lambda: False)():
            # per-kv-head weights [num_kv, qb] -> repeat per GQA group member
            qw = w.repeat_interleave(group, dim=0).view(1, num_heads, qb, 1)
        else:
            qw = w.view(w.size(0), 1, qb, 1)
        weighted = probs * qw
        contrib = (weighted.view(B, num_kv, group, qb, Sk).sum(dim=(2, 3))
                   if group > 1 else weighted.sum(dim=2))
        acc.add_(contrib.to(torch.float32))
    return acc


# ── KVZip: max cross-attention from replay queries to CONTEXT keys ─────────────
def _kvzip_max_scores(query, key, num_kv, head_dim, ctx_len, q_block_size=256):
    """KVZip reconstruction scoring. query [B,H,Sq_replay,hd], key [B,num_kv,Sk,hd]
    (Sk = ctx_len + replay-so-far). Returns max attention each CONTEXT key [0:ctx_len]
    receives from any replay query/head -> [B, num_kv, ctx_len] fp32."""
    B, num_heads, Sq, _ = query.shape
    Sk = key.shape[2]
    group = num_heads // num_kv
    sm = 1.0 / math.sqrt(head_dim)
    device = query.device
    key_rep = repeat_kv(key, group)
    out = torch.full((B, num_kv, ctx_len), -1.0, dtype=torch.float32, device=device)
    k_pos = torch.arange(Sk, device=device)
    q_offset = Sk - Sq                          # replay queries start at global pos ctx_len
    for qs in range(0, Sq, q_block_size):
        qe = min(qs + q_block_size, Sq); qb = qe - qs
        s = torch.matmul(query[:, :, qs:qe, :], key_rep.transpose(2, 3)) * sm   # [B,H,qb,Sk]
        qpos = torch.arange(qs + q_offset, qe + q_offset, device=device).view(qb, 1)
        s.masked_fill_((k_pos.view(1, Sk) > qpos).view(1, 1, qb, Sk), float("-inf"))
        probs = F.softmax(s, dim=-1)[:, :, :, :ctx_len].float()                 # attn to context keys
        blockmax = probs.view(B, num_kv, group, qb, ctx_len).amax(dim=(2, 3))   # max over group-heads & queries
        out = torch.maximum(out, blockmax)
    return out


# ── the registered attention function ─────────────────────────────────────────
def waits_attention_forward(module, query, key, value, attention_mask,
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
    cache = getattr(module, "_waits_cache", None)

    # Ada-KV decode: mask per-head padded (invalid) positions of the pad-to-max compressed cache
    ada_mask = None
    if Sq == 1 and cache is not None and getattr(cache, "_ada_mask", None):
        vm = cache._ada_mask.get(module.layer_idx)            # [B, num_kv, max_b] bool
        if vm is not None:
            Skc = key.shape[2]; extra = Skc - vm.shape[2]
            full = vm if extra <= 0 else torch.cat(
                [vm, torch.ones(vm.shape[0], vm.shape[1], extra, dtype=torch.bool, device=vm.device)], dim=2)
            am = torch.zeros(full.shape[0], full.shape[1], 1, full.shape[2], dtype=query.dtype, device=query.device)
            am.masked_fill_(~full.unsqueeze(2), float("-inf"))
            ada_mask = am.repeat_interleave(group, dim=1)     # [B, num_heads, 1, Skc]

    is_causal = attention_mask is None and Sq > 1 and ada_mask is None
    out = F.scaled_dot_product_attention(
        query, key_rep, value_rep,
        attn_mask=ada_mask if ada_mask is not None else (None if is_causal else attention_mask),
        dropout_p=0.0, is_causal=is_causal, scale=scaling,
    )
    out = out.transpose(1, 2).contiguous()   # [B, Sq, num_heads, hd]

    # compression on prefill
    if cache is not None and getattr(cache, "_kvzip", False):
        # KVZip: never auto-compress here; only collect max cross-attention during the replay.
        if getattr(cache, "_kvzip_collecting", False) and Sq > 1:
            ctx = cache._kvzip_ctx_len
            sc = _kvzip_max_scores(query, key, num_kv, head_dim, ctx)   # Q-tiled, mem-safe
            prev = cache._kvzip_scores.get(module.layer_idx)
            cache._kvzip_scores[module.layer_idx] = sc if prev is None else torch.maximum(prev, sc)
            # drop this layer's just-appended REPLAY KV immediately -> cache stays ~1x context
            # (no 2x peak), so the long reconstruction prefill fits on 1 GPU.
            lyr = cache.layers[module.layer_idx]
            lyr.keys = lyr.keys[:, :, :ctx, :]
            lyr.values = lyr.values[:, :, :ctx, :]
        return out, None
    if cache is not None and Sq > 1 and getattr(cache, "selector", None) is not None:
        layer_idx = module.layer_idx
        scorer = cache.get_scorer(layer_idx)
        selector = cache.selector
        if selector.needs_scores(layer_idx) and scorer is not None and scorer.needs_scores():
            scorer.prepare_prefill(Sq, query.device, query.dtype, query=query, key=key, num_kv=num_kv)
            direct = scorer.score_keys(query, key, num_kv)   # attention-free path (None -> attention)
            scores = direct if direct is not None else _accumulate_scores(
                query, key, scorer, num_kv, head_dim, attention_mask)
            scorer.finalize_prefill()
            if cache.key_prior is not None:
                scores = scores * _key_prior_curve(cache.key_prior, key.shape[2], scores.device)
            if cache.value_weight:
                # value-aware scoring: tokens whose value vectors are small contribute
                # little to the output even at equal attention. score *= ||v_k||^p
                p = float(cache.value_weight)
                vn = value.norm(dim=-1).to(scores.dtype)          # [B, num_kv, Sk]
                scores = scores * (vn if p == 1.0 else vn.pow(p))
            indices = selector.select(layer_idx, scores, key.shape[2])
            cache.compress(layer_idx, indices)
            if getattr(selector, "last_valid_mask", None) is not None:
                cache._ada_mask[layer_idx] = selector.last_valid_mask
        elif not selector.needs_scores(layer_idx):
            indices = selector.select(layer_idx, None, key.shape[2])
            cache.compress(layer_idx, indices)
            if getattr(selector, "last_valid_mask", None) is not None:
                cache._ada_mask[layer_idx] = selector.last_valid_mask
    return out, None


ALL_ATTENTION_FUNCTIONS.register("waits", waits_attention_forward)


# ── wiring ────────────────────────────────────────────────────────────────────
def init_cache(model, compression_config):
    """Switch the model to the 'waits' attention impl and stash a pre-hook on each
    attention module that captures the cache (so the attention fn can compress it).
    Pass compression_config=None to disable compression (plain attention)."""
    model.config._attn_implementation = "waits"
    model._waits_compression_config = compression_config
    for layer in model.model.layers:
        attn = layer.self_attn
        if getattr(attn, "_waits_hooked", False):
            continue
        def _pre_hook(mod, args, kwargs):
            mod._waits_cache = kwargs.get("past_key_values", None)
            return None
        attn.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        attn._waits_hooked = True


def make_cache(model, compression_config):
    return CompressedCache(model.config, compression_config)


def load_compressed_model(model_path, dtype=torch.bfloat16, device_map=None):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device_map,
        attn_implementation="sdpa",
    ).eval()
    return model, tok


# ── pipeline API: `model.init_cache(cfg)` + `model.generate(...)` ────────────────
#
# The eval/RL pipeline (longbench.py / RL / evaluate_needle) drives compression as
# `model.init_cache(cfg)` then `model.generate(...)`. v5's generate needs the cache
# passed via `past_key_values=...`, so we bind an `init_cache` method onto the model
# and wrap its `generate` to auto-inject the compressed cache.
def attach_pipeline_api(model):
    """Give a model the `model.init_cache(cfg)` + `model.generate(...)` interface:
    `init_cache` switches on the "waits" attention and remembers `cfg`; the wrapped
    `generate` injects a fresh CompressedCache (unless one is given) and renames v5's
    `num_logits_to_keep` -> `logits_to_keep`."""
    if getattr(model, "_waits_pipeline_attached", False):
        return model
    model._waits_cfg = None
    _orig_generate = model.generate

    def _bound_init_cache(cfg):
        model._waits_cfg = cfg
        init_cache(model, cfg)        # module-level wiring (attn impl + hooks)

    def _bound_generate(*args, **kwargs):
        if "num_logits_to_keep" in kwargs:                 # v5 renamed this kwarg
            kwargs.setdefault("logits_to_keep", kwargs.pop("num_logits_to_keep"))
        if model._waits_cfg is not None and kwargs.get("past_key_values") is None:
            kwargs["past_key_values"] = make_cache(model, model._waits_cfg)
        return _orig_generate(*args, **kwargs)

    model.init_cache = _bound_init_cache
    model.generate = _bound_generate
    model._waits_pipeline_attached = True
    return model


def load_pipeline_model(model_path, dtype=torch.bfloat16, device_map=None):
    """Load a model for the v5 path with the 4.46.2-compatible pipeline API bound.
    Returns just the model (callers load the tokenizer themselves, as in utils.py)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device_map,
        attn_implementation="sdpa",
    ).eval()
    return attach_pipeline_api(model)
