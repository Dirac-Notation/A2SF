"""KV-cache compression for transformers v5 via the AttentionInterface plugin (experiment engine).

Model-agnostic: one attention function ("kvcompress") is registered into `ALL_ATTENTION_FUNCTIONS`.
Each v5 `*Attention.forward` appends the new K/V to the cache and calls it with the RoPE-applied query and the
full cached key/value. On prefill the function
  1. computes the attention output with SDPA (FlashAttention-2) on the full K/V, and
  2. scores every key with the configured scorer (`scorers/`), selects the kept indices (`selectors/`), and
     shrinks this layer's cache (`CompressedCache.compress`).
`CompressedCache.get_seq_length` reports the true number of tokens seen, so RoPE positions stay correct.

Usage:
    model = load_pipeline_model("meta-llama/Llama-3.1-8B-Instruct")
    model.init_cache(CompressionConfig(compression_method="qtrust", observation_window=16, total_budget=128,
                                       recent_budget=16, tau=1.0, kernel=True))
    out = model.generate(input_ids, max_new_tokens=64)     # a fresh compressed cache is injected per call
"""
import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from qtrust.kernels import window_score, hop2_score
from .scorers import build_scorers
from .scorers.snap import SnapScorer
from .scorers.twohop import QTrustScorer
from .selectors import build_selector


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    b, nkv, slen, hd = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    return hidden_states[:, :, None, :, :].expand(b, nkv, n_rep, slen, hd).reshape(b, nkv * n_rep, slen, hd)


class CompressionConfig(dict):
    """compression_method: full | qtrust | snap | streamingllm | keydiff | l2norm | triattention | kvzip
    total_budget, recent_budget, n_sink, observation_window (W), ada_kv, score_pool (SnapKV max pooling),
    qtrust: tau, temperature, window_one, kernel; triattention: triattention_stats."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class CompressedCache(DynamicCache):
    """DynamicCache whose `get_seq_length` reports the TRUE number of seen tokens (for RoPE positions),
    independent of how much KV was evicted. `compress(layer_idx, indices)` shrinks one layer in place."""

    def __init__(self, config=None, compression_config=None):
        super().__init__(config=config)
        self._seen = 0
        num_kv = (config.num_attention_heads if getattr(config, "num_key_value_heads", None) is None
                  else config.num_key_value_heads)
        cc = compression_config or {}
        self.scorers = build_scorers(compression_config, config.num_hidden_layers, num_kv)
        self.selector = build_selector(compression_config, config.num_hidden_layers)
        self.hop_kernel = bool(cc.get("kernel") or False)          # Q-Trust: Triton path reusing the FA2 LSE
        self.score_pool = int(cc.get("score_pool") or 0)           # SnapKV's max pooling over the key axis
        self._ada_mask = {}                                         # Ada-KV: per-layer valid mask (pad-to-max)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen += key_states.shape[-2]
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen

    def get_scorer(self, layer_idx):
        return None if self.scorers is None else self.scorers[layer_idx]

    def compress(self, layer_idx, indices):
        if indices is None:
            return
        layer = self.layers[layer_idx]
        gidx = indices.to(layer.keys.device).unsqueeze(-1).expand(-1, -1, -1, layer.keys.size(-1))
        layer.keys = layer.keys.gather(dim=2, index=gidx)
        layer.values = layer.values.gather(dim=2, index=gidx)


def _accumulate_scores(query, key, scorer, num_kv, head_dim, attn_mask, q_block_size=128):
    """PyTorch path. query [B, H, Sq, hd], key [B, num_kv, Sk, hd] -> scores [B, num_kv, Sk] fp32."""
    B, num_heads, Sq, _ = query.shape
    Sk = key.shape[2]
    group = num_heads // num_kv
    sm_scale = 1.0 / math.sqrt(head_dim)
    device = query.device
    key_rep = repeat_kv(key, group)
    acc = torch.zeros(B, num_kv, Sk, dtype=torch.float32, device=device)
    k_pos = torch.arange(Sk, device=device)
    q_offset = Sk - Sq
    qidx = scorer.query_indices() if hasattr(scorer, "query_indices") else None
    if qidx is not None:
        # selective computation: only the listed query rows are computed (causal mask by true position)
        for bs in range(0, qidx.numel(), q_block_size):
            qi = qidx[bs:bs + q_block_size].to(device)
            qb = qi.numel()
            s = torch.matmul(query[:, :, qi, :], key_rep.transpose(2, 3)) * sm_scale
            causal = k_pos.view(1, Sk) > (qi + q_offset).view(qb, 1)
            s.masked_fill_(causal.view(1, 1, qb, Sk), float("-inf"))
            if attn_mask is not None:
                s = s + attn_mask[:, :, qi, :].to(s.dtype)
            probs = F.softmax(s, dim=-1)
            w = scorer.get_query_weights_at(qi).to(probs.dtype)                 # [num_kv, qb]
            qw = w.repeat_interleave(group, dim=0).view(1, num_heads, qb, 1)
            weighted = probs * qw
            contrib = (weighted.view(B, num_kv, group, qb, Sk).sum(dim=(2, 3))
                       if group > 1 else weighted.sum(dim=2))
            acc.add_(contrib.to(torch.float32))
        return acc
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
            qw = w.repeat_interleave(group, dim=0).view(1, num_heads, qb, 1)   # per-KV-head weights
        else:
            qw = w.view(w.size(0), 1, qb, 1)
        weighted = probs * qw
        contrib = (weighted.view(B, num_kv, group, qb, Sk).sum(dim=(2, 3))
                   if group > 1 else weighted.sum(dim=2))
        acc.add_(contrib.to(torch.float32))
    return acc


def _kvzip_max_scores(query, key, num_kv, head_dim, ctx_len, q_block_size=256):
    """KVzip reconstruction scoring: max attention each CONTEXT key [0:ctx_len] receives from any replay
    query/head. query [B,H,Sq_replay,hd], key [B,num_kv,Sk,hd] -> [B, num_kv, ctx_len] fp32."""
    B, num_heads, Sq, _ = query.shape
    Sk = key.shape[2]
    group = num_heads // num_kv
    sm = 1.0 / math.sqrt(head_dim)
    device = query.device
    key_rep = repeat_kv(key, group)
    out = torch.full((B, num_kv, ctx_len), -1.0, dtype=torch.float32, device=device)
    k_pos = torch.arange(Sk, device=device)
    q_offset = Sk - Sq
    for qs in range(0, Sq, q_block_size):
        qe = min(qs + q_block_size, Sq); qb = qe - qs
        s = torch.matmul(query[:, :, qs:qe, :], key_rep.transpose(2, 3)) * sm
        qpos = torch.arange(qs + q_offset, qe + q_offset, device=device).view(qb, 1)
        s.masked_fill_((k_pos.view(1, Sk) > qpos).view(1, 1, qb, Sk), float("-inf"))
        probs = F.softmax(s, dim=-1)[:, :, :, :ctx_len].float()
        out = torch.maximum(out, probs.view(B, num_kv, group, qb, ctx_len).amax(dim=(2, 3)))
    return out


def kvcompress_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    num_kv = key.shape[1]
    num_heads = query.shape[1]
    group = num_heads // num_kv
    Sq = query.shape[2]
    head_dim = query.shape[-1]
    if scaling is None:
        scaling = head_dim ** -0.5

    key_rep = repeat_kv(key, group)
    value_rep = repeat_kv(value, group)
    cache = getattr(module, "_kv_cache", None)

    # Ada-KV decode: mask the padded (invalid) positions of the pad-to-max compressed cache
    ada_mask = None
    if Sq == 1 and cache is not None and getattr(cache, "_ada_mask", None):
        vm = cache._ada_mask.get(module.layer_idx)                      # [B, num_kv, max_b] bool
        if vm is not None:
            Skc = key.shape[2]; extra = Skc - vm.shape[2]
            full = vm if extra <= 0 else torch.cat(
                [vm, torch.ones(vm.shape[0], vm.shape[1], extra, dtype=torch.bool, device=vm.device)], dim=2)
            am = torch.zeros(full.shape[0], full.shape[1], 1, full.shape[2], dtype=query.dtype, device=query.device)
            am.masked_fill_(~full.unsqueeze(2), float("-inf"))
            ada_mask = am.repeat_interleave(group, dim=1)

    # transformers v5 passes attention_mask=None for an unpadded causal prefill: apply the causal mask here
    is_causal = attention_mask is None and Sq > 1 and ada_mask is None
    use_kernel = (is_causal and cache is not None and getattr(cache, "hop_kernel", False)
                  and getattr(cache, "selector", None) is not None
                  and isinstance(cache.get_scorer(module.layer_idx), QTrustScorer)
                  and cache.get_scorer(module.layer_idx).needs_scores())
    lse = None
    if use_kernel:
        # the same FlashAttention-2 call SDPA makes (bit-identical output), which also returns the row LSE
        out, lse = torch.ops.aten._scaled_dot_product_flash_attention(
            query, key_rep, value_rep, 0.0, True, False, scale=scaling)[:2]
    else:
        out = F.scaled_dot_product_attention(
            query, key_rep, value_rep,
            attn_mask=ada_mask if ada_mask is not None else (None if is_causal else attention_mask),
            dropout_p=0.0, is_causal=is_causal, scale=scaling)
    out = out.transpose(1, 2).contiguous()                             # [B, Sq, num_heads, hd]

    if cache is not None and getattr(cache, "_kvzip", False):
        # KVzip: never compress here; only collect the max attention during the reconstruction pass
        if getattr(cache, "_kvzip_collecting", False) and Sq > 1:
            ctx = cache._kvzip_ctx_len
            sc = _kvzip_max_scores(query, key, num_kv, head_dim, ctx)
            prev = cache._kvzip_scores.get(module.layer_idx)
            cache._kvzip_scores[module.layer_idx] = sc if prev is None else torch.maximum(prev, sc)
            lyr = cache.layers[module.layer_idx]                        # drop the replay KV right away
            lyr.keys = lyr.keys[:, :, :ctx, :]
            lyr.values = lyr.values[:, :, :ctx, :]
        return out, None

    if cache is not None and Sq > 1 and getattr(cache, "selector", None) is not None:
        layer_idx = module.layer_idx
        scorer = cache.get_scorer(layer_idx)
        selector = cache.selector
        if scorer is not None and scorer.needs_scores():
            scorer.prepare_prefill(Sq, query.device, query.dtype, query=query, key=key, num_kv=num_kv)
            direct = scorer.score_keys(query, key, num_kv)               # attention-free scorers
            if direct is None and lse is not None:
                # Q-Trust, Triton path: A(q, i) = exp(q.k*scale - LSE_q), no softmax recomputation
                s_win = window_score(query, key, lse, scaling, window=scorer.first_hop_window)
                scorer.set_first_hop(s_win)
                direct = hop2_score(query, key, lse, scorer._w, scaling, qidx=scorer.query_indices())
            if direct is None and scorer.first_hop_window:
                # Q-Trust, PyTorch path: first hop = SnapKV window score
                hop = SnapScorer(num_kv, observation_window=scorer.first_hop_window)
                hop.prepare_prefill(Sq, query.device, query.dtype)
                scorer.set_first_hop(_accumulate_scores(query, key, hop, num_kv, head_dim, attention_mask))
            scores = direct if direct is not None else scorer.normalize_scores(_accumulate_scores(
                query, key, scorer, num_kv, head_dim, attention_mask))
            scorer.finalize_prefill()
            if cache.score_pool > 1:
                k = cache.score_pool
                scores = F.max_pool1d(scores, kernel_size=k, padding=k // 2, stride=1)[..., :scores.shape[-1]]
            indices = selector.select(layer_idx, scores, key.shape[2])
            cache.compress(layer_idx, indices)
            if getattr(selector, "last_valid_mask", None) is not None:
                cache._ada_mask[layer_idx] = selector.last_valid_mask
    return out, None


ALL_ATTENTION_FUNCTIONS.register("kvcompress", kvcompress_attention_forward)


def init_cache(model, compression_config):
    """Switch the model to the "kvcompress" attention and let each attention module see the cache."""
    model.config._attn_implementation = "kvcompress"
    model._kv_compression_config = compression_config
    for module in model.modules():
        if hasattr(module, "layer_idx") and type(module).__name__.endswith("Attention") \
                and not getattr(module, "_kv_hooked", False):
            def _pre_hook(mod, args, kwargs):
                mod._kv_cache = kwargs.get("past_key_values", None)
            module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
            module._kv_hooked = True


def make_cache(model, compression_config):
    return CompressedCache(model.config, compression_config)


def attach_pipeline_api(model):
    """`model.init_cache(cfg)` remembers cfg (None = no compression); `model.generate(...)` then injects a fresh
    CompressedCache per call and renames `num_logits_to_keep` -> `logits_to_keep`."""
    if getattr(model, "_kv_pipeline_attached", False):
        return model
    model._kv_cfg = None
    _orig_generate = model.generate

    def _bound_init_cache(cfg):
        model._kv_cfg = cfg
        init_cache(model, cfg)

    def _bound_generate(*args, **kwargs):
        if "num_logits_to_keep" in kwargs:
            kwargs.setdefault("logits_to_keep", kwargs.pop("num_logits_to_keep"))
        if model._kv_cfg is not None and kwargs.get("past_key_values") is None:
            kwargs["past_key_values"] = make_cache(model, model._kv_cfg)
        return _orig_generate(*args, **kwargs)

    model.init_cache = _bound_init_cache
    model.generate = _bound_generate
    model._kv_pipeline_attached = True
    return model


def load_pipeline_model(model_path, dtype=torch.bfloat16, device_map="auto"):
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=device_map,
                                                 attn_implementation="sdpa").eval()
    return attach_pipeline_api(model)
