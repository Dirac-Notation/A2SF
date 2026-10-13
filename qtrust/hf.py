"""Q-Trust for Hugging Face transformers (v5 attention interface), model-agnostic.

transformers v5 lets a model use any function registered in `ALL_ATTENTION_FUNCTIONS`. Each attention
module first appends the new K/V to the cache (`past_key_values.update`) and then calls the registered
function with the RoPE-applied query and the full cached key/value. We register "qtrust", which

  1. computes the attention output exactly as SDPA does (FlashAttention-2), and
  2. on prefill, scores the keys with Q-Trust and shrinks this layer's cache to `budget` entries.

Decoding then attends only to the kept KV. `QTrustCache.get_seq_length` returns the true number of
tokens seen, so RoPE positions stay correct after eviction.

Usage:
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa")
    enable_qtrust(model)
    cache = QTrustCache(model.config, budget=128, window=16, recent=16, tau=1.0)
    out = model.generate(input_ids, past_key_values=cache, max_new_tokens=64, do_sample=False)

Batch size 1 and no padding (the prefill is a plain causal pass). transformers passes
attention_mask=None for such a pass, so this function applies the causal mask itself.
"""
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .reference import qtrust_scores, select_keys


class QTrustCache(DynamicCache):
    """DynamicCache that Q-Trust compresses right after prefill.

    budget : KV entries kept per layer and KV head (including the `recent` tail)
    window : W, the number of last prompt queries treated as the request (first hop)
    recent : number of last prompt tokens always kept
    tau    : selective computation threshold (0 = dense second hop)
    window_weight : "one" (paper, Eq. 6) or "s" (see qtrust.reference.query_weights)
    use_kernel    : Triton path (needs a GPU with Triton); False = plain PyTorch reference
    """

    def __init__(self, config, budget=128, window=16, recent=16, tau=1.0, window_weight="one", use_kernel=True):
        super().__init__(config=config)
        self.budget, self.window, self.recent = int(budget), int(window), int(recent)
        self.tau, self.window_weight, self.use_kernel = float(tau), window_weight, bool(use_kernel)
        self._seen = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen += key_states.shape[-2]
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen                                   # true position count, not the cached length

    def compress(self, layer_idx, indices):
        """Keep only `indices` [1, KV, budget] of this layer's keys and values."""
        layer = self.layers[layer_idx]
        gidx = indices.unsqueeze(-1).expand(-1, -1, -1, layer.keys.size(-1))
        layer.keys = layer.keys.gather(2, gidx)
        layer.values = layer.values.gather(2, gidx)


def qtrust_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """Registered as "qtrust". query [B, H, Sq, D], key/value [B, KV, Sk, D] (the whole cached sequence)."""
    B, H, Sq, D = query.shape
    KV = key.shape[1]
    G = H // KV
    scaling = D ** -0.5 if scaling is None else scaling
    cache = getattr(module, "_qtrust_cache", None)
    prefill = Sq > 1 and attention_mask is None
    compress = prefill and isinstance(cache, QTrustCache) and key.shape[2] > cache.budget
    if compress and B != 1:
        raise ValueError("Q-Trust compression supports batch size 1")

    key_rep = key.repeat_interleave(G, dim=1) if G > 1 else key
    value_rep = value.repeat_interleave(G, dim=1) if G > 1 else value
    lse = None
    if compress and cache.use_kernel:
        # the same FlashAttention-2 call SDPA makes, which also returns the row log-sum-exp
        out, lse = torch.ops.aten._scaled_dot_product_flash_attention(
            query, key_rep, value_rep, 0.0, True, False, scale=scaling)[:2]
    else:
        out = F.scaled_dot_product_attention(query, key_rep, value_rep, attn_mask=attention_mask,
                                             is_causal=prefill, scale=scaling)

    if compress:
        if lse is not None:
            from .fast import qtrust_scores_fast
            score = qtrust_scores_fast(query, key, lse, scaling, cache.window, cache.tau, cache.window_weight)
        else:
            score = qtrust_scores(query[0], key[0], cache.window, cache.tau, cache.window_weight, scaling)
        idx = select_keys(score, cache.budget, cache.recent)            # [KV, budget]
        cache.compress(module.layer_idx, idx.unsqueeze(0))
    return out.transpose(1, 2).contiguous(), None


ALL_ATTENTION_FUNCTIONS.register("qtrust", qtrust_attention)


def enable_qtrust(model):
    """Switch every attention module to "qtrust" and let it see the cache passed to forward/generate."""
    model.config._attn_implementation = "qtrust"
    for module in model.modules():
        is_attention = hasattr(module, "layer_idx") and type(module).__name__.endswith("Attention")
        if is_attention and not getattr(module, "_qtrust_hooked", False):
            def _grab_cache(mod, args, kwargs):
                mod._qtrust_cache = kwargs.get("past_key_values", None)
            module.register_forward_pre_hook(_grab_cache, with_kwargs=True)
            module._qtrust_hooked = True
    return model
