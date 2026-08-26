"""KVZip (Kim et al., NeurIPS'25): query-agnostic KV eviction via context reconstruction.

Faithful 2-phase implementation on the v5 plugin:
  1. prefill the context (NO compression -> full KV cached);
  2. reconstruction pass: re-feed "Repeat the previous context:\n" + context as queries;
     each context KV pair is scored by the MAX attention it receives from any replay
     query/head (collected in waits_attention_forward when cache._kvzip_collecting);
  3. discard the replay KV, compress each layer's CONTEXT cache to budget by that score
     (+ recent window + sinks), then greedy-decode.

NOTE: attention-BASED (not attention-free); query-agnostic + static. ~2x prefill cost.
"""
import torch

from .compress import make_cache, CompressionConfig
from .selectors.token import TokenSelector
from .selectors.base import uniform_budgets

REPEAT_PROMPT = "\n\nRepeat the previous context exactly:\n\n"


@torch.inference_mode()
def kvzip_generate(model, tokenizer, input_ids, max_new_tokens=64,
                   budget=128, recent_budget=16, n_sink=4, stop_token_ids=None):
    """Returns generated token ids (1D tensor, excluding the prompt)."""
    device = model.device
    input_ids = input_ids.to(device)
    ctx_len = input_ids.shape[1]
    nL = model.config.num_hidden_layers
    eos = tokenizer.eos_token_id
    stop = set(stop_token_ids or []) | ({eos} if eos is not None else set())

    cfg = CompressionConfig(compression_method="full", total_budget=budget,
                            recent_budget=recent_budget, n_sink=n_sink)
    model.init_cache(cfg)                       # ensure waits attn wired
    cache = make_cache(model, cfg)
    cache._kvzip = True
    cache._kvzip_collecting = False
    cache._kvzip_scores = {}

    # phase 0: prefill context (full, no compression) -> last-token logits
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    next_logits = out.logits[:, -1, :]

    # phase 1: reconstruction = ONE long prefill (append the prompt; KVZip's design).
    # Holds context+replay KV (~2x) so run on >=2 GPUs (run_kvzip.sh: gpus_per_model 2).
    # On the longest prompts (2x can OOM) -> recency fallback in phase 2.
    rep = tokenizer(REPEAT_PROMPT, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    replay = torch.cat([rep, input_ids], dim=1)
    cache._kvzip_ctx_len = ctx_len
    cache._kvzip_collecting = True
    try:
        model(input_ids=replay, past_key_values=cache, use_cache=True, logits_to_keep=1)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        cache._kvzip_scores = {}                 # OOM -> recency fallback below
    cache._kvzip_collecting = False

    # phase 2: drop replay KV + compress context by reconstruction score (or recency on OOM)
    sel = TokenSelector(uniform_budgets(nL, budget), recent_budget=recent_budget, n_sink=n_sink)
    for L in range(nL):
        layer = cache.layers[L]
        layer.keys = layer.keys[:, :, :ctx_len, :].contiguous()
        layer.values = layer.values[:, :, :ctx_len, :].contiguous()
        scores = cache._kvzip_scores.get(L)
        if scores is None:                       # reconstruction OOM'd -> recency keeps recent+sink
            B, nkv = layer.keys.shape[0], layer.keys.shape[1]
            scores = torch.arange(ctx_len, device=layer.keys.device, dtype=torch.float32
                                  ).view(1, 1, ctx_len).expand(B, nkv, ctx_len)
        idx = sel.select(L, scores, ctx_len)
        cache.compress(L, idx)
    cache._seen = ctx_len                        # true positions (RoPE) preserved post-evict

    # phase 3: greedy decode
    gen = []
    nxt = int(next_logits.argmax(-1).item())
    for _ in range(max_new_tokens):
        if nxt in stop:
            break
        gen.append(nxt)
        step = torch.tensor([[nxt]], device=device)
        out = model(input_ids=step, past_key_values=cache, use_cache=True, logits_to_keep=1)
        nxt = int(out.logits[:, -1, :].argmax(-1).item())
    return torch.tensor(gen, dtype=torch.long)
