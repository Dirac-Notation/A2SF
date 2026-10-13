# Q-Trust: request-guided query weighting for KV cache compression

Q-Trust decides which KV entries to keep right after prefill. Like SnapKV or H2O, it scores every key
by the attention it receives from prefill queries and keeps the top `budget`. The difference is **which
queries it trusts**: the user's request (the last `W` prompt tokens) tells it which document positions
matter, and the queries at those positions are trusted to point to the keys generation will need.

## Algorithm

For one layer and one KV head (prefill of `N` tokens, `A(q, i)` = causal attention probability, summed
over the query heads that share the KV head under GQA):

1. **First hop.** Attention of the request to every key:
   `s(j) = Σ_{t ∈ last W} A(t, j)`  (this is the SnapKV score)
2. **Query weights.** Request queries get weight 1, every other query gets its first-hop score:
   `w(q) ∝ 1` if `q` is in the last `W`, else `s(q)`; normalize so that `Σ_q w(q) = 1`.
3. **Selective computation.** Drop queries with `w(q) < τ / N` (τ times the uniform weight).
   Request queries are never dropped. `τ = 0` is the dense version.
4. **Second hop.** Accumulate the weighted attention:
   `score(i) = Σ_{q ≥ i} w(q) A(q, i)  ∝  s(i) + Σ_{q ∉ last W} s(q) A(q, i)`
   The first term keeps what the request attends to directly; the second adds the keys that the
   queries of those tokens attend to (e.g., a value stated far from where the request's topic is mentioned).
5. **Selection.** Always keep the last `recent` tokens; fill the remaining `budget − recent` slots with the
   highest-scoring keys among positions `[0, N − recent)`, independently per layer and KV head.

Defaults (all experiments in the paper): `W = 16`, `recent = 16`, `τ = 1`, no attention-sink tokens
forced, no pooling. The paper's main budget is 128 KV entries per layer and KV head.

## Files (read in this order)

| File | What it is |
|---|---|
| `qtrust/reference.py` | **The algorithm in ~100 lines of plain PyTorch.** Steps 1–5 above, one function each. Read this first; it is the specification. |
| `qtrust/hf.py` | Drop-in use with Hugging Face transformers v5: an attention function registered as `"qtrust"` plus a cache (`QTrustCache`) that is compressed after prefill. Model-agnostic (Llama, Qwen2, Mistral, ...). |
| `qtrust/fast.py` | The same scores with Triton kernels (`score_fast == score_reference` up to ~1e-6). |
| `qtrust/kernels.py` | Triton kernels for the two score passes. They reuse the row log-sum-exp (LSE) that FlashAttention-2 already computes, so `A(q, i) = exp(q·k_i·scale − LSE_q)` needs no softmax recomputation. |
| `tests/test_kernel_vs_reference.py` | Checks the Triton path against the reference (random GQA inputs, dense and sparse). |
| `examples/run_qtrust.py` | Generates with full cache vs. Q-Trust on a synthetic needle-in-a-haystack prompt. |

## Quick start (transformers v5)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qtrust.hf import enable_qtrust, QTrustCache

name = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda",
                                             attn_implementation="sdpa").eval()
enable_qtrust(model)                                    # switch attention to "qtrust"

ids = tok(long_prompt, return_tensors="pt").input_ids.to(model.device)
cache = QTrustCache(model.config, budget=128, window=16, recent=16, tau=1.0)   # fresh cache per prompt
out = model.generate(ids, past_key_values=cache, max_new_tokens=64, do_sample=False)
```

```bash
python tests/test_kernel_vs_reference.py
python examples/run_qtrust.py --model meta-llama/Llama-3.1-8B-Instruct --budget 128
```

Tested with Python 3.12, PyTorch 2.8 (CUDA 12.8), transformers 5.10.2, Triton 3.4 (bundled with PyTorch).
`use_kernel=False` runs the pure-PyTorch reference (no Triton needed, slower and more memory).

## Porting Q-Trust to another framework (vLLM, SGLang, TensorRT-LLM, ...)

What the method needs from the host, per layer, right after the prefill attention:

1. **Post-RoPE queries `q [H, N, D]` and keys `k [KV, N, D]`** of the whole prompt. Use the same
   positions the attention used; scores are computed on RoPE-applied vectors.
2. **Attention probabilities `A(q, i)`**, either recomputed (`reference.py`, needs a softmax over each
   row) or recovered from the FlashAttention row LSE (`kernels.py`, no softmax). If your attention kernel
   can return the LSE, the kernel path costs two extra passes over Q·K and no extra memory beyond `[KV, N]`.
3. **GQA:** sum the attention of the `G = H / KV` query heads that share a KV head. Scores and the
   selection are per KV head, because the cache is stored per KV head.
4. **Eviction:** keep `budget` entries per layer and KV head (`select_keys`), and gather keys *and* values
   with the same indices. Different KV heads keep different positions.
5. **Positions after eviction:** the next token's RoPE position must stay `N` (the true token count), not
   the shortened cache length. `QTrustCache.get_seq_length` does this in transformers; in a paged
   engine keep the logical position counter separate from the number of stored slots.
6. **Decoding** is unchanged: attend to the kept entries plus newly generated tokens (nothing is evicted
   during decoding).

Cost: the first hop reads only the last `W` query rows (`O(W·N)`); the second hop reads the kept query rows
(`O(|Q|·N)`, about 13% of the rows at `τ = 1` on 8K-token prompts). With the kernels, prefill time grows by
2.1% / 2.4% / 3.5% / 4.6% at 4K / 8K / 16K / 32K tokens (Llama-3.1-8B, RTX 3090).

## Notes

- `window_weight="s"` is a variant in which the request queries are also weighted by their own `s(q)`
  instead of 1 (step 2). The two variants differ by less than about 0.3 points on LongBench at budget 128.
  The default `"one"` is the formulation in the paper.
- The code assumes batch size 1 and an unpadded prompt (one causal prefill), which is how all experiments
  were run. Batched serving needs per-sequence lengths in steps 1–5; nothing in the method couples sequences.
- SnapKV's max-pooling of the scores is not part of Q-Trust.
