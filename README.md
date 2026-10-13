# Q-Trust: request-guided query weighting for KV cache compression

Code for the method and every experiment in the paper.

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

## Repository layout

| Path | What it is |
|---|---|
| `qtrust/` | **The method**, standalone (no dependency on the rest of the repository). See below. |
| `kvcompress/` | The engine used for every experiment in the paper: one transformers-v5 attention plugin that runs Q-Trust and all baselines (H2O, TOVA, SnapKV (+ max pooling), StreamingLLM, KeyDiff, L2-norm, TriAttention, KVzip) and Ada-KV head-wise budgets. Q-Trust's kernel path imports `qtrust/kernels.py`. |
| `longbench.py`, `longbench_eval.py` | LongBench generation (multi-GPU) and scoring. |
| `experiments/` | Every other experiment in the paper (Section 2 observations, NIAH, cost, figures, bootstrap CIs). |
| `config/`, `datasets/`, `utils.py` | Model / dataset settings, the LongBench builder, model loading and chat formats. |

### The method package `qtrust/`


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

## Reproducing the paper

Run everything from the repository root. Models load from the Hugging Face Hub by the ids in
`config/model2path.json` (`llama3-8b`, `qwen2`, `mistral-7b-v03`; Llama needs an accepted license / HF token).

```bash
python datasets/make_longbench_dataset.py         # THUDM/LongBench (16 English tasks) -> datasets/longbench/*.jsonl
```

### LongBench (Tables 2, 4, 5, A.1, B.1, Figures 3, A.1)

`longbench.py` writes `result_txt/pred/<budget>/<run_name>/<dataset>.jsonl`; `longbench_eval.py <that dir>`
writes `result.json` with per-task scores and the average. Each model instance takes `--gpus_per_model` GPUs
(2 for budgets 512/1024 and for KVzip in the paper, RTX 3090); all visible GPUs are used.

```bash
M=llama3-8b; B=128
python longbench.py --model $M --budget $B --method full
python longbench.py --model $M --budget $B --method qtrust --kernel --tau 1          # Q-Trust
python longbench.py --model $M --budget $B --method snap --window 16                 # SnapKV
python longbench.py --model $M --budget $B --method snap --window 16 --score_pool 7  # SnapKV + max pooling
python longbench.py --model $M --budget $B --method snap --window 1                  # TOVA
python longbench.py --model $M --budget $B --method snap --window 32768              # H2O
python longbench.py --model $M --budget $B --method streamingllm --n_sink 4
python longbench.py --model $M --budget $B --method keydiff
python longbench.py --model $M --budget $B --method l2norm
python experiments/longbench/calibrate_triattention.py --model $M --out runs/triattention_stats/${M}_synth_stats.pt
python longbench.py --model $M --budget $B --method triattention --triattention_stats runs/triattention_stats/${M}_synth_stats.pt
python longbench.py --model $M --budget $B --method kvzip --gpus_per_model 2
python longbench.py --model $M --budget $B --method qtrust --kernel --tau 1 --ada_kv   # + Ada-KV (any method)
for d in result_txt/pred/$B/*; do python longbench_eval.py $d; done
```

- **Budgets** (Appendix B, Figure 3): `--budget 256 / 512 / 1024`.
- **Window** (Table 4): `--window 4 ... 256`; request span as the window: `--window_span last|full`
  after `python experiments/longbench/make_request_spans.py`.
- **tau** (Section 3.2 table): `--tau 0` (dense) ... `4`. **Temperature** (Table 5): `--temperature 2|4|8` with `--tau 0`.
- **Q-Trust variants.** By default `longbench.py` weights the request queries by their own `s` like every
  other query (`window_weight="s"` in `qtrust/`); this is the setting of the LongBench tables. `--window_one`
  gives them weight 1 as in Eq. (6) (`window_weight="one"`, the `qtrust/` default), the setting of the NIAH runs.
  The kernel and the PyTorch path (`--kernel` off) compute the same scores up to floating-point error.
- `--recent_budget 0` is treated as the default 16 (the behaviour all reported runs had).
- **Paired bootstrap CIs**: `python experiments/longbench/paired_bootstrap.py <run_dir_A> <run_dir_B>`.
- **Budget curves**: `python experiments/longbench/budget_curve.py --models llama3-8b --out budget_curve_llama.pdf`
  (and `--models qwen2,mistral-7b-v03`).

### Section 2: query informativeness (Llama-3.1-8B, 315 sampled LongBench prompts)

The first 20 prompts of each task, middle-truncated to 8K tokens, prompts under 1K tokens dropped.
All capture / intervention scripts take `--shard i/n` to split the prompts over GPUs.

```bash
O=experiments/observation
python $O/capture_iota.py --out runs/iota_llama3-8b                                   # iota of every query
for m in llama3-8b qwen2 mistral-7b-v03; do python $O/capture_link.py --model $m --out runs/link_$m; done
python $O/intervene.py --design quartile   --out runs/intervene_quartile               # Figure 1b
python $O/intervene.py --design percentile --out runs/intervene_percentile             # Section 2.4
python $O/intervene.py --design weights    --out runs/intervene_weights                # Section 2.4

python $O/analyze_iota.py runs/iota_llama3-8b                                          # Sections 2.2-2.3
python $O/analyze_link.py runs/link_llama3-8b runs/link_qwen2 runs/link_mistral-7b-v03 # Table 1, Section 2.4
python $O/analyze_intervene.py --design quartile runs/intervene_quartile
python $O/analyze_intervene.py --design percentile runs/intervene_percentile
python $O/analyze_intervene.py --design weights runs/intervene_weights
python $O/fig_infoquery.py --iota runs/iota_llama3-8b --quartile runs/intervene_quartile --out fig_infoquery.pdf
```

### NIAH (Section 4.3, Figure 5)

```bash
python experiments/niah/build_niah.py --haystack_dir <dir of jsonl files with a "context" field> --out datasets/niah_single.jsonl
python experiments/niah/niah_eval.py --data datasets/niah_single.jsonl --out runs/niah_llama3-8b --budget 128
python experiments/niah/fig_niah.py --data datasets/niah_single.jsonl --runs runs/niah_llama3-8b --out fig_niah.pdf
```
220 prompts: 4 lengths (4K-32K) x 11 depths x 5 needles. The paper's haystack was long source documents of
narrativeqa, gov_report, qmsum, multi_news, qasper and multifieldqa_en (not LongBench samples).

### Cost (Table 3, Section 3.2)

```bash
python experiments/cost/prefill_cost.py --mode table --docs <docs> --lens 4096,8192,16384 --out cost_1gpu.json
CUDA_VISIBLE_DEVICES=0,1 python experiments/cost/prefill_cost.py --mode table --docs <docs> --lens 32768 --out cost_32k.json
python experiments/cost/prefill_cost.py --mode tau --docs <docs> --lens 16384 --out cost_tau.json
python experiments/cost/rows_computed.py --data datasets/longbench --out rows.json
python tests/test_kernel_vs_reference.py                                               # kernel accuracy
```
`<docs>`: five documents of at least 32K tokens (a JSON list of `[name, token_ids]` or a jsonl with a
"context" field); the same documents are cut to every length.
