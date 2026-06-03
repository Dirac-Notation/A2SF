# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A2SF (Accumulative Attention Score with Forgetting) is a KV cache compression technique
for LLM inference. It reduces memory by selectively retaining key-value pairs using
accumulative attention scores with a forgetting mechanism. Includes both heuristic
compression methods and a Reinforcement Learning approach for learning per-prompt
sigmoid compression policies.

## Champion (current)

- Name: `RL_minattn_v5_maxo`. **LB128 = 26.95** on LLaMA-3.2-1B.
- Architecture: `NeuralUCBAgent` (13-action paired sigmoid grid + per-task linear residual
  heads + sigmoid output + 2-view MiniAttn encoder + `MLPResidualBlock × 2`). ~1.41M params.
- State features come from a frozen pretrained mini-attn encoder, not from target-model
  layer-0 attention. mini-attn ckpts: `runs/mini_attn_v5/mini_attn_best.pt` (1B),
  `runs/mini_attn_v5_8b/mini_attn_best.pt` (8B).
- Reward = MaxO (per-sample max over GT and full-cache scores).

Champion training command:
```
python RL/train.py --model llama3-1b --save_dir runs/<run_name> --budget 128 \
    --data_file datasets/training/scored/llama3-1b/train.jsonl \
    --mini_attn_ckpt runs/mini_attn_v5/mini_attn_best.pt \
    --epochs 200 --ucb_topk 4 --ucb_beta 1.0
```

`--budget 128` even when eval budget is larger — this is the documented sweet spot.

`--action_subset {full,hard}`: `hard` = 5 actions (a=0 + a=10 × {1,16,32,128}), used for
ablation Config B. `full` = all 13 actions (champion).

## Action grid

13 paired (a, b) actions, defined in `RL/a2sf_model.py` as `SIGMOID_A_VALUES`/
`SIGMOID_B_VALUES`: `(0, 1)` + cartesian({0.01, 0.1, 10} × {1, 16, 32, 128}).
`HARD_LIKE_INDICES = [0, 9, 10, 11, 12]` for the hard-only subset.

## Sigmoid scorer math

`utils_real_drop/scorers/sigmoid.py`:
`w[q] = 1 / (1 + exp(-a * (q - (N - b - 0.5))))` — real sigmoid, midpoint between
tokens N-b-1 and N-b. Matches paper §4.1. The (a, b) action grid is interpreted with
this formula in both training data generation (`script/generate_sigmoid_dataset.py`)
and inference; both must stay in sync.

## Architecture

### KV Cache Compression (`utils_real_drop/`)
- `kv_llama.py` — `KVLlamaForCausalLM`: extends HF Llama. Entry point is
  `model.init_cache(compression_config)` (pass `None` for no compression).
  `LlamaAttention.forward`: `cache.update(k, v)` → `repeat_kv` → `compressed_attention(...)`
  returns `(out, scores)` → `cache.compress(layer_idx, scores, seq_len_k)` (only on prefill).
- `attention.py` — `compressed_attention(query, key, value, *, scorer, attn_mask, head_dim)`.
  Two paths:
  - Fast path (no scorer or already prefilled): `F.scaled_dot_product_attention`.
  - Score-accumulating: Q-tiled single pass; per-key fp32 scores accumulated in KV-head
    space, weighted by `scorer.get_query_weights(...)`.
- `cache.py` — `CompressedKVCache(Cache)`: HF-compatible. Owns K/V tensors, per-layer
  scorer list, single selector. `compress` calls `selector.select` then gathers.
- `scorers/` — per-query weight curves only (budget/recent/select-free).
  - `Scorer` base, `A2SFScorer`, `SnapScorer`, `SigmoidScorer`.
  - Adding a new scorer: subclass `Scorer`, register in `_REGISTRY` in `__init__.py`.
- `selectors/` — score → kept indices. Owns layer-aware budget.
  - `Selector` base, `TokenSelector` (default top-k + always-keep recent),
    `ChunkSelector` (ChunkKV; optional LIR via `layer_group_size > 1`).
  - Budget strategies: `uniform_budgets`, `pyramid_budgets` (PyramidKV).

### Reinforcement Learning (`RL/`)
- `a2sf_model.py` — `A2SFModel` ties env + agent + runner. `ModelConfig` defines the
  default 13-action grid + encoder settings.
- `train.py` — Champion training loop: NeuralUCB top-K MSE loss + Σ⁻¹ Sherman-Morrison
  rank-1 update. Reads `training_data.jsonl` with `action_scores_*_by_budget` fields.
- `agent/neural_ucb_agent.py` — only agent class (`NeuralUCBAgent`). Hardcoded:
  paired_actions, sigmoid output, 2 views, task_cond_head, residual backbone.
- `env/env.py` — `A2SFEnv`: single-step bandit env.
- `env/encoder.py` — `AttentionEncoder` (uses target-model layer-0 attention).
- `env/mini_attn_encoder.py` — `MiniAttnEncoder` (champion path; frozen mini-attn ckpt).
  State features = stats mode (top-K positions + score + entropy/mean/std per head, 2 views).
- `env/model_runner.py` — `A2SFModelRunner`: LLM inference + cache during episodes.

### Evaluation
- `longbench.py` / `longbench_RL.py` — Multi-GPU pipelines (heuristic / RL).
  Method aliases used in eval scripts: TOVA = `--method snap --window 1`,
  SnapKV-N = `--method snap --window N`, H2O = `--method snap --window 32768`.
  Optional `--chunk_size` (ChunkKV), `--chunk_group_size` (LIR), `--pyramid_kv`,
  `--fixed_actions_json` (per-task fixed action; bypasses agent for ablation Config A).
- `longbench_eval.py` — Scoring (F1 / ROUGE / EM / class match / fuzzy sim).
- `evaluate_needle.py` — Needle-in-haystack benchmark.

### Configuration (`config/`)
- `model2path.json` — model shortname → HF model id.
- `dataset2maxlen.json`, `dataset2prompt.json` — per-dataset eval settings.
- `task2dataset.json` — 6 task families: Code Complete, Few Shot, Single-doc QA,
  Multi-doc QA, Summarization, Passage Retrieval.

### Supported Models
LLaMA 3.2 1B Instruct, LLaMA 3.1 8B Instruct, Qwen 2.5 7B Instruct (`config/model2path.json`).

## Output convention
Eval predictions go in `result_txt/pred/<budget>/<run_name>/`, never directly under
`result_txt/pred/`. `result_txt/backup/...` is read-only.

## Environment
```
conda activate A2SF
```
Pinned: `transformers==4.46.2`, `datasets<4.0.0`, `sentence-transformers==2.7.0`.
