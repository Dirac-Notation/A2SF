# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A2SF (Accumulative Attention Score with Forgetting) is a KV cache compression technique for LLM inference. It reduces memory by selectively retaining key-value pairs using accumulative attention scores with a forgetting mechanism. The project includes both heuristic compression methods and a Reinforcement Learning approach for learning optimal compression policies.

## Environment Setup

```bash
conda env create -n A2SF python=3.8
conda activate A2SF
pip install -r pip.txt
```

Key pinned dependencies: `transformers==4.46.2`, `datasets<4.0.0`, `sentence-transformers==2.7.0`.

## Common Commands

### Run LongBench evaluation (heuristic methods)
```bash
python longbench.py --model llama3-1b --method snap --window 16 --budget 256
python longbench_eval.py result_txt/pred/llama3-1b_snap_16_256
```

Supported `--method` values: `full`, `a2sf`, `snap`, `sigmoid`. (`h2o` was removed.)

### Run LongBench evaluation (RL agent)
```bash
python longbench_RL.py --model llama3-1b --budget 1024 --rl_checkpoint runs/a2sf_rl/policy_final.pt
python longbench_eval.py result_txt/pred/llama3-1b_sigmoid_1024_RL
```

### Train RL agent
```bash
python RL/training/run.py --model llama3-1b --epochs 2000 --save_dir runs/a2sf_rl
```

## Architecture

### KV Cache Compression (`utils_real_drop/`)
Responsibilities are split into four model-agnostic pieces plus a Llama wiring layer:

- `kv_llama.py` — `KVLlamaForCausalLM`: Custom Llama model extending HuggingFace. Entry point is `model.init_cache(compression_config)` (pass `None` for no compression). Its `LlamaAttention.forward` does: `cache.update(k, v)` → `repeat_kv` → `compressed_attention(...)` → `cache.compress(layer_idx, selected)`.
- `attention.py` — `compressed_attention(query, key, value, *, policy, attn_mask, head_dim)`: model-agnostic attention kernel. Two paths:
  - Fast path (no policy or already prefilled): `F.scaled_dot_product_attention` with `is_causal`.
  - Score-accumulating path: K-tiled online softmax (true flash-attention style, never materializes `[B,H,qb,Sk]`), followed by a second K-tiled pass that reconstructs probabilities from saved `(m, l)` to feed the policy's per-query weights. Scores are accumulated directly in KV-head space.
- `cache.py` — `CompressedKVCache(Cache)`: HuggingFace-compatible cache. Stores K/V tensors, owns the per-layer policies, exposes `update`, `compress`, `get_seq_length` (always returns the *logical* length so position ids keep advancing after compression). Knows nothing about attention math.
- `policies/` — Compression policies, all inheriting `CompressionPolicy`:
  - `base.py` — abstract base + `_topk_with_recent` helper for the "score topk + always-keep recent" pattern. `recent_budget=16` is hardcoded by design.
  - `a2sf.py` — A2SF: exponential forgetting window over query positions
  - `snap.py` — SnapKV: only queries inside the observation window contribute
  - `sigmoid.py` — Sigmoid-shaped forgetting window
  - `__init__.py` — `build_policies(compression_config, num_layers, num_kv_heads)` dispatcher with `_REGISTRY`. Adding a new method = subclass `CompressionPolicy` + register one line.

Each policy implements only `prepare_prefill`, `get_query_weights(q_start, q_end, ...)`, and `select(scores, seq_len_k)`. The attention kernel handles the rest.

### Reinforcement Learning (`RL/`)
- `a2sf_model.py` — `A2SFModel`: ties together environment, agent, and runner. `ModelConfig` defines action space (a_values, b_values).
- `agent/neural_ucb_agent.py` — `NeuralUCBAgent`: multi-metric bandit agent with separate `MetricPolicyNetwork` per evaluation metric, using UCB exploration.
- `env/env.py` — `A2SFEnv`: single-step RL environment.
- `env/encoder.py` — `AttentionEncoder`: encodes attention patterns to state features.
- `env/model_runner.py` — `A2SFModelRunner`: manages LLM inference and cache during RL episodes.
- `training/` — Training loop (`trainer.py`), config (`training_config.py`), data loading (`dataloader.py`), entry point (`run.py`).

### Evaluation
- `longbench.py` / `longbench_RL.py` — Multi-GPU evaluation pipelines (heuristic vs RL).
- `longbench_eval.py` — Scoring: F1 (QA), ROUGE (summarization), exact match (retrieval), class match (classification), fuzzy similarity (code).
- `evaluate_needle.py` — Needle-in-haystack benchmark.

### Configuration (`config/`)
- `model2path.json` — Maps model shortnames to HuggingFace model IDs.
- `dataset2maxlen.json` — Max token lengths per dataset.
- `dataset2prompt.json` — Dataset-specific prompts.
- `task2dataset.json` — Maps task categories to dataset names.

### Supported Models
Llama 3.1 8B Instruct, Llama 3.2 1B Instruct, Qwen 2.5 7B Instruct (mapped via `config/model2path.json`).
