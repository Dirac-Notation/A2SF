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
- `kv_llama.py` — `KVLlamaForCausalLM`: Custom Llama model extending HuggingFace with configurable KV cache compression. Entry point is `model.init_cache(use_compression, select_budget, recent_budget)`.
- `kv_cache/` — Compressor implementations, all inheriting `BaseCompressor`:
  - `a2sf_cache.py` — A2SF: exponential forgetting factor for temporal weighting of attention scores
  - `h2o_cache.py` — H2O: simple score accumulation on prefill then selection
  - `sigmoid_cache.py` — Sigmoid-based compression
  - `snap_cache.py` — SnapKV compression
- All methods balance `select_budget` (scored token selection) vs `recent_budget` (always-keep recent tokens).

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
