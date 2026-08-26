# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WAITS is a KV cache compression technique for LLM inference. It reduces memory by
selectively retaining key-value pairs using accumulative attention scores with a
sigmoid forgetting curve. Includes heuristic compression baselines and a bandit
(routing) approach that learns a per-(task, metric) sigmoid-curve policy.
Per-PROMPT policies are a closed research branch (no learnable signal; history #55/#57).
(The conda env and repo directory are still named `A2SF` for legacy/infra reasons.)

## Champion (current) — U5b 5-curve set + (task, metric) routing

**The deployed/champion method (set 2026-07-11, history #64-65): a FIXED, model-agnostic
action set of 5 regime-representative curves + (task, metric) routing.** The set (U5b) =
the family's regime anchors: `{(0,1)=H2O-limit, (0.01,128)=gentle-window, (1,16)=mid-slope,
(10,1)=TOVA-limit, (10,16)=SnapKV-window}`. The SELECTOR is the trained agent artifact
`runs/selectors/<m>_u5b.npz` (RL/train.py --actions ... --save_agent; TRUE-bandit LinUCB,
epochs 64). Deploy tables `runs/waits_tables/waits_<m>_u5b.json` are derived caches
regenerated from the agent. Fast eval: `script/fast_store.py eval --agent ...` over the
canonical store `result_txt/backup/fast_store/`.

- Champion LB128 (= `<m>_WAITS_128` in backup): **1B 26.73 · 8B 36.99 · Qwen 41.96 ·
  Mistral 33.34** — parity with the old 13-grid (mean +0.06) at 62% fewer actions and a
  STATEABLE construction rule. Old 13-grid champion rows (`WAITS(metric)`, 26.72/36.64/
  41.92/33.32) remain in backup as the predecessor/ablation.
- Per-model best-k search values (26.89/37.03/42.11/33.45) are eval-informed — diagnostic
  only, never claim them.
- Selection granularity is PROVEN optimal at (task,metric): finer (sub-cell clusters,
  per-prompt — 14 probes) degrades monotonically; see history #60-63.
- Reproduce the champion selector: `python RL/train.py --model <m>
  --recipe datasets/training/raw/recipe_v3_<m>/train.jsonl
  --ext_scores datasets/training/raw/recipe_v3_<m>_extA/budget_128.jsonl
  --actions "0:1,0.01:128,1:16,10:1,10:16" --epochs 64 --seed 0
  --save_agent runs/selectors/<m>_u5b.npz` then
  `python script/fast_store.py eval --model <m> --agent runs/selectors/<m>_u5b.npz`.
- Reward = GT (`action_scores_gt_by_budget`), budget 128. D-I-A-R-L in the file docstring.

**Legacy per-prompt branch (research, hit a wall):** the older mini-attn prompt-encoder path
(the removed mini-attn `NeuralUCBAgent` path, recoverable from git / logs/history) reached LB128 = 26.94 on 1B
(`script/repro_2694.sh 42`), but that 26.94 is the +1σ tail of a 50-seed dist (mean 26.60 ± 0.31):
per-prompt input gives no clean gradient (62% of prompts tie across all 13 actions), so it does NOT
reliably beat the (task, metric) routing champion. Kept as a documented research branch, not the
deployed method. See `result_txt/analysis/action_proxy/ANALYSIS.md`, memories
`experiment_action_proxy` / `experiment_2694_repro_recipe`.

`--budget 128` even when eval budget is larger — the documented sweet spot.

**Active research state (2026-07):** curve-family axes are SATURATED (key-prior/exotic
query curves/per-layer/per-head all ≤ noise — proxy AND real LB; history #58). Current
direction is SHRINKING the action grid: (task,metric) routing peaks at k≈3-4 actions and
the 13-action grid adds transfer noise (index-predicted: 1B 26.82@k=4, 8B 36.97@k=3;
real-LB confirmation pending). See `result_txt/analysis/curve_screening/SCREENING_REPORT.md`.

## Action grid

13 paired (a, b) actions, defined in `RL/action_grid.py` as `SIGMOID_A_VALUES`/
`SIGMOID_B_VALUES`: `(0, 1)` + cartesian({0.01, 0.1, 10} × {1, 16, 32, 128}).
`HARD_LIKE_INDICES = [0, 9, 10, 11, 12]` for the hard-only subset.

## WAITS scorer math (unified, sigmoid-based)

`utils_real_drop/scorers/waits.py` (`WaitsScorer`, registry key `"waits"`):
`w[q] = 1 / (1 + exp(-a * (q - (N - b - 0.5))))` — real sigmoid, midpoint between
tokens N-b-1 and N-b. Matches paper §4.1. The (a, b) action grid is interpreted with
this formula in both training data generation (`RL/dataset.py`)
and inference; both must stay in sync.
The two earlier scorers (an exponential-forgetting variant + the sigmoid form) were UNIFIED into
this single `waits` scorer (2026-06-16); there are NO method aliases, `"waits"` is the only name
(callers must use it exactly). The transformers attention-impl plugin is registered under the
same name `"waits"` (`model.config._attn_implementation = "waits"`).

## Architecture

### KV Cache Compression (`utils_real_drop/`)
Runs on **transformers v5** (5.10.2, env `A2SF`). Model-agnostic: ONE registered
attention function, no per-model subclasses. (History: the old 4.46.2 per-model
`kv_{llama,qwen,opt}.py` + `cache.py` + `attention.py` were removed at history #43;
recover from git if ever needed.)
- `compress.py` — the whole mechanism. Registers `"waits"` into transformers'
  `ALL_ATTENTION_FUNCTIONS`. v5's `*Attention.forward` does `past_key_values.update(k,v,layer)`
  then calls our `waits_attention_forward(module, q, k, v, attn_mask, scaling, ...)`:
  - **output** always via `F.scaled_dot_product_attention` (flash) on `repeat_kv`'d full K/V;
  - **scoring** (prefill only): decoupled SnapKV-style windowed pass `_accumulate_scores` —
    only queries in `[scorer.score_query_start(seq), seq)` (forgetting weight ≥
    `SCORE_WEIGHT_EPS`=1e-5) are revisited, single matmul if ≤ `q_block_size` else Q-tiled
    (a=0/H2O → all queries, tiled). Steep sigmoid → tiny window → ~flash-only speed; scores
    match full-pass within ~1e-6. Then `selector.select` → `CompressedCache.compress`.
  - `CompressedCache(DynamicCache)` tracks `_seen` separately so `get_seq_length` returns the
    true token count (RoPE positions stay correct regardless of eviction); `compress(layer, idx)`
    gathers that layer's `keys`/`values`. `scorers/` + `selectors/` are reused unchanged.
- **Pipeline API.** `utils.load_model(shortname)` / `utils.load_compressed_lm(path)` →
  `compress.load_pipeline_model`, which `attach_pipeline_api`: binds `model.init_cache(cfg)`
  (`None`=no compression) and wraps `model.generate` to auto-inject a fresh `CompressedCache`
  and rename `num_logits_to_keep`→`logits_to_keep`. So longbench.py / RL / evaluate_needle /
  longbench_oracle.py / benchmark_ttft.py all just call `init_cache` + `generate`.
  Sanity/determinism: `script/verify_pipeline.py` (full/snap/sigmoid greedy dumps). The
  4.46.2↔v5 bit-identical equivalence proven during migration is logged in history #41-#43.
- `scorers/` — per-query weight curves only (budget/recent/select-free).
  - Registry: `waits` (the unified sigmoid scorer), `snap`, `triattention`,
    `streamingllm`, `keydiff`, `l2norm` (attention-free baselines). NO aliases.
  - Adding a new scorer: subclass `Scorer`, register in `_REGISTRY` in `__init__.py`.
- `selectors/` — score → kept indices. Owns layer-aware budget.
  - `Selector` base, `TokenSelector` (default top-k + always-keep recent),
    `ChunkSelector` (ChunkKV; optional LIR via `layer_group_size > 1`),
    `AdaSelector` (Ada-KV head-adaptive budgets; pad-to-max accuracy sim),
    `OracleSelector` (diagnostic).
  - Budget strategies: `uniform_budgets`, `pyramid_budgets` (PyramidKV).
  - NOTE: selector-side compositions (AdaKV/ChunkKV/PyramidKV) lift ALL scorers —
    they are comparison rows, never "our" improvement (scorer-side only counts).
- `kvzip.py` — KVZip baseline (replay-reconstruction scoring; `--method kvzip`,
  needs `--gpus_per_model 2+` for the ~2x cache peak).
- Experimental scorer-side flags (default OFF, no behavior change): `key_prior`
  (key-position prior curve; proven no-gain, kept as paper ablation) and
  `value_weight` (score ×‖v_k‖^p, value-aware scoring — under test).

### Reinforcement Learning (`RL/`) — the (task, metric) routing DIARL (6 files)
The whole RL folder is the deployed routing policy; the per-prompt/deep-agent/joint machinery was
removed 2026-06-16 (recover from git / `logs/history/54...`).
- `action_grid.py` — D/A: the 13 paired (a, b) actions (`SIGMOID_A/B_VALUES`, `HARD_LIKE_INDICES`,
  `NUM_ACTIONS`). Single source of truth for the action grid.
- `metadata.py` — I: task/metric vocab + one-hot index helpers (`*_TYPE_ORDER`, `*_to_index`,
  `resolve_task_type`) + `dataset_metric()` (dataset→metric for applying the table at eval).
- `dataset.py` — D: generate `(task, metric, action_scores_gt_by_budget)` training data
  (full-cache + 13-action compressed inference, scored). Multi-GPU. `--actions "a:b,..."`
  scores a custom action list (grid-extension), `--ada_kv` scores under Ada-KV selection.
- `model.py` — A: `RoutingNeuralUCB` — per-arm LinUCB over `phi=[task_oh|metric_oh]`
  (`phi/select/update/greedy_action/export_table`). NeuralUCB Σ⁻¹ core, linear features.
- `train.py` — R, L: bandit loop (TRUE feedback) + Sherman-Morrison update + greedy eval on the LB
  index; `--export_table` writes a `longbench.py --waits_table` block (the deploy path).
- Eval = `longbench.py --waits_table` (the routing policy is a (task,metric)→(a,b) lookup table).
  `longbench_RL.py` is a thin orchestrator: train routing → export table → run that eval.

### Evaluation
- `longbench.py` / `longbench_RL.py` — Multi-GPU pipelines (heuristic / RL).
  Baseline aliases: TOVA = `--method snap --window 1`, SnapKV = `--method snap
  --window 16` (canonical; "SnapKV-16" simplified 2026-06-16), H2O = `--method snap
  --window 32768`. Fixed WAITS action: `--method waits --window <b> --sigmoid_a <a>`.
  Optional `--chunk_size` (ChunkKV), `--chunk_group_size` (LIR), `--pyramid_kv`,
  `--ada_kv`, `--n_sink`, `--triattention_stats`, `--key_prior`, `--value_weight`.
  (`--fixed_actions_json` was removed.)
  Cross-server: `--shard_count N --shard_id i` (+ `--shard_weights "1,1,1.5,1.5"` for
  GPU-speed-proportional splitting; 3090=1.0, 4090=1.5). Each shard writes locally → rsync +
  cat-merge per-dataset jsonl on the host before scoring.
- **Per-model chat template** (`utils.build_chat_prompt` + `config/model2chat.json`): mode per
  model — `inst` = `[INST]{p}[/INST]` (Llama/Mistral; preserves existing baselines), `native` =
  `tokenizer.apply_chat_template` (Qwen ChatML etc.), `raw` = none. Unlisted → `native`. Few-shot/
  completion datasets (`utils.NO_CHAT_DATASETS`) skip wrapping. `chat_stop_strings` gives the
  matching stop. Used by longbench.py / longbench_RL.py / build_lb_index.py (replaces the old
  `"llama" in model_name` checks; evaluate_needle still uses its own model-name space).
- `longbench_eval.py` — Scoring (F1 / ROUGE / EM / class match / fuzzy sim).
- `evaluate_needle.py` — Needle-in-haystack benchmark.

### Configuration (`config/`)
- `model2path.json` — model shortname → HF model id. `model2maxlen.json` — per-model
  truncation length. `model2chat.json` — chat-template mode (see Evaluation).
- `dataset2maxlen.json`, `dataset2prompt.json` — per-dataset eval settings.
- `task2dataset.json` — 6 task families: Code Complete, Few Shot, Single-doc QA,
  Multi-doc QA, Summarization, Passage Retrieval.

### Supported Models
LLaMA 3.2 1B / 3.1 8B Instruct, Qwen 2.5, Mistral-7B-Instruct-v0.2 (`config/model2path.json`),
and any other HF model that uses transformers' AttentionInterface (the `"waits"` plugin is
model-agnostic; no per-model code). Mistral-Instruct shares Llama-2's `[INST]` chat format
(handled in longbench.py). (OPT was dropped: its 2048-token context is meaningless for
LongBench.) Qwen3-8B was fully benchmarked then DROPPED (history #56): needs
`enable_thinking=False` chat handling, and it repetition-collapses under weak compression
(StreamingLLM/KeyDiff) — reverted to Qwen 2.5. Sliding-window models (Gemma 3/4) need
flash-attn + per-model handling and are a poor KV-compression fit; MLA models (DeepSeek)
need latent-KV-specific handling.

## Output convention
Eval predictions go in `result_txt/pred/<budget>/<run_name>/`, never directly under
`result_txt/pred/`. `result_txt/backup/...` is read-only.
**Fast-eval store (canonical): `result_txt/backup/fast_store/store_<model>_128.jsonl.gz`**
— per-sample × per-action {pred, score}; scores are official-final; ALL routing/action
analyses look up here (never rescore preds). API/add-new-action: `script/fast_store.py`.
`runs/` keeps only active files (indexes, waits_tables, triattention_stats,
repro_2694_seed42); legacy RL runs archived under `runs/_archive_legacy_rl/`.

## Environment
```
conda activate A2SF    # python 3.12, torch 2.8.0+cu128, transformers 5.10.2 — the only env
```
Deps in `pip.txt` (`transformers==5.10.2`, `datasets<4.0.0`, …). The repo is **v5-only**
(history #43-#44): all code paths use the `compress` plugin. Rebuild from scratch:
`conda create -n A2SF python=3.12 && pip install torch==2.8.0 --index-url
https://download.pytorch.org/whl/cu128 && pip install -r pip.txt` (install the cu128 torch
*before* pip.txt, whose `torch` line is unpinned). Champion + paper baselines were generated
under 4.46.2 but reproduce under v5 within bf16 noise (single-prompt greedy bit-identical;
SnapKV-16 LB 25.46 vs 25.66 backup, extractive datasets exact, generative-QA ±~1 from fp
divergence); they live in `result_txt/backup/` (read-only) and are unaffected.

**Servers (history #44): all of 17 (local), 18, 19, 20 run this same v5 env + code.** eslab18
was a Python 3.8 env and was rebuilt at py3.12; 19/20/local were upgraded in place (so 18 is a
fresh pip.txt build, the others carry a few older secondary deps — core transformers/tokenizers/
torch/hub match). Server code is kept in sync from local (17) by rsync, not git (eslab19's copy
isn't a git checkout); `config/*` must stay identical across servers.
