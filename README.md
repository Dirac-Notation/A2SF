# WAITS — KV-Cache Compression with a Sigmoid Forgetting Curve

WAITS reduces LLM inference memory by selectively retaining key-value pairs using
accumulative attention scores weighted by a sigmoid forgetting curve
`w[q] = 1 / (1 + exp(-a * (q - (N - b - 0.5))))`. A learned bandit policy routes each
(task, metric) cell to one of 5 regime-representative `(a, b)` curves, whose limits
recover H2O (`a=0`), TOVA (`a→∞, b=1`), and SnapKV (`a→∞, b=window`).

(The repo/env name `A2SF` is the project's legacy name.)

## Environment

```bash
conda create -n A2SF python=3.12
conda activate A2SF
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r pip.txt   # transformers==5.10.2, datasets<4.0.0, ...
```

Install the cu128 torch BEFORE `pip.txt` (its `torch` line is unpinned).

## Quickstart

The compression mechanism is a single model-agnostic transformers-v5 attention plugin
(`utils_real_drop/compress.py`, registered as `"waits"`). Any HF model using the
AttentionInterface works — no per-model code.

```python
import utils

model, tokenizer = utils.load_model("llama3-8b")   # config/model2path.json shortnames

# fixed WAITS action (a, b) at budget 128
model.init_cache({
    "compression_method": "waits", "sigmoid_a": 1.0, "recent_budget": 16,
    "select_budget": 128,
})
out = model.generate(**inputs, max_new_tokens=64)

model.init_cache(None)   # disable compression
```

## Evaluation

```bash
# LongBench, routed WAITS (champion): (task, metric) -> (a, b) lookup table
python longbench.py --model llama3-8b --budget 128 \
    --waits_table runs/waits_tables/waits_llama3-8b_u5b.json --run_name WAITS_8b

# Baselines (aliases of the snap scorer): TOVA = --method snap --window 1,
# SnapKV = --method snap --window 16, H2O = --method snap --window 32768.
python longbench.py --model llama3-8b --budget 128 --method snap --window 16 --run_name SnapKV

# Fixed WAITS action
python longbench.py --model llama3-8b --budget 128 --method waits --window 16 --sigmoid_a 1 --run_name waits_1_16
```

Other entry points: `evaluate_needle.py` (needle-in-a-haystack), `benchmark_ttft.py`
(prefill latency), `longbench_oracle.py` (oracle upper bound). Multi-server eval uses the
sample-level global queue (`script/orchestrator_lb.py` + `script/worker_lb.py`,
see `script/GLOBAL_QUEUE.md`).

## Training the routing policy

```bash
# 1) score the training recipe (multi-GPU, full-cache + per-action inference)
python RL/dataset.py --model llama3-8b ...

# 2) train the LinUCB router and export a deploy table
python RL/train.py --model llama3-8b \
    --recipe datasets/training/raw/recipe_v3_llama3-8b/train.jsonl \
    --actions "0:1,0.01:128,1:16,10:1,10:16" --epochs 64 --seed 0 \
    --save_agent runs/selectors/llama3-8b_u5b.npz
```

See `RL/README.md` for the D-I-A-R-L pipeline layout and `CLAUDE.md` for the full
architecture notes.

## Repository layout

| path | what |
|---|---|
| `utils_real_drop/` | the v5 compression plugin: `compress.py` + `scorers/` (waits, snap, triattention, streamingllm, keydiff, l2norm) + `selectors/` (top-k, ChunkKV, Ada-KV, oracle) + `kvzip.py` |
| `RL/` | the (task, metric) routing bandit (D-I-A-R-L) + version-locked submitted-method workbench |
| `longbench*.py`, `evaluate_needle.py` | evaluation pipelines |
| `datasets/` | training-recipe generators (non-LongBench sources) + eval data prep |
| `script/` | index/store builders, eval orchestration, campaign runners |
| `config/` | model paths, per-dataset prompts/lengths, task families, chat-template modes |

Supported eval models: LLaMA 3.2 1B / 3.1 8B Instruct, Qwen 2.5 7B, Mistral-7B-Instruct-v0.2.
