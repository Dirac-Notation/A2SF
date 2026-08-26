# RL — WAITS (task, metric) routing policy

This folder is the **deployed WAITS routing policy**, organized as a clean D-I-A-R-L pipeline.
The policy is a contextual bandit (NeuralUCB) over the observable metadata state
`phi(s) = [task_oh | metric_oh]`. Since the state is categorical, NeuralUCB's Sherman-Morrison
`Sigma^-1` core reduces exactly to **per-arm LinUCB on one-hot features** (no encoder, no deep
net), and the greedy policy is a `(task, metric) -> (a, b)` **lookup table**, applied at eval
through the standard `longbench.py --waits_table`.

| file | DIARL | what |
|---|---|---|
| `action_grid.py` | D/A | the 13 paired `(a, b)` sigmoid actions (single source of truth) |
| `metadata.py`    | I    | task/metric vocab + one-hot index helpers + `dataset_metric()` |
| `dataset.py`     | D    | generate `(task, metric, action_scores_gt_by_budget)` training data |
| `model.py`       | A    | `RoutingNeuralUCB` (`phi/select/update/greedy_action/export_table`) |
| `train.py`       | R, L | bandit loop + Sherman-Morrison update + eval on LB index + `--export_table` |

## Train + deploy
```
# 1) train the routing policy and export a longbench --waits_table block
python RL/train.py --model llama3-1b \
    --recipe datasets/training/raw/recipe_v3_1b/train.jsonl \
    --index runs/fast_lb_eval/index_llama3-1b_128.pt --budget 128 \
    --export_table runs/waits_tables/waits_llama3-1b.json

# 2) eval on LongBench (or use longbench_RL.py which does both steps)
python longbench.py --model llama3-1b --budget 128 \
    --waits_table runs/waits_tables/waits_llama3-1b.json --run_name WAITS_llama3-1b
```

## Per-prompt submitted-method workbench (version-locked)

The per-prompt mini-attn machinery was removed 2026-06-16 (history #54), then partially
RESTORED 2026-07-24 (history #68) as the version-locked workbench that reproduces the
NeurIPS-submitted per-prompt numbers (1B 26.94, `script/repro_2694.sh`). It is a closed
research branch - keep for rebuttal/camera-ready reproduction, do not extend:

| file | what |
|---|---|
| `train_perprompt_submitted.py` | the submitted trainer (old train.py: top-K UCB + MSE, mini-attn state) |
| `a2sf_model.py` | ModelConfig + agent/env wiring; carries a deliberate inline copy of the 13-grid |
| `agent/` | `NeuralUCBAgent` (submitted), `SimpleUCBAgent`/`LoRAUCBAgent` (sweep variants) |
| `env/` | episode env + `mini_attn_encoder` (submitted state) + legacy `AttentionEncoder` |
| `nll_reward.py` | N1 gold-NLL reward probe (closed 2026-07-13, history #67) |

Consumers outside this folder: `script/{fast_lb_select,fast_lb_eval,repro_2694.sh}`,
`script/gsm8k_cot_eval.py`, `benchmark_ttft.py --rl_ckpt`.
