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

The earlier per-prompt mini-attn / deep-agent / joint machinery was removed 2026-06-16
(see `logs/history/54...`; recover the code from git history if reviving a per-prompt RL model).
