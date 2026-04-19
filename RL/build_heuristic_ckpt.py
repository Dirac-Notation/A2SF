"""Build a HeuristicAgent checkpoint from offline reward data.

For each (task_type, metric_type) pair, picks the action that maximized the mean
reward across training samples. Saves as a checkpoint loadable by A2SFModel.

Usage:
  python -m RL.build_heuristic_ckpt --budget 128 \
      --data_file RL/training/1b_fix_exp_aug4/training_data_backup.jsonl \
      --save runs/heuristic_1b_128/policy_best.pt
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL.env.encoder import TASK_TYPE_ORDER, TASK_TYPE_TO_INDEX
from RL.a2sf_model import ModelConfig

# Metric ordering matches encoder.py
METRIC_TYPE_ORDER = [
    "qa_f1_score", "rouge_score", "classification_score", "retrieval_score",
    "count_score", "code_sim_score", "rouge_zh_score", "retrieval_zh_score",
    "classification_zh_score", "unknown",
]
METRIC_TYPE_TO_INDEX = {n: i for i, n in enumerate(METRIC_TYPE_ORDER)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--data_file", type=str, required=True)
    p.add_argument("--save", type=str, required=True)
    p.add_argument("--model", type=str, default="llama3-1b")
    return p.parse_args()


def main():
    args = parse_args()
    bkey = str(args.budget)

    # ── Load records ──
    per_key: Dict = defaultdict(list)  # type: ignore
    global_rewards = []
    all_metric_seen = set()
    all_task_seen = set()
    with open(args.data_file) as f:
        for line in f:
            r = json.loads(line)
            sc = r.get("action_scores_by_budget", {}).get(bkey, [])
            if not sc:
                if int(r.get("token_budget", -1)) == int(args.budget):
                    sc = r.get("action_scores", [])
            if not isinstance(sc, list) or len(sc) != 14 or not any(float(x) > 0 for x in sc):
                continue
            task = str(r.get("task_type", "unknown"))
            metric = str(r.get("metric_type", "qa_f1_score"))
            arr = np.array([float(x) for x in sc])
            per_key[(task, metric)].append(arr)
            global_rewards.append(arr)
            all_metric_seen.add(metric)
            all_task_seen.add(task)

    print(f"budget {args.budget}: loaded {sum(len(v) for v in per_key.values())} samples, "
          f"{len(per_key)} (task, metric) cells")
    print(f"  tasks seen: {sorted(all_task_seen)}")
    print(f"  metrics seen: {sorted(all_metric_seen)}")

    # ── Global fallback: best action across all samples ──
    R_all = np.stack(global_rewards)
    global_best = int(R_all.mean(axis=0).argmax())
    print(f"  global fallback best action: {global_best}")

    # ── Build lookup table [num_task, num_metric] ──
    T = len(TASK_TYPE_ORDER)
    M = len(METRIC_TYPE_ORDER)
    table = np.full((T, M), global_best, dtype=np.int64)
    cell_counts = np.zeros((T, M), dtype=np.int64)
    cell_mean_reward = np.full((T, M), np.nan)

    for (task, metric), rs in per_key.items():
        t_idx = TASK_TYPE_TO_INDEX.get(task, TASK_TYPE_TO_INDEX.get("unknown"))
        m_idx = METRIC_TYPE_TO_INDEX.get(metric, METRIC_TYPE_TO_INDEX.get("unknown"))
        if t_idx is None or m_idx is None:
            continue
        R = np.stack(rs)
        best_a = int(R.mean(axis=0).argmax())
        table[t_idx, m_idx] = best_a
        cell_counts[t_idx, m_idx] = len(rs)
        cell_mean_reward[t_idx, m_idx] = float(R[:, best_a].mean())

    # Summary
    print(f"\n{'task':<22} {'metric':<22} {'N':>4} {'best_a':>6} {'mean_r':>8}")
    for (task, metric) in sorted(per_key.keys()):
        t_idx = TASK_TYPE_TO_INDEX.get(task, TASK_TYPE_TO_INDEX.get("unknown"))
        m_idx = METRIC_TYPE_TO_INDEX.get(metric, METRIC_TYPE_TO_INDEX.get("unknown"))
        a = int(table[t_idx, m_idx])
        n = int(cell_counts[t_idx, m_idx])
        rm = float(cell_mean_reward[t_idx, m_idx])
        print(f"{task[:22]:<22} {metric[:22]:<22} {n:>4} {a:>6} {rm:>8.4f}")

    # ── Save checkpoint ──
    mc = ModelConfig(model=args.model)
    a_values = mc.a_values.detach().cpu()
    b_values = mc.b_values.detach().cpu()

    arch = {
        "state_dim": 131090,          # placeholder; HeuristicAgent doesn't use state features
        "num_heads": 32,              # placeholder
        "num_metric_types": M,
        "num_task_types": T,
        "side_dim": 65536,            # placeholder
        "metric_heads": list(METRIC_TYPE_ORDER),
        "a_values": a_values,
        "b_values": b_values,
        "hidden": 256,
        "heuristic": True,
        "rwr": False,
        "rwr_flat": False,
        "compact": False,
        "binary": False,
        "feature_dim": 256,
        "num_actions": 14,
        "num_a_values": len(a_values),
        "num_b_values": len(b_values),
        "backbone_depth": 2,
        "dropout": 0.0,
        "action_table_int": table.tolist(),          # (T, M)
    }

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    # Save minimal state dict containing the action_table buffer
    state_dict = {
        "a_values": a_values,
        "b_values": b_values,
        "action_table": torch.tensor(table, dtype=torch.long),
        "action_counts": torch.zeros(M, 14, dtype=torch.long),
    }
    torch.save({
        "iteration": 0,
        "agent_state_dict": state_dict,
        "arch_config": arch,
    }, args.save)
    print(f"\nSaved heuristic ckpt → {args.save}")


if __name__ == "__main__":
    main()
