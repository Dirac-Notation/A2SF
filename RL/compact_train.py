"""Compact agent training — single supervised pipeline across all budgets.

Pipeline:
  1. Load training data for all budgets (128, 256, 512).
  2. Build state features via the existing AttentionEncoder (cached per prompt, since
     attention features don't depend on budget — only the budget slot does).
  3. Train CompactAgent with soft cross-entropy on reward distribution per sample.
  4. Evaluate MRR periodically.
  5. Save a standard-format checkpoint (arch_config + agent_state_dict) so
     existing longbench_RL.py can load it.

Usage:
  python -m RL.compact_train \
      --model llama3-1b \
      --save runs/compact_rl/policy_final.pt \
      --budgets 128 256 512 \
      --epochs 300
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL.a2sf_model import ModelConfig
from RL.env import A2SFModelRunner, A2SFEnv
from RL.agent.compact_agent import CompactAgent


def parse_args():
    p = argparse.ArgumentParser(description="Train compact RL agent (budget-conditioned)")
    p.add_argument("--model", type=str, default="llama3-1b")
    p.add_argument("--save_dir", type=str, default="runs/compact_rl")
    p.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--data_dir", type=str, default="RL/training/1b_fix_exp_aug4")
    p.add_argument("--data_file", type=str, default="training_data_backup.jsonl")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--soft_T", type=float, default=0.05,
                   help="softmax temperature over rewards (lower=sharper target)")
    p.add_argument("--label_smooth", type=float, default=0.05)
    p.add_argument("--loss", type=str, default="multilabel",
                   choices=["soft_ce", "hard_ce", "multilabel", "mse"])
    p.add_argument("--ml_delta", type=float, default=0.02,
                   help="(multilabel) near-optimal threshold")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class MultiBudgetDataset(Dataset):
    """Each item: (state_vec, budget, reward_vec, metric_type).

    States are cached per-prompt (identical across budgets, since the encoder
    doesn't use budget).
    """

    def __init__(self, samples: List[Dict], state_cache: Dict[int, torch.Tensor]):
        self.samples = samples
        self.state_cache = state_cache

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        state = self.state_cache[s["prompt_id"]]
        return {
            "state": state,
            "budget": int(s["budget"]),
            "rewards": torch.tensor(s["rewards"], dtype=torch.float32),
            "metric_type": str(s.get("metric_type", "qa_f1_score")),
        }


def _collate(batch):
    return {
        "state": torch.stack([b["state"] for b in batch], dim=0),
        "budget": torch.tensor([b["budget"] for b in batch], dtype=torch.float32),
        "rewards": torch.stack([b["rewards"] for b in batch], dim=0),
        "metric_types": [b["metric_type"] for b in batch],
    }


def load_multi_budget(data_path: str, budgets: List[int]) -> List[Dict]:
    """Flatten: one record per (prompt, budget) with non-zero rewards."""
    raw = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw.append(json.loads(line))

    flat = []
    for i, r in enumerate(raw):
        scores_by_b = r.get("action_scores_by_budget", {})
        for bkey in [str(b) for b in budgets]:
            sc = scores_by_b.get(bkey, [])
            if not isinstance(sc, list) or len(sc) == 0:
                continue
            if not any(float(x) > 0 for x in sc):
                continue
            flat.append({
                "prompt_id": i,
                "prompt": r["input_prompt"],
                "budget": int(bkey),
                "rewards": [float(x) for x in sc],
                "metric_type": str(r.get("metric_type", "qa_f1_score")),
                "task_type": str(r.get("task_type", "unknown")),
                "dataset": r.get("dataset"),
                "generation_length": r.get("generation_length", 0),
            })
    return flat


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Load data ──
    data_path = os.path.join(args.data_dir, args.data_file)
    samples = load_multi_budget(data_path, args.budgets)
    print(f"Loaded {len(samples)} (prompt, budget) records from {data_path}")

    # ── Spin up encoder (we need its attention features once per prompt) ──
    print(f"Loading model {args.model} to build encoder…")
    model_cfg = ModelConfig(model=args.model)
    runner = A2SFModelRunner(model_cfg)
    env = A2SFEnv(runner, model_cfg)
    device = next(runner.model.model.layers[0].parameters()).device
    env.device = device
    encoder = env.context_encoder

    # Discover unique prompts & cache states
    unique_prompts: Dict[int, Dict] = {}
    for s in samples:
        if s["prompt_id"] not in unique_prompts:
            unique_prompts[s["prompt_id"]] = s
    print(f"Encoding {len(unique_prompts)} unique prompts …")

    state_cache: Dict[int, torch.Tensor] = {}
    t0 = time.time()
    for k, rec in unique_prompts.items():
        feats = encoder.encode_context(
            text=rec["prompt"],
            generation_length=int(rec["generation_length"]),
            token_budget=int(args.budgets[0]),           # budget slot is unused at encoder level
            metric_type=rec["metric_type"],
            task_type=rec["task_type"],
            dataset=rec["dataset"],
        )
        state_cache[k] = feats.to("cpu")
        if len(state_cache) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  encoded {len(state_cache)}/{len(unique_prompts)}  ({elapsed:.0f}s)")
    print(f"Done encoding ({time.time() - t0:.0f}s).")

    # ── Build dataset + loader ──
    dataset = MultiBudgetDataset(samples, state_cache)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=_collate)

    # ── Build agent ──
    state_dim = int(encoder.output_dim)
    num_heads = int(encoder.num_heads)
    num_metric_types = int(encoder.num_metric_types)
    num_task_types = int(getattr(encoder, "num_task_types", 0))
    side_dim = int(encoder.side_dim)

    metric_heads = sorted({s["metric_type"] for s in samples})
    agent = CompactAgent(
        state_dim=state_dim,
        a_values=model_cfg.a_values,
        b_values=model_cfg.b_values,
        metric_heads=metric_heads,
        num_heads=num_heads,
        num_metric_types=num_metric_types,
        num_task_types=num_task_types,
        side_dim=side_dim,
        hidden=args.hidden,
        budget_slot=True,
    ).to(device)

    n_params = sum(p.numel() for p in agent.parameters())
    print(f"Agent built. params={n_params}  hidden={args.hidden}  budget_slot={True}")

    optimizer = optim.Adam(agent.parameters(), lr=args.lr, weight_decay=1e-4)

    # ── Per-sample optimal action index for MRR ──
    per_sample_opt = [int(np.argmax(s["rewards"])) for s in samples]

    # ── Training ──
    log = []
    for ep in range(1, args.epochs + 1):
        agent.train()
        epoch_losses = []
        for batch in loader:
            state = batch["state"].to(device)
            budget = batch["budget"].to(device)
            rewards = batch["rewards"].to(device)
            metric_types = batch["metric_types"]

            # Set budget feature per sample (broadcast across batch)
            val = (budget - 128.0) / (512.0 - 128.0)
            val = val.clamp(0.0, 1.0).unsqueeze(-1)
            agent._current_budget_feat = val

            out = agent.forward(state, metric_type=metric_types)
            logits = out["logits"]

            if args.loss == "soft_ce":
                T = args.soft_T
                target = torch.softmax(rewards / T, dim=-1)
                if args.label_smooth > 0:
                    ls = args.label_smooth
                    target = (1 - ls) * target + ls / target.size(-1)
                log_p = torch.log_softmax(logits, dim=-1)
                loss = -(target * log_p).sum(dim=-1).mean()
            elif args.loss == "hard_ce":
                target_idx = rewards.argmax(dim=-1)
                loss = F.cross_entropy(logits, target_idx, label_smoothing=args.label_smooth)
            elif args.loss == "multilabel":
                # Positives = any action within `ml_delta` of the sample's max reward
                best = rewards.max(dim=-1, keepdim=True).values
                target = (rewards >= best - args.ml_delta).float()
                loss = F.binary_cross_entropy_with_logits(logits, target)
            else:  # mse
                loss = F.mse_loss(torch.sigmoid(logits), rewards)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        # MRR on training set (quick check)
        agent.eval()
        total_rr = 0.0
        n_total = 0
        with torch.no_grad():
            all_states = torch.stack([state_cache[s["prompt_id"]] for s in samples], dim=0).to(device)
            budgets = torch.tensor([s["budget"] for s in samples], dtype=torch.float32, device=device)
            val = ((budgets - 128.0) / (512.0 - 128.0)).clamp(0.0, 1.0).unsqueeze(-1)
            agent._current_budget_feat = val
            pred = agent.forward(all_states, metric_type=[s["metric_type"] for s in samples])["reward_pred"]
            ranks = torch.argsort(pred, descending=True, dim=-1)
            for i in range(len(samples)):
                opt = per_sample_opt[i]
                rank = (ranks[i] == opt).nonzero().item() + 1
                total_rr += 1.0 / rank
                n_total += 1
        mrr = total_rr / max(1, n_total)

        mean_loss = float(np.mean(epoch_losses))
        print(f"Epoch {ep:3d}/{args.epochs}  loss={mean_loss:.4f}  MRR={mrr:.4f}")
        log.append({"epoch": ep, "loss": mean_loss, "mrr": mrr})

    # ── Save ──
    arch_config = {
        "state_dim": int(agent.state_dim),
        "num_heads": int(agent.num_heads),
        "num_metric_types": int(agent.num_metric_types),
        "num_task_types": int(agent.num_task_types),
        "side_dim": int(agent.side_dim),
        "metric_heads": list(agent.metric_heads),
        "a_values": agent.a_values.detach().cpu(),
        "b_values": agent.b_values.detach().cpu(),
        "hidden": int(args.hidden),
        "compact": True,     # marker so inference picks CompactAgent
        # plus legacy fields so old load paths don't explode
        "feature_dim": 256,
        "num_actions": int(agent.num_actions),
        "num_a_values": int(agent.num_a_values),
        "num_b_values": int(agent.num_b_values),
        "backbone_depth": 2,
        "dropout": 0.0,
    }

    final_path = os.path.join(args.save_dir, "policy_final.pt")
    torch.save({
        "iteration": args.epochs,
        "agent_state_dict": agent.state_dict(),
        "arch_config": arch_config,
    }, final_path)
    print(f"Saved to {final_path}")

    with open(os.path.join(args.save_dir, "compact_train.log.json"), "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
