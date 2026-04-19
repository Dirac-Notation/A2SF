"""RWR training — offline policy learning with reward-weighted CE.

Per-budget, single-model pipeline:
  for each sample: rewards = [r_1, ..., r_14] for 14 discrete a-values
  target distribution p* = softmax(r / T)
  policy π = softmax(logits)
  loss = -Σ_k p*_k log π_k              (KL(p* || π) up to a constant)

Inference in longbench_RL uses RWRAgent.act(), which performs
expected-action decoding:  α = Σ_k π(a_k|s) · a_k   (continuous).

Usage:
  python -m RL.rwr_train --model llama3-1b --budget 128 --save_dir runs/rwr_128 \
                         --target_T 0.1 --epochs 100
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL.a2sf_model import ModelConfig
from RL.env import A2SFModelRunner, A2SFEnv
from RL.agent.rwr_agent import RWRAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="llama3-1b")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--budget", type=int, required=True, choices=[128, 256, 512])
    p.add_argument("--data_file", type=str,
                   default="RL/training/1b_fix_exp_aug4/training_data_backup.jsonl")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--target_T", type=float, default=0.1,
                   help="temperature for target distribution softmax(r/T)")
    p.add_argument("--decoding_T", type=float, default=1.0,
                   help="temperature used at inference softmax(logits/T); leave at 1.0")
    p.add_argument("--label_smooth", type=float, default=0.0,
                   help="add ε/14 uniform mixture to target to avoid degenerate argmax")
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=0,
                   help="save periodic policy_epoch_N.pt every N epochs; 0 = disable (best+final only)")
    p.add_argument("--decoding_mode", type=str, default="argmax",
                   choices=["expected", "argmax"],
                   help="Phase F observation: argmax beats expected-\u03b1 even under noise.")
    p.add_argument("--select_by", type=str, default="argmax",
                   choices=["expected", "argmax", "max"],
                   help="Which val metric to select best checkpoint on.")
    return p.parse_args()


class RWRDataset(Dataset):
    def __init__(self, records, states):
        self.records = records
        self.states = states

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return {
            "state": self.states[r["_prompt_id"]],
            "rewards": torch.tensor(r["rewards"], dtype=torch.float32),
            "metric_type": r["metric_type"],
        }


def _collate(batch):
    return {
        "state": torch.stack([b["state"] for b in batch], dim=0),
        "rewards": torch.stack([b["rewards"] for b in batch], dim=0),
        "metric_types": [b["metric_type"] for b in batch],
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    bkey = str(args.budget)

    # ── Load records (supports two formats) ──
    # 1B format: action_scores_by_budget = { "128": [...], "256": [...], ... }
    # 8B format: action_scores = [...] + token_budget (per-file per-budget)
    records = []
    with open(args.data_file) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            sc = r.get("action_scores_by_budget", {}).get(bkey, [])
            if not isinstance(sc, list) or len(sc) != 14 or not any(float(x) > 0 for x in sc):
                # Try flat format
                if int(r.get("token_budget", -1)) == int(args.budget):
                    sc = r.get("action_scores", [])
                    if not isinstance(sc, list) or len(sc) != 14 or not any(float(x) > 0 for x in sc):
                        continue
                else:
                    continue
            records.append({
                "_prompt_id": i,
                "prompt": r["input_prompt"],
                "metric_type": str(r.get("metric_type", "qa_f1_score")),
                "task_type": str(r.get("task_type", "unknown")),
                "dataset": r.get("dataset"),
                "generation_length": int(r.get("generation_length", 0)),
                "rewards": [float(x) for x in sc],
            })
    n = len(records)
    print(f"Loaded {n} samples for budget {args.budget}")

    # ── Load encoder ──
    print("Loading encoder …")
    mc = ModelConfig(model=args.model)
    runner = A2SFModelRunner(mc)
    env = A2SFEnv(runner, mc)
    device = next(runner.model.model.layers[0].parameters()).device
    env.device = device
    enc = env.context_encoder

    # ── Encode all unique prompts ──
    prompt_to_state: Dict[int, torch.Tensor] = {}
    t0 = time.time()
    for r in records:
        pid = r["_prompt_id"]
        if pid in prompt_to_state:
            continue
        feats = enc.encode_context(
            text=r["prompt"],
            generation_length=r["generation_length"],
            token_budget=int(args.budget),
            metric_type=r["metric_type"],
            task_type=r["task_type"],
            dataset=r["dataset"],
        ).detach().cpu()
        prompt_to_state[pid] = feats
        if len(prompt_to_state) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  encoded {len(prompt_to_state)}   ({elapsed:.0f}s)")
    print(f"Encoded {len(prompt_to_state)} unique prompts ({time.time() - t0:.0f}s)")

    # ── Train/val split ──
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    n_val = int(args.val_frac * n)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    tr_records = [records[i] for i in tr_idx]
    val_records = [records[i] for i in val_idx]
    tr_ds = RWRDataset(tr_records, prompt_to_state)
    val_ds = RWRDataset(val_records, prompt_to_state)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=0, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=_collate)

    # ── Build agent ──
    state_dim = int(enc.output_dim)
    num_heads = int(enc.num_heads)
    num_metric_types = int(enc.num_metric_types)
    num_task_types = int(getattr(enc, "num_task_types", 0))
    side_dim = int(enc.side_dim)
    metric_heads = sorted({r["metric_type"] for r in records})

    agent = RWRAgent(
        state_dim=state_dim,
        a_values=mc.a_values,
        b_values=mc.b_values,
        metric_heads=metric_heads,
        num_heads=num_heads,
        num_metric_types=num_metric_types,
        num_task_types=num_task_types,
        side_dim=side_dim,
        hidden=args.hidden,
        decoding_temperature=args.decoding_T,
        decoding_mode=args.decoding_mode,
    ).to(device)
    print(f"RWR agent params: {sum(p.numel() for p in agent.parameters())}")

    opt = optim.Adam(agent.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_reward = -1.0
    best_ckpt = os.path.join(args.save_dir, "policy_best.pt")
    final_ckpt = os.path.join(args.save_dir, "policy_final.pt")
    log = []

    def build_target(rewards):
        # Soft target: softmax(r / T) + ε/K smoothing
        p = F.softmax(rewards / args.target_T, dim=-1)
        if args.label_smooth > 0:
            K = p.size(-1)
            p = (1 - args.label_smooth) * p + args.label_smooth / K
        return p

    def eval_val():
        agent.eval()
        nll = 0.0
        count = 0
        expected_rewards = []
        argmax_rewards = []
        oracle_rewards = []
        with torch.no_grad():
            for batch in val_loader:
                st = batch["state"].to(device)
                r = batch["rewards"].to(device)
                out = agent.forward(st, metric_type=batch["metric_types"])
                logits = out["logits"]
                logp = F.log_softmax(logits, dim=-1)
                target = build_target(r)
                sample_nll = -(target * logp).sum(dim=-1)
                nll += float(sample_nll.sum().item())
                count += logp.size(0)

                # Expected-action decoding
                probs = torch.softmax(logits, dim=-1)
                a_vals = agent.a_values.view(1, -1)
                a_exp = (probs * a_vals).sum(dim=-1)             # (B,)
                # Interpolated reward at a_exp (linear interp over 14 points)
                r_expected = _interp_reward(a_exp, agent.a_values, r)
                expected_rewards.extend(r_expected.cpu().tolist())

                # Argmax-decoded reward
                am_idx = probs.argmax(dim=-1)
                r_argmax = r.gather(1, am_idx.unsqueeze(1)).squeeze(1)
                argmax_rewards.extend(r_argmax.cpu().tolist())

                # Oracle
                r_oracle = r.max(dim=-1).values
                oracle_rewards.extend(r_oracle.cpu().tolist())

        agent.train()
        return {
            "nll": nll / max(1, count),
            "val_reward_expected": float(np.mean(expected_rewards)),
            "val_reward_argmax": float(np.mean(argmax_rewards)),
            "val_reward_oracle": float(np.mean(oracle_rewards)),
        }

    for epoch in range(1, args.epochs + 1):
        losses = []
        for batch in tr_loader:
            st = batch["state"].to(device)
            r = batch["rewards"].to(device)
            target = build_target(r)
            out = agent.forward(st, metric_type=batch["metric_types"])
            logp = F.log_softmax(out["logits"], dim=-1)
            loss = -(target * logp).sum(dim=-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))

        vm = eval_val()
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={np.mean(losses):.4f}  "
              f"val_nll={vm['nll']:.4f}  "
              f"r_argmax={vm['val_reward_argmax']:.4f}  "
              f"r_expected={vm['val_reward_expected']:.4f}  "
              f"oracle={vm['val_reward_oracle']:.4f}")
        log.append({"epoch": epoch, "loss": float(np.mean(losses)), **vm})

        if args.select_by == "expected":
            sel = vm["val_reward_expected"]
        elif args.select_by == "argmax":
            sel = vm["val_reward_argmax"]
        else:
            sel = max(vm["val_reward_expected"], vm["val_reward_argmax"])
        if sel > best_val_reward:
            best_val_reward = sel
            torch.save({"iteration": epoch,
                        "agent_state_dict": agent.state_dict(),
                        "arch_config": _arch(args, agent)},
                       best_ckpt)

        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"policy_epoch_{epoch}.pt")
            torch.save({"iteration": epoch,
                        "agent_state_dict": agent.state_dict(),
                        "arch_config": _arch(args, agent)},
                       ckpt_path)

    # final snapshot
    torch.save({"iteration": args.epochs,
                "agent_state_dict": agent.state_dict(),
                "arch_config": _arch(args, agent)},
               final_ckpt)

    with open(os.path.join(args.save_dir, "rwr_train.log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nBest val expected-reward: {best_val_reward:.4f}  →  {best_ckpt}")
    print(f"Final: {final_ckpt}")


def _interp_reward(a_q: torch.Tensor, a_grid: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate reward at query a-value within the 14-point grid.

    a_q:      (B,)
    a_grid:   (K,) sorted ascending (14 discrete a values)
    rewards:  (B, K) rewards at each grid point
    Returns: (B,) interpolated reward.
    """
    a_q = a_q.clamp(min=a_grid.min().item(), max=a_grid.max().item())
    K = a_grid.numel()
    # For each a_q find upper and lower grid index
    # a_grid is sorted ascending by construction
    idx_hi = torch.searchsorted(a_grid, a_q).clamp(1, K - 1)
    idx_lo = (idx_hi - 1).clamp(0, K - 2)
    a_lo = a_grid[idx_lo]; a_hi = a_grid[idx_hi]
    t = (a_q - a_lo) / (a_hi - a_lo + 1e-12)
    r_lo = rewards.gather(1, idx_lo.unsqueeze(1)).squeeze(1)
    r_hi = rewards.gather(1, idx_hi.unsqueeze(1)).squeeze(1)
    return (1 - t) * r_lo + t * r_hi


def _arch(args, agent):
    return {
        "state_dim": int(agent.state_dim),
        "num_heads": int(agent.num_heads),
        "num_metric_types": int(agent.num_metric_types),
        "num_task_types": int(agent.num_task_types),
        "side_dim": int(agent.side_dim),
        "metric_heads": list(agent.metric_heads),
        "a_values": agent.a_values.detach().cpu(),
        "b_values": agent.b_values.detach().cpu(),
        "hidden": int(args.hidden),
        "decoding_temperature": float(args.decoding_T),
        "decoding_mode": str(args.decoding_mode),
        "rwr": True,
        "compact": False,
        "binary": False,
        "feature_dim": 256,
        "num_actions": int(agent.num_actions),
        "num_a_values": int(agent.num_a_values),
        "num_b_values": int(agent.num_b_values),
        "backbone_depth": 2,
        "dropout": 0.0,
    }


if __name__ == "__main__":
    main()
