"""Train binary 'should use action 0?' agent — simple supervised learning.

Per-budget training: one model per budget. Uses the full 131K attention state
from the existing AttentionEncoder.

Usage:
  python -m RL.binary_train --model llama3-1b --budget 128 --save_dir runs/binary_128
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
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL.a2sf_model import ModelConfig
from RL.env import A2SFModelRunner, A2SFEnv
from RL.agent.binary_agent import BinaryAgent, DEFAULT_BACKUP


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="llama3-1b")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--budget", type=int, required=True, choices=[128, 256, 512])
    p.add_argument("--data_file", type=str, default="RL/training/1b_fix_exp_aug4/training_data_backup.jsonl")
    p.add_argument("--delta", type=float, default=0.02, help="near-optimal tolerance for label")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--backup_idx", type=int, default=None,
                   help="action index to use when predicting 'forget'. Default: budget-specific.")
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class BinaryDataset(Dataset):
    def __init__(self, records, states):
        self.records = records
        self.states = states

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return {
            "state": self.states[r["_prompt_id"]],
            "label": torch.tensor(r["label"], dtype=torch.float32),
            "best_r": torch.tensor(r["rewards"][int(np.argmax(r["rewards"]))], dtype=torch.float32),
            "r0": torch.tensor(r["rewards"][0], dtype=torch.float32),
            "r_backup": torch.tensor(r["rewards"][r["backup_idx"]], dtype=torch.float32),
            "metric_type": r["metric_type"],
        }


def _collate(batch):
    return {
        "state": torch.stack([b["state"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "best_r": torch.stack([b["best_r"] for b in batch], dim=0),
        "r0": torch.stack([b["r0"] for b in batch], dim=0),
        "r_backup": torch.stack([b["r_backup"] for b in batch], dim=0),
        "metric_types": [b["metric_type"] for b in batch],
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Load records for this budget ──
    bkey = str(args.budget)
    backup_idx = args.backup_idx if args.backup_idx is not None else DEFAULT_BACKUP[args.budget]
    print(f"Budget {args.budget}, backup action idx = {backup_idx}")

    records = []
    with open(args.data_file) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            sc = r.get("action_scores_by_budget", {}).get(bkey, [])
            if not isinstance(sc, list) or len(sc) != 14 or not any(float(x) > 0 for x in sc):
                continue
            rewards = np.array([float(x) for x in sc])
            label = int(rewards[0] >= rewards.max() - args.delta)
            records.append({
                "_prompt_id": i,
                "prompt": r["input_prompt"],
                "metric_type": str(r.get("metric_type", "qa_f1_score")),
                "task_type": str(r.get("task_type", "unknown")),
                "dataset": r.get("dataset"),
                "generation_length": int(r.get("generation_length", 0)),
                "rewards": rewards.tolist(),
                "label": label,
                "backup_idx": backup_idx,
            })
    n = len(records)
    pos_rate = float(np.mean([r["label"] for r in records]))
    print(f"Loaded {n} samples for budget {args.budget}. pos rate (label=1) = {pos_rate:.3f}")

    # ── Encode states ──
    print("Loading encoder …")
    mc = ModelConfig(model=args.model)
    runner = A2SFModelRunner(mc)
    env = A2SFEnv(runner, mc)
    device = next(runner.model.model.layers[0].parameters()).device
    env.device = device
    enc = env.context_encoder

    prompt_to_state = {}
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
            print(f"  encoded {len(prompt_to_state)}/{n}   ({elapsed:.0f}s)")
    print(f"Encoded {len(prompt_to_state)} unique prompts ({time.time() - t0:.0f}s)")

    # ── Train/val split ──
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    n_val = int(args.val_frac * n)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    tr_records = [records[i] for i in tr_idx]
    val_records = [records[i] for i in val_idx]
    print(f"Train {len(tr_records)}, Val {len(val_records)}")

    tr_ds = BinaryDataset(tr_records, prompt_to_state)
    val_ds = BinaryDataset(val_records, prompt_to_state)
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

    agent = BinaryAgent(
        state_dim=state_dim,
        a_values=mc.a_values,
        b_values=mc.b_values,
        metric_heads=metric_heads,
        num_heads=num_heads,
        num_metric_types=num_metric_types,
        num_task_types=num_task_types,
        side_dim=side_dim,
        hidden=args.hidden,
        backup_idx=backup_idx,
    ).to(device)

    n_params = sum(p.numel() for p in agent.parameters())
    print(f"Agent params: {n_params}")

    # ── Class weight for BCE ──
    # Many samples have label=0 (action 0 not best). We want to prefer balanced.
    w_pos = (1 - pos_rate) / max(pos_rate, 1e-4)  # inverse freq
    pos_weight = torch.tensor(w_pos, device=device)
    print(f"pos_weight for BCE: {w_pos:.3f}")

    opt = optim.Adam(agent.parameters(), lr=args.lr, weight_decay=1e-4)

    # ── Training ──
    best_val_reward = -1.0
    best_ckpt_path = os.path.join(args.save_dir, "policy_best.pt")
    final_ckpt_path = os.path.join(args.save_dir, "policy_final.pt")

    def eval_val():
        agent.eval()
        n_correct = 0; n_total = 0
        rewards_if_pred = []
        rewards_always_0 = []
        rewards_oracle = []
        with torch.no_grad():
            for batch in val_loader:
                st = batch["state"].to(device)
                out = agent.forward(st, metric_type=batch["metric_types"])
                p0 = out["p_zero"]
                pred = (p0 >= 0.5).long()
                lab = batch["label"].long().to(device)
                n_correct += int((pred == lab).sum().item())
                n_total += len(lab)
                for i, p in enumerate(pred.cpu().tolist()):
                    if p == 1:  # predicted action 0 is best
                        rewards_if_pred.append(float(batch["r0"][i]))
                    else:
                        rewards_if_pred.append(float(batch["r_backup"][i]))
                    rewards_always_0.append(float(batch["r0"][i]))
                    rewards_oracle.append(float(batch["best_r"][i]))
        agent.train()
        return {
            "acc": n_correct / max(1, n_total),
            "val_reward": float(np.mean(rewards_if_pred)),
            "val_always0": float(np.mean(rewards_always_0)),
            "val_oracle": float(np.mean(rewards_oracle)),
        }

    for epoch in range(1, args.epochs + 1):
        losses = []
        for batch in tr_loader:
            st = batch["state"].to(device)
            lab = batch["label"].to(device)
            out = agent.forward(st, metric_type=batch["metric_types"])
            logit = out["logits"]
            loss = nn.functional.binary_cross_entropy_with_logits(logit, lab, pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))

        vm = eval_val()
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={np.mean(losses):.4f}  "
              f"val_acc={vm['acc']:.3f}  val_reward={vm['val_reward']:.4f}  "
              f"always0={vm['val_always0']:.4f}  oracle={vm['val_oracle']:.4f}")

        # save best
        if vm["val_reward"] > best_val_reward:
            best_val_reward = vm["val_reward"]
            arch = _arch(args, agent, backup_idx)
            torch.save({"iteration": epoch, "agent_state_dict": agent.state_dict(),
                        "arch_config": arch}, best_ckpt_path)

    # always save final
    arch = _arch(args, agent, backup_idx)
    torch.save({"iteration": args.epochs, "agent_state_dict": agent.state_dict(),
                "arch_config": arch}, final_ckpt_path)
    print(f"\nSaved best → {best_ckpt_path} (val_reward={best_val_reward:.4f})")
    print(f"Saved final → {final_ckpt_path}")


def _arch(args, agent, backup_idx):
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
        "backup_idx": int(backup_idx),
        "binary": True,
        "compact": False,
        "feature_dim": 256,
        "num_actions": int(agent.num_actions),
        "num_a_values": int(agent.num_a_values),
        "num_b_values": int(agent.num_b_values),
        "backbone_depth": 2,
        "dropout": 0.0,
    }


if __name__ == "__main__":
    main()
