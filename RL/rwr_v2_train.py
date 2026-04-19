"""RWRv2 training — learned embeddings + task-conditional heads + adaptive T.

Per-sample adaptive target temperature:
  T_i = max(T_floor, (r_max - r_2nd)_i) / kappa
  target_i = softmax(r_i / T_i)

Motivation: sample-wise peakiness should be consistent across reward scales.
  - Clear-winner samples (gap > 0.1): T ≈ 0.05-0.2 → target peaky, commit to winner.
  - Near-tie samples (gap < 0.02): T ≈ 0.04 (after floor) → target less-peaky, respects tie.

Inference: argmax(π).  Checkpoint best-by-val-r_argmax.
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
from RL.agent.rwr_v2_agent import RWRv2Agent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="llama3-1b")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--data_file", type=str,
                   default="RL/training/1b_fix_exp_aug4/training_data_backup.jsonl")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--task_emb_dim", type=int, default=32)
    p.add_argument("--metric_emb_dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--T_floor", type=float, default=0.02,
                   help="lower bound for per-sample target temperature")
    p.add_argument("--kappa", type=float, default=1.0,
                   help="peakiness divisor: T = max(floor, gap)/kappa; larger kappa → peaker targets")
    p.add_argument("--decoding_mode", type=str, default="argmax", choices=["expected", "argmax"])
    p.add_argument("--select_by", type=str, default="argmax", choices=["expected", "argmax", "max"])
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=0)
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
            "task_type": r["task_type"],
        }


def _collate(batch):
    return {
        "state": torch.stack([b["state"] for b in batch], dim=0),
        "rewards": torch.stack([b["rewards"] for b in batch], dim=0),
        "metric_types": [b["metric_type"] for b in batch],
        "task_types": [b["task_type"] for b in batch],
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    bkey = str(args.budget)
    records = []
    with open(args.data_file) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            sc = r.get("action_scores_by_budget", {}).get(bkey, [])
            if not sc and int(r.get("token_budget", -1)) == int(args.budget):
                sc = r.get("action_scores", [])
            if not isinstance(sc, list) or len(sc) != 14 or not any(float(x) > 0 for x in sc):
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

    print("Loading encoder …")
    mc = ModelConfig(model=args.model)
    runner = A2SFModelRunner(mc)
    env = A2SFEnv(runner, mc)
    device = next(runner.model.model.layers[0].parameters()).device
    env.device = device
    enc = env.context_encoder

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
            print(f"  encoded {len(prompt_to_state)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"Encoded {len(prompt_to_state)} unique prompts ({time.time() - t0:.0f}s)", flush=True)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    n_val = int(args.val_frac * n)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    tr_records = [records[i] for i in tr_idx]
    val_records = [records[i] for i in val_idx]
    tr_loader = DataLoader(RWRDataset(tr_records, prompt_to_state),
                           batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=_collate)
    val_loader = DataLoader(RWRDataset(val_records, prompt_to_state),
                            batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate)

    state_dim = int(enc.output_dim)
    num_heads = int(enc.num_heads)
    num_metric_types = int(enc.num_metric_types)
    num_task_types = int(getattr(enc, "num_task_types", 0))
    side_dim = int(enc.side_dim)
    metric_heads = sorted({r["metric_type"] for r in records})

    agent = RWRv2Agent(
        state_dim=state_dim,
        a_values=mc.a_values,
        b_values=mc.b_values,
        metric_heads=metric_heads,
        num_heads=num_heads,
        num_metric_types=num_metric_types,
        num_task_types=num_task_types,
        side_dim=side_dim,
        hidden=args.hidden,
        task_emb_dim=args.task_emb_dim,
        metric_emb_dim=args.metric_emb_dim,
        dropout=args.dropout,
        decoding_mode=args.decoding_mode,
    ).to(device)
    print(f"RWRv2 agent params: {sum(p.numel() for p in agent.parameters())}", flush=True)

    opt = optim.Adam(agent.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0
    best_ckpt = os.path.join(args.save_dir, "policy_best.pt")
    final_ckpt = os.path.join(args.save_dir, "policy_final.pt")
    log = []

    def compute_target(rewards):
        # Per-sample adaptive T: T_i = max(floor, gap_i) / kappa
        sorted_r, _ = torch.sort(rewards, dim=-1, descending=True)
        gap = (sorted_r[:, 0] - sorted_r[:, 1]).clamp_min(args.T_floor)        # (B,)
        T = gap / args.kappa                                                    # (B,)
        target = F.softmax(rewards / T.unsqueeze(-1), dim=-1)                   # (B, 14)
        return target, T

    def eval_val():
        agent.eval()
        per_task_stats = {}
        losses, argmax_r, expected_r, oracle_r = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                st = batch["state"].to(device)
                r = batch["rewards"].to(device)
                out = agent.forward(st, metric_type=batch["metric_types"])
                logits = out["logits"]
                target, _ = compute_target(r)
                logp = F.log_softmax(logits, dim=-1)
                losses.append(float(-(target * logp).sum(-1).mean().item()))

                probs = torch.softmax(logits, dim=-1)
                am_idx = probs.argmax(-1)
                rr = r.gather(1, am_idx.unsqueeze(1)).squeeze(1)
                argmax_r.extend(rr.cpu().tolist())
                a_vals = agent.a_values.view(1, -1)
                a_exp = (probs * a_vals).sum(-1)
                # simple: round-to-nearest idx approximation for expected reward
                idx_from_a = torch.argmin(torch.abs(a_exp.unsqueeze(-1) - agent.a_values.unsqueeze(0)), dim=-1)
                rr_exp = r.gather(1, idx_from_a.unsqueeze(1)).squeeze(1)
                expected_r.extend(rr_exp.cpu().tolist())
                oracle_r.extend(r.max(-1).values.cpu().tolist())

                # per-task breakdown
                for i, t in enumerate(batch["task_types"]):
                    s = per_task_stats.setdefault(t, [[], []])
                    s[0].append(float(rr[i]))
                    s[1].append(float(r[i].max().item()))

        agent.train()
        summary = {
            "nll_mean": float(np.mean(losses)),
            "r_argmax": float(np.mean(argmax_r)),
            "r_expected": float(np.mean(expected_r)),
            "r_oracle": float(np.mean(oracle_r)),
            "per_task": {t: (float(np.mean(rs[0])), float(np.mean(rs[1])), len(rs[0]))
                         for t, rs in per_task_stats.items()},
        }
        return summary

    for epoch in range(1, args.epochs + 1):
        losses = []
        for batch in tr_loader:
            st = batch["state"].to(device)
            r = batch["rewards"].to(device)
            target, T_used = compute_target(r)
            out = agent.forward(st, metric_type=batch["metric_types"])
            logp = F.log_softmax(out["logits"], dim=-1)
            loss = -(target * logp).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))

        vm = eval_val()
        pt = "  ".join(f"{t[:8]}={vm['per_task'][t][0]:.3f}/{vm['per_task'][t][1]:.3f}"
                       for t in sorted(vm["per_task"].keys()))
        print(f"Epoch {epoch:3d}/{args.epochs}  L={np.mean(losses):.4f}  "
              f"r_arg={vm['r_argmax']:.4f}  r_exp={vm['r_expected']:.4f}  "
              f"oracle={vm['r_oracle']:.4f}  | {pt}", flush=True)
        log.append({"epoch": epoch, "loss": float(np.mean(losses)), **vm})

        sel = vm["r_argmax"] if args.select_by == "argmax" else (
            vm["r_expected"] if args.select_by == "expected" else
            max(vm["r_argmax"], vm["r_expected"]))
        if sel > best_val:
            best_val = sel
            torch.save({"iteration": epoch,
                        "agent_state_dict": agent.state_dict(),
                        "arch_config": _arch(args, agent)},
                       best_ckpt)

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save({"iteration": epoch,
                        "agent_state_dict": agent.state_dict(),
                        "arch_config": _arch(args, agent)},
                       os.path.join(args.save_dir, f"policy_epoch_{epoch}.pt"))

    torch.save({"iteration": args.epochs,
                "agent_state_dict": agent.state_dict(),
                "arch_config": _arch(args, agent)},
               final_ckpt)
    with open(os.path.join(args.save_dir, "train.log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nBest val r_argmax: {best_val:.4f}  →  {best_ckpt}")


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
        "task_emb_dim": int(args.task_emb_dim),
        "metric_emb_dim": int(args.metric_emb_dim),
        "dropout": float(args.dropout),
        "decoding_mode": str(args.decoding_mode),
        "rwr_v2": True,
        "rwr": False,
        "compact": False,
        "binary": False,
        "feature_dim": 256,
        "num_actions": int(agent.num_actions),
        "num_a_values": int(agent.num_a_values),
        "num_b_values": int(agent.num_b_values),
        "backbone_depth": 2,
        "T_floor": float(args.T_floor),
        "kappa": float(args.kappa),
    }


if __name__ == "__main__":
    main()
