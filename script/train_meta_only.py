"""Train and fast-evaluate a meta-only agent (seq_len + metric_oh + task_oh only).

Builds 18-dim states directly from training jsonl metadata — no LLM encoding needed.
Architecture: NeuralUCBAgent(state_dim=19, side_dim=1, num_views=1) — 1-dim dummy
side feature always zero, so only the 18-dim meta flows through the backbone.
Fast LB eval reuses runs/fast_lb_eval/lb_states_pre_rope_mean.pt (first 18 dims).

Usage:
    python script/train_meta_only.py --save_dir runs/meta_only_v0
"""
import argparse, json, os, random, sys, time
from typing import Dict, List

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL.a2sf_model import SIGMOID_A_VALUES, SIGMOID_B_VALUES
from RL.agent.neural_ucb_agent import NeuralUCBAgent
from RL.env.encoder import metric_type_to_index, task_type_to_index, METRIC_TYPE_ORDER, TASK_TYPE_ORDER

MAX_POS_EMBEDDINGS = 131072   # LLaMA-3.2-1B max_position_embeddings
META_DIM = 1 + len(METRIC_TYPE_ORDER) + len(TASK_TYPE_ORDER)  # seq_len + metric_oh + task_oh
DUMMY_DIM = 1                  # 1-dim zeroed dummy side feature
STATE_DIM  = META_DIM + DUMMY_DIM
SCORE_FIELD = "action_scores_maxo_by_budget"
SCORE_BUDGET = "128"
LB_STATES_PATH = "runs/fast_lb_eval/lb_states_pre_rope_mean.pt"


def build_meta_state(record: dict) -> torch.Tensor:
    """18-dim meta from jsonl record + 1-dim dummy → (19,)."""
    length = int(record.get("length", 0))
    seq_len_feat = min(float(length), MAX_POS_EMBEDDINGS) / MAX_POS_EMBEDDINGS

    metric_idx = metric_type_to_index(record.get("metric_type"))
    metric_oh = torch.zeros(len(METRIC_TYPE_ORDER))
    metric_oh[metric_idx] = 1.0

    task_idx = task_type_to_index(
        task_type=record.get("task_type"),
        dataset=record.get("dataset"),
    )
    task_oh = torch.zeros(len(TASK_TYPE_ORDER))
    task_oh[task_idx] = 1.0

    meta = torch.cat([
        torch.tensor([seq_len_feat]),
        metric_oh,
        task_oh,
        torch.zeros(DUMMY_DIM),   # dummy side feature
    ])
    return meta.float()


class MetaDataset(Dataset):
    def __init__(self, records: List[dict], score_field: str):
        self.samples = []
        for r in records:
            raw = r.get(score_field)
            scores = raw.get(SCORE_BUDGET) if isinstance(raw, dict) else raw
            if not scores or len(scores) != 13:
                continue
            state = build_meta_state(r)
            rewards = torch.tensor(scores, dtype=torch.float32)
            self.samples.append((state, rewards))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", default="datasets/training/scored/llama3-1b/train.jsonl")
    p.add_argument("--score_field", default=SCORE_FIELD)
    p.add_argument("--val_data_file", default=None)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--ucb_topk", type=int, default=4)
    p.add_argument("--ucb_beta", type=float, default=1.0)
    p.add_argument("--backbone_depth", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def fast_lb_eval(agent: NeuralUCBAgent, states_path: str) -> float:
    """Predict actions on precomputed LB states, return avg reward estimate."""
    if not os.path.exists(states_path):
        return float("nan")
    d = torch.load(states_path, map_location="cpu", weights_only=False)
    dataset_keys = sorted(k for k in d.keys() if k.endswith("/states"))

    # dataset -> action assignment file for scoring
    # We use the agent's argmax prediction on meta-only states
    # and compare against the per-sample val reward from the state file (if present).
    # If no reward info, return mean max pred.
    all_rewards = []
    for dk in dataset_keys:
        lb_states = d[dk].float()   # (N, 88)
        meta_states = lb_states[:, :META_DIM]  # (N, 18)
        # Append dummy dim
        dummy = torch.zeros(meta_states.size(0), DUMMY_DIM)
        states_19 = torch.cat([meta_states, dummy], dim=-1)  # (N, 19)
        with torch.no_grad():
            preds = agent.forward(states_19)["reward_pred"]  # (N, 13)
            max_pred = preds.max(dim=-1).values
        all_rewards.append(max_pred)
    return torch.cat(all_rewards).mean().item()


def val_r_argmax(agent, val_dataset) -> float:
    """Correlation between argmax(pred) and argmax(true) reward on val set."""
    if len(val_dataset) == 0:
        return float("nan")
    states = torch.stack([s for s, _ in val_dataset.samples])
    rewards = torch.stack([r for _, r in val_dataset.samples])
    with torch.no_grad():
        pred = agent.forward(states)["reward_pred"]
    pred_argmax = pred.argmax(dim=-1)
    true_argmax = rewards.argmax(dim=-1)
    # reward at predicted argmax vs true max
    n = states.size(0)
    pred_r = rewards[torch.arange(n), pred_argmax]
    oracle_r = rewards.max(dim=-1).values.clamp(min=1e-12)
    r_arg = (pred_r / oracle_r).mean().item()
    return r_arg


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    a_vals = torch.tensor(SIGMOID_A_VALUES, dtype=torch.float32)
    b_vals = torch.tensor(SIGMOID_B_VALUES, dtype=torch.float32)
    n_actions = len(a_vals)

    # Load data
    val_file = args.val_data_file or args.data_file.replace("/train/", "/validation/")
    print(f"Train: {args.data_file}")
    print(f"Val:   {val_file}")

    train_records, val_records = [], []
    def _valid(r):
        raw = r.get(args.score_field)
        scores = raw.get(SCORE_BUDGET) if isinstance(raw, dict) else raw
        return scores and len(scores) == n_actions

    with open(args.data_file) as f:
        for line in f:
            r = json.loads(line)
            if _valid(r):
                train_records.append(r)
    if os.path.exists(val_file):
        with open(val_file) as f:
            for line in f:
                r = json.loads(line)
                if _valid(r):
                    val_records.append(r)

    print(f"Train samples: {len(train_records)}, Val samples: {len(val_records)}")
    print(f"state_dim={STATE_DIM}  meta_dim={META_DIM}  (task types={len(TASK_TYPE_ORDER)}, metric types={len(METRIC_TYPE_ORDER)})")

    train_ds = MetaDataset(train_records, args.score_field)
    val_ds   = MetaDataset(val_records,   args.score_field)
    loader   = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    agent = NeuralUCBAgent(
        state_dim=STATE_DIM,
        a_values=a_vals,
        b_values=b_vals,
        num_metric_types=len(METRIC_TYPE_ORDER),
        num_task_types=len(TASK_TYPE_ORDER),
        side_dim=DUMMY_DIM,
        num_heads=1,
        backbone_depth=args.backbone_depth,
        dropout=0.0,
        paired_actions=True,
        num_hidden_pool=0,
        task_cond_head=True,
        num_views=1,
    )
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    print(f"Agent params: {sum(p.numel() for p in agent.parameters()):,}")

    best_val = -1.0
    best_epoch = 0
    log = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        agent.train()
        epoch_loss = 0.0
        n_batches = 0
        for states, rewards in loader:
            # UCB top-K selection
            with torch.no_grad():
                agent.eval()
                preds = agent.forward(states)["reward_pred"]  # (B, 13)
                agent.train()

            ucb_scores = preds   # beta=0 (no covariance update in fast loop)
            # top-K per sample
            topk = min(args.ucb_topk, n_actions)
            _, topk_idx = ucb_scores.topk(topk, dim=-1)  # (B, K)

            # gather rewards and preds for selected actions
            B = states.size(0)
            sel_rewards = rewards[torch.arange(B).unsqueeze(1), topk_idx]  # (B, K)

            # forward for grad
            preds2 = agent.forward(states)["reward_pred"]  # (B, 13)
            sel_preds = preds2[torch.arange(B).unsqueeze(1), topk_idx]    # (B, K)

            loss = F.mse_loss(sel_preds, sel_rewards)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        agent.eval()
        r_arg = val_r_argmax(agent, val_ds)
        if r_arg > best_val:
            best_val = r_arg
            best_epoch = epoch
            torch.save({"epoch": epoch, "agent_state_dict": agent.state_dict(),
                        "val_r_argmax": best_val}, f"{args.save_dir}/policy_best.pt")

        entry = {"epoch": epoch, "loss": epoch_loss / max(1, n_batches),
                 "val_r_argmax": r_arg}
        log.append(entry)

        if epoch % 20 == 0 or epoch == args.epochs:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:3d}/{args.epochs}  loss={entry['loss']:.5f}  "
                  f"val_r_arg={r_arg:.4f}  best={best_val:.4f}@{best_epoch}  "
                  f"({elapsed:.0f}s)", flush=True)

    torch.save({"epoch": args.epochs, "agent_state_dict": agent.state_dict()},
               f"{args.save_dir}/policy_final.pt")
    with open(f"{args.save_dir}/train.log.json", "w") as f:
        json.dump(log, f)

    print(f"\nTraining done. Best val_r_argmax={best_val:.4f} @ epoch {best_epoch}")
    print(f"Checkpoint: {args.save_dir}/policy_best.pt")


if __name__ == "__main__":
    main()
