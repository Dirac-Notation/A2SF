"""Joint training: ChunkAttnEncoder + NeuralUCBAgent end-to-end.

The LLaMA embed_tokens + input_layernorm are FROZEN.
ChunkAttnEncoder and NeuralUCBAgent are trained jointly.

State layout: [meta(18) | encoded(128)] = 146-dim
  meta = [seq_len(1), metric_oh(10), task_oh(7)]
  encoded = ChunkAttnEncoder output (128-dim)

Usage:
    python RL/train_joint.py \\
        --model llama3-1b \\
        --save_dir runs/joint_v0 \\
        --data_file datasets/training/scored/llama3-1b/train.jsonl \\
        --epochs 100 --lr 3e-4 --batch_size 32 --gpu 0
"""
from __future__ import annotations

import argparse, json, math, os, random, sys, time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL.a2sf_model import ModelConfig, SIGMOID_A_VALUES, SIGMOID_B_VALUES
from RL.agent.neural_ucb_agent import NeuralUCBAgent
from RL.env.chunk_attn_encoder import ChunkAttnEncoder
from RL.env.encoder import (
    METRIC_TYPE_ORDER, TASK_TYPE_ORDER,
    metric_type_to_index, task_type_to_index,
)

SCORE_BUDGET = "128"
META_DIM     = 1 + len(METRIC_TYPE_ORDER) + len(TASK_TYPE_ORDER)  # 18


# ──────────────────────────────────────────────────────────────
# Gradient flow test
# ──────────────────────────────────────────────────────────────

def test_gradient_flow(encoder: ChunkAttnEncoder, agent: NeuralUCBAgent,
                       d_model: int, device: torch.device):
    """
    Verify that gradients reach every trainable parameter.
    Uses random dummy embeddings (no LLaMA needed for the test).
    Runs two lengths: short (L=64) and long (L=512) to exercise
    the sliced-linear at different n_chunks values.
    """
    print("\n" + "═" * 60)
    print("Gradient flow test")
    print("═" * 60)

    a_vals = torch.tensor(SIGMOID_A_VALUES, dtype=torch.float32)
    b_vals = torch.tensor(SIGMOID_B_VALUES, dtype=torch.float32)
    n_actions = len(a_vals)

    encoder.train()
    agent.train()
    all_params = dict(encoder.named_parameters())
    all_params.update({f"agent.{k}": v for k, v in agent.named_parameters()})

    issues = []

    n_tasks   = len(TASK_TYPE_ORDER)
    n_metrics = len(METRIC_TYPE_ORDER)

    for L in [64, 512]:
        # Zero grads
        for p in list(encoder.parameters()) + list(agent.parameters()):
            p.grad = None

        # One sample per task so ALL task_heads receive gradient
        states_list = []
        for t in range(n_tasks):
            embeds  = torch.randn(L, d_model, device=device)
            encoded = encoder.forward_embeds(embeds.float())
            meta    = torch.zeros(META_DIM, device=device)
            meta[0] = 0.5
            meta[1] = 1.0                          # metric_oh[0]
            meta[1 + n_metrics + t] = 1.0          # task_oh[t]
            states_list.append(torch.cat([meta, encoded]))
        state  = torch.stack(states_list, dim=0)   # (n_tasks, 146)
        out    = agent.forward(state)
        reward = out["reward_pred"]                # (n_tasks, 13)

        loss = F.mse_loss(reward, torch.rand_like(reward))
        loss.backward()

        n_chunks = math.ceil(L / encoder.chunk_size)

        print(f"\n  L={L}  n_chunks={n_chunks}  loss={loss.item():.4f}")
        print(f"  {'Parameter':45s}  {'grad_norm':>12}  status")
        print(f"  {'─'*45}  {'─'*12}  {'─'*6}")

        for name, param in encoder.named_parameters():
            if param.grad is None:
                g = "NONE ✗"
                issues.append(f"L={L}: encoder.{name} — NO GRAD")
            else:
                gn = param.grad.norm().item()
                g = f"{gn:.2e}  ✓"
            print(f"  encoder.{name:36s}  {g}")

        for name, param in agent.named_parameters():
            if name.startswith("inverse_lambdas") or name.startswith("action_counts"):
                continue  # buffers, not trained
            if param.grad is None:
                g = "NONE ✗"
                issues.append(f"L={L}: agent.{name} — NO GRAD")
            else:
                gn = param.grad.norm().item()
                g = f"{gn:.2e}  ✓"
            print(f"  agent.{name:38s}  {g}")

        # Special check: sliced_w — only last n_chunks cols should have grad
        sw_grad = encoder.sliced_w.grad
        if sw_grad is not None:
            active_cols = (sw_grad[:, -n_chunks:].abs().sum(0) > 0).sum().item()
            dead_cols   = (sw_grad[:, :-n_chunks].abs().sum(0) > 0).sum().item()
            print(f"\n  sliced_w: active cols = {active_cols}/{n_chunks}  "
                  f"(dead cols with grad = {dead_cols}, expected 0)")
            if active_cols < n_chunks:
                issues.append(f"L={L}: sliced_w only {active_cols}/{n_chunks} cols have grad")

    print("\n" + "─" * 60)
    if issues:
        print("✗ ISSUES FOUND:")
        for iss in issues:
            print(f"  • {iss}")
        raise RuntimeError("Gradient flow test FAILED. Fix issues before training.")
    else:
        print("✓ PASS — gradients flow to all parameters.")
    print("═" * 60 + "\n")


# ──────────────────────────────────────────────────────────────
# Meta state helpers
# ──────────────────────────────────────────────────────────────

def build_meta(record: dict, max_seq_len: float = 131072.0) -> torch.Tensor:
    """Returns (META_DIM,) float32."""
    length = int(record.get("length", 0))
    seq_feat = min(float(length), max_seq_len) / max_seq_len

    m_idx = metric_type_to_index(record.get("metric_type"))
    m_oh  = torch.zeros(len(METRIC_TYPE_ORDER))
    m_oh[m_idx] = 1.0

    t_idx = task_type_to_index(task_type=record.get("task_type"),
                                dataset=record.get("dataset"))
    t_oh  = torch.zeros(len(TASK_TYPE_ORDER))
    t_oh[t_idx] = 1.0

    return torch.cat([torch.tensor([seq_feat]), m_oh, t_oh])   # (18,)


def get_scores(record: dict, n_actions: int, score_field: str):
    raw = record.get(score_field)
    scores = raw.get(SCORE_BUDGET) if isinstance(raw, dict) else raw
    if scores and len(scores) == n_actions:
        return torch.tensor(scores, dtype=torch.float32)
    return None


# ──────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(encoder: ChunkAttnEncoder, agent: NeuralUCBAgent,
             records: List[dict], n_actions: int, device: torch.device,
             score_field: str, max_samples: int = 200) -> float:
    encoder.eval(); agent.eval()
    subset = records[:max_samples]
    r_args = []
    for rec in subset:
        reward_t = get_scores(rec, n_actions, score_field)
        if reward_t is None:
            continue
        encoded = encoder.encode_context(rec["input_prompt"], detach=True).to(device)
        meta    = build_meta(rec).to(device)
        state   = torch.cat([meta, encoded]).unsqueeze(0)
        pred    = agent.forward(state)["reward_pred"].squeeze(0)   # (13,)
        pred_a  = int(pred.argmax())
        oracle  = float(reward_t.max())
        r_args.append(float(reward_t[pred_a]) / max(oracle, 1e-12))
    encoder.train(); agent.train()
    return float(sum(r_args) / max(len(r_args), 1))


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="llama3-1b")
    p.add_argument("--save_dir",    required=True)
    p.add_argument("--data_file",   default="datasets/training/scored/llama3-1b/train.jsonl")
    p.add_argument("--val_data_file", default=None)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--ucb_topk",    type=int,   default=4)
    p.add_argument("--ucb_beta",    type=float, default=1.0)
    p.add_argument("--hidden",      type=int,   default=128)
    p.add_argument("--n_heads",     type=int,   default=4)
    p.add_argument("--chunk_size",  type=int,   default=16)
    p.add_argument("--max_chunks",  type=int,   default=2048)
    p.add_argument("--max_input_length", type=int, default=32768)
    p.add_argument("--score_field", default="action_scores_maxo_by_budget",
                   choices=["action_scores_maxo_by_budget",
                            "action_scores_fc_by_budget",
                            "action_scores_gt_by_budget"])
    p.add_argument("--gpu",         default="0")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load LLaMA (frozen, for embed_tokens only) ────────────────────
    print("Loading target model …")
    from RL.a2sf_model import ModelConfig
    from RL.env.model_runner import A2SFModelRunner
    mc = ModelConfig.sigmoid(model=args.model)
    runner = A2SFModelRunner(mc)
    target_model    = runner.model
    target_tokenizer= runner.tokenizer
    # Freeze entire target model
    for p in target_model.parameters():
        p.requires_grad_(False)
    target_model.eval()
    print(f"  {args.model} loaded (frozen)")

    d_model = int(target_model.config.hidden_size)

    # ── Build encoder ─────────────────────────────────────────────────
    encoder = ChunkAttnEncoder(
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        d_model=d_model,
        hidden=args.hidden,
        n_heads=args.n_heads,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
        max_input_length=args.max_input_length,
    ).to(device)
    enc_params = sum(p.numel() for p in encoder.parameters())
    print(f"  ChunkAttnEncoder: {enc_params:,} params")

    # ── Build agent ───────────────────────────────────────────────────
    a_vals    = torch.tensor(SIGMOID_A_VALUES, dtype=torch.float32)
    b_vals    = torch.tensor(SIGMOID_B_VALUES, dtype=torch.float32)
    n_actions = len(a_vals)
    state_dim = META_DIM + args.hidden   # 18 + 128 = 146

    agent = NeuralUCBAgent(
        state_dim       = state_dim,
        a_values        = a_vals,
        b_values        = b_vals,
        num_metric_types= len(METRIC_TYPE_ORDER),
        num_task_types  = len(TASK_TYPE_ORDER),
        side_dim        = args.hidden,   # 128-dim encoded as single view
        num_heads       = 1,
        backbone_depth  = 2,
        dropout         = 0.0,
        paired_actions  = True,
        num_hidden_pool = 0,
        task_cond_head  = True,
        num_views       = 1,
    ).to(device)
    agent_params = sum(p.numel() for p in agent.parameters())
    print(f"  NeuralUCBAgent:   {agent_params:,} params")
    print(f"  Total trainable:  {enc_params + agent_params:,} params")

    # ── Gradient flow test ────────────────────────────────────────────
    test_gradient_flow(encoder, agent, d_model, device)

    # ── Load data ─────────────────────────────────────────────────────
    val_file = args.val_data_file or args.data_file.replace("/train/", "/validation/")
    train_recs, val_recs = [], []
    with open(args.data_file) as f:
        for line in f:
            r = json.loads(line)
            if r.get("input_prompt") and get_scores(r, n_actions, args.score_field) is not None:
                train_recs.append(r)
    if os.path.exists(val_file):
        with open(val_file) as f:
            for line in f:
                r = json.loads(line)
                if r.get("input_prompt") and get_scores(r, n_actions, args.score_field) is not None:
                    val_recs.append(r)
    print(f"Train: {len(train_recs)}  Val: {len(val_recs)}")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(agent.parameters()),
        lr=args.lr
    )

    best_val = -1.0
    log = []
    t0  = time.time()

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_recs)
        encoder.train(); agent.train()
        epoch_loss = 0.0
        n_batches  = 0

        # Accumulate batch_size encoded states, then one agent update
        i = 0
        while i < len(train_recs):
            batch = train_recs[i:i + args.batch_size]
            i    += args.batch_size

            # ── Encode batch (sequential, gradients kept) ──────────────
            states_list  = []
            rewards_list = []
            for rec in batch:
                reward_t = get_scores(rec, n_actions, args.score_field)
                if reward_t is None:
                    continue
                embeds  = encoder._get_embeds(
                    target_tokenizer(
                        rec["input_prompt"], return_tensors="pt",
                        truncation=True, max_length=args.max_input_length,
                        add_special_tokens=False,
                    ).input_ids.squeeze(0)
                )                                           # (L, d_model), detached
                encoded = encoder.forward_embeds(embeds)   # (128,), grad flows
                meta    = build_meta(rec).to(device)
                states_list.append(torch.cat([meta, encoded]))  # (146,)
                rewards_list.append(reward_t.to(device))

            if not states_list:
                continue

            states  = torch.stack(states_list,  dim=0)    # (B, 146)
            rewards = torch.stack(rewards_list, dim=0)    # (B, 13)

            # ── UCB top-K selection (no grad needed for selection) ─────
            with torch.no_grad():
                agent.eval()
                preds_no_grad = agent.forward(states)["reward_pred"]
                agent.train()

            topk     = min(args.ucb_topk, n_actions)
            _, top_idx = preds_no_grad.topk(topk, dim=-1)          # (B, K)

            # ── Forward with grad, loss on top-K ──────────────────────
            preds    = agent.forward(states)["reward_pred"]         # (B, 13)
            B = states.size(0)
            sel_pred = preds[torch.arange(B).unsqueeze(1), top_idx]  # (B, K)
            sel_tgt  = rewards[torch.arange(B).unsqueeze(1), top_idx]

            loss = F.mse_loss(sel_pred, sel_tgt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(agent.parameters()), 1.0
            )
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        # ── Validation ────────────────────────────────────────────────
        val_r = evaluate(encoder, agent, val_recs, n_actions, device, args.score_field) \
                if val_recs else float("nan")

        if not math.isnan(val_r) and val_r > best_val:
            best_val = val_r
            torch.save({
                "epoch":   epoch,
                "encoder": encoder.state_dict(),
                "agent":   agent.state_dict(),
                "arch": {
                    "hidden":      args.hidden,
                    "n_heads":     args.n_heads,
                    "chunk_size":  args.chunk_size,
                    "max_chunks":  args.max_chunks,
                    "state_dim":   state_dim,
                    "a_values":    a_vals,
                    "b_values":    b_vals,
                    "num_metric_types": len(METRIC_TYPE_ORDER),
                    "num_task_types":   len(TASK_TYPE_ORDER),
                },
                "val_r_argmax": best_val,
            }, f"{args.save_dir}/best.pt")

        entry = {"epoch": epoch,
                 "loss":  epoch_loss / max(1, n_batches),
                 "val_r": val_r}
        log.append(entry)

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:3d}/{args.epochs}  "
                  f"loss={entry['loss']:.5f}  "
                  f"val_r={val_r:.4f}  best={best_val:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    torch.save({"epoch": args.epochs,
                "encoder": encoder.state_dict(),
                "agent":   agent.state_dict()},
               f"{args.save_dir}/final.pt")
    with open(f"{args.save_dir}/train.log.json", "w") as f:
        json.dump(log, f)

    print(f"\nDone. Best val_r_argmax={best_val:.4f}")
    print(f"Log: {args.save_dir}/train.log.json")


if __name__ == "__main__":
    main()
