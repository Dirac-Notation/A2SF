"""Train the SimpleUCBAgent (champion) on sigmoid-paired action data.

Recipe:
  python RL/train.py --model llama3-1b --budget 128 \
    --data_file datasets/training/scored/llama3-1b/train.jsonl \
    --score_field     action_scores_maxo_by_budget \
    --val_score_field action_scores_maxo_by_budget \
    --mini_attn_ckpt  runs/mini_attn_v5/mini_attn_best.pt \
    --save_dir runs/simple_ucb_v5_maxo

Loss: per-batch top-K UCB selection + MSE on selected actions, with
per-task Σ⁻¹ Sherman-Morrison update. Inference: argmax(reward_pred).
Ckpt selection: best val r_argmax.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL.a2sf_model import ModelConfig
from RL.env import A2SFModelRunner, A2SFEnv
from RL.agent.neural_ucb_agent import NeuralUCBAgent
from RL.agent.lora_ucb_agent import LoRAUCBAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="llama3-1b")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--data_file", type=str,
                   default="datasets/training/scored/llama3-1b/train.jsonl",
                   help="Train split jsonl. Validation jsonl inferred by replacing "
                        "'train/' with 'validation/' in this path (override with --val_data_file).")
    p.add_argument("--val_data_file", type=str, default=None)
    p.add_argument("--score_field", type=str,
                   default="action_scores_maxo_by_budget",
                   help="Per-budget reward dict used as training target.")
    p.add_argument("--val_score_field", type=str,
                   default="action_scores_maxo_by_budget",
                   help="Per-budget reward dict used for val metric (independent of training).")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    # UCB exploration
    p.add_argument("--loss", choices=["mse", "listwise"], default="mse",
                   help="listwise = the repro_2694 recipe loss (reconstructed 2026-08-26, "
                        "bit-exact vs runs/repro_2694_seed42/train.log.json).")
    p.add_argument("--loss_temp", type=float, default=0.1,
                   help="listwise softmax temperature (repro_2694 recipe uses 0.1).")
    p.add_argument("--ucb_topk", type=int, default=4,
                   help="MSE loss is computed over the top-K UCB-ranked actions per sample.")
    p.add_argument("--ucb_beta", type=float, default=1.0,
                   help="Exploration bonus scale: UCB = reward_pred + beta·sqrt(featᵀ Σ⁻¹ feat).")

    # Encoder (frozen mini-attn)
    p.add_argument("--encoder_topk", type=int, default=16,
                   help="Number of top-K positions per head in stats feature mode.")
    p.add_argument("--mini_attn_ckpt", type=str, default="runs/mini_attn_v5/mini_attn_best.pt",
                   help="Frozen MiniAttnEncoder checkpoint.")
    p.add_argument("--encoder_max_input_length", type=int, default=32768)
    p.add_argument("--encoder_include_hidden_pool", action="store_true", default=True,
                   help="Champion always-on default (history file 9). Adds hidden_size dims to state.")
    p.add_argument("--no_encoder_include_hidden_pool", dest="encoder_include_hidden_pool",
                   action="store_false")
    p.add_argument("--encoder_hidden_pool_window", type=int, default=0,
                   help="0=full prompt mean, N>0=mean over last N tokens.")
    p.add_argument("--action_subset", type=str, default="full",
                   choices=["full", "hard"],
                   help="full=all 13 actions; hard=only hard-like+a=0 (5 actions, ablation Config B).")
    # Extra view experiments (A-D)
    p.add_argument("--extra_view", type=str, default="none",
                   choices=["none", "pre_rope_mean", "pre_rope_max", "pre_rope_z_max",
                            "snap_l0", "position_prior"],
                   help="Append extra 35-d view after mini_attn stats. 'none'=champion baseline.")
    p.add_argument("--pre_rope_query_window", type=int, default=16,
                   help="Query window size for pre_rope_* and snap_l0 extra views.")
    p.add_argument("--encoder_feature_mode", type=str, default="stats",
                   choices=["stats", "endaligned"],
                   help="stats=top-K positions (35d); endaligned=256-bin distribution (259d). "
                        "Experiment C uses endaligned.")
    # Pre-computed states (skip encoding phase)
    p.add_argument("--states_file", type=str, default=None,
                   help="Path to .pt file with pre-computed states {prompt_id→Tensor}. "
                        "If provided, skips model loading and encoding.")
    p.add_argument("--save_states", type=str, default=None,
                   help="After encoding, save {prompt_id→Tensor} dict to this .pt path.")

    # Agent variant
    p.add_argument("--agent_variant", type=str, default="neural_ucb",
                   choices=["neural_ucb", "lora_ucb"],
                   help="neural_ucb: champion (MLPResidualBlock + per-task residual). "
                        "lora_ucb: per-task LoRA inside MLPResidualBlock + per-metric heads only.")
    p.add_argument("--metric_heads_list", type=str,
                   default="qa_f1_score,rouge_score,code_sim_score,classification_score,retrieval_score",
                   help="Comma-separated metric heads (for lora_ucb). Default: 5 English LB metrics.")
    p.add_argument("--block_hidden", type=int, default=512,
                   help="Hidden dim inside each LoRAResidualBlock (lora_ucb).")
    p.add_argument("--lora_rank", type=int, default=4,
                   help="LoRA rank for per-task delta in lora_ucb backbone.")
    p.add_argument("--backbone_route_by", type=str, default="task", choices=["task", "metric"],
                   help="lora_ucb: which signal indexes per-route LoRA in backbone.")
    p.add_argument("--head_route_by", type=str, default="metric", choices=["task", "metric"],
                   help="lora_ucb: which signal selects per-route head.")
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

    # Action subset for ablation. When "hard", we take indices [0, 9, 10, 11, 12]
    # (a=0 + a=10 paired with each b) — drops the gradual sigmoids a∈{0.01, 0.1}.
    from RL.a2sf_model import HARD_LIKE_INDICES
    if args.action_subset == "hard":
        action_indices = list(HARD_LIKE_INDICES)
    else:
        action_indices = None  # full grid

    # Detect action count from first record's score field.
    expected_action_size = None
    with open(args.data_file) as _f:
        for _line in _f:
            _r = json.loads(_line)
            _sc = _r.get(args.score_field, {}).get(bkey, [])
            if isinstance(_sc, list) and len(_sc) > 0:
                expected_action_size = len(_sc); break
    if expected_action_size is None:
        raise ValueError(f"No '{args.score_field}' found in {args.data_file}")
    print(f"expected_action_size={expected_action_size}")

    val_data_file = args.val_data_file or args.data_file.replace("/train/", "/validation/")
    print(f"Train data: {args.data_file}")
    print(f"Val data:   {val_data_file}")

    tr_records, val_records = [], []
    with open(args.data_file) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            tr_sc = r.get(args.score_field, {}).get(bkey, [])
            if not isinstance(tr_sc, list) or len(tr_sc) != expected_action_size or not any(float(x) > 0 for x in tr_sc):
                continue
            rewards_full = [float(x) for x in tr_sc]
            rewards_keep = (
                [rewards_full[i] for i in action_indices] if action_indices is not None
                else rewards_full
            )
            tr_records.append({
                "_prompt_id": i,
                "prompt": r["input_prompt"],
                "metric_type": str(r.get("metric_type", "qa_f1_score")),
                "task_type": str(r.get("task_type", "unknown")),
                "dataset": r.get("dataset"),
                "generation_length": int(r.get("generation_length", 0)),
                "length": int(r.get("length", 0)),
                "rewards": rewards_keep,
            })

    if not os.path.exists(val_data_file):
        raise FileNotFoundError(f"Validation data file not found: {val_data_file}")
    val_offset = len(tr_records)
    with open(val_data_file) as f:
        for j, line in enumerate(f):
            r = json.loads(line)
            val_sc = r.get(args.val_score_field, {}).get(bkey, [])
            if not isinstance(val_sc, list) or len(val_sc) != expected_action_size:
                continue
            val_full = [float(x) for x in val_sc]
            val_keep = (
                [val_full[i] for i in action_indices] if action_indices is not None
                else val_full
            )
            val_records.append({
                "_prompt_id": val_offset + j,
                "prompt": r["input_prompt"],
                "metric_type": str(r.get("metric_type", "qa_f1_score")),
                "task_type": str(r.get("task_type", "unknown")),
                "dataset": r.get("dataset"),
                "generation_length": int(r.get("generation_length", 0)),
                "length": int(r.get("length", 0)),
                "rewards": val_keep,
            })

    records = tr_records + val_records
    print(f"Loaded {len(tr_records)} train + {len(val_records)} val samples")

    # ── Encoder / state-vector setup ──────────────────────────────────────────
    mc = ModelConfig.sigmoid(model=args.model)
    mc.encoder_topk = int(args.encoder_topk)
    mc.mini_attn_ckpt = str(args.mini_attn_ckpt or "")
    mc.encoder_max_input_length = int(args.encoder_max_input_length)
    mc.encoder_include_hidden_pool = bool(args.encoder_include_hidden_pool)
    mc.encoder_hidden_pool_window = int(args.encoder_hidden_pool_window)
    mc.extra_view = str(args.extra_view)
    mc.pre_rope_query_window = int(args.pre_rope_query_window)
    mc.encoder_feature_mode = str(args.encoder_feature_mode)
    # extra_view needs single_view=True: [meta | mini(35) | extra(35)]
    # single_view=False duplicates mini to 70d which breaks the 2-view layout
    if args.extra_view != "none":
        mc.single_view = True
    if action_indices is not None:
        idx_t = torch.tensor(action_indices, dtype=torch.long)
        mc.a_values = mc.a_values[idx_t].clone()
        mc.b_values = mc.b_values[idx_t].clone()
        print(f"action_subset='{args.action_subset}' → {len(action_indices)} actions: "
              f"a={mc.a_values.tolist()}, b={mc.b_values.tolist()}", flush=True)

    prompt_to_state: Dict[int, torch.Tensor] = {}

    if args.states_file and os.path.exists(args.states_file):
        # Fast path: load pre-computed states, skip model loading entirely.
        print(f"Loading pre-computed states from {args.states_file} …")
        saved = torch.load(args.states_file, map_location="cpu", weights_only=False)
        for k, v in saved.items():
            if isinstance(k, int):
                prompt_to_state[k] = v
        state_dim = int(saved["state_dim"])
        num_metric_types = int(saved.get("num_metric_types", len([])))
        num_task_types = int(saved.get("num_task_types", 0))
        side_dim = int(saved.get("side_dim", 0))
        num_heads = int(saved.get("num_heads", 1))
        num_hidden_pool = int(saved.get("num_hidden_pool", 0))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loaded {len(prompt_to_state)} states (state_dim={state_dim})", flush=True)
    else:
        print("Loading encoder …")
        runner = A2SFModelRunner(mc)
        env = A2SFEnv(runner, mc)
        device = next(runner.model.model.layers[0].parameters()).device
        env.device = device
        enc = env.context_encoder

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

        if args.save_states:
            os.makedirs(os.path.dirname(args.save_states) or ".", exist_ok=True)
            save_dict = dict(prompt_to_state)
            save_dict["state_dim"] = int(enc.output_dim)
            save_dict["num_metric_types"] = int(enc.num_metric_types)
            save_dict["num_task_types"] = int(getattr(enc, "num_task_types", 0))
            save_dict["side_dim"] = int(getattr(enc, "side_dim", 0))
            save_dict["num_heads"] = int(getattr(enc, "num_heads", 1))
            save_dict["num_hidden_pool"] = int(getattr(enc, "hidden_pool_dim", 0))
            torch.save(save_dict, args.save_states)
            print(f"Saved states → {args.save_states}", flush=True)

        state_dim = int(enc.output_dim)
        num_metric_types = int(enc.num_metric_types)
        num_task_types = int(getattr(enc, "num_task_types", 0))
        side_dim = int(getattr(enc, "side_dim", 0))
        num_heads = int(getattr(enc, "num_heads", 1))
        num_hidden_pool = int(getattr(enc, "hidden_pool_dim", 0))

    tr_loader = DataLoader(RWRDataset(tr_records, prompt_to_state),
                           batch_size=args.batch_size, shuffle=True,
                           num_workers=0, collate_fn=_collate)
    val_loader = DataLoader(RWRDataset(val_records, prompt_to_state),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=_collate)

    if args.agent_variant == "lora_ucb":
        metric_heads_list = [m.strip() for m in args.metric_heads_list.split(",") if m.strip()]
        agent = LoRAUCBAgent(
            state_dim=state_dim,
            a_values=mc.a_values,
            b_values=mc.b_values,
            num_metric_types=num_metric_types,
            num_task_types=num_task_types,
            metric_heads=metric_heads_list,
            hidden=int(args.hidden),
            block_hidden=int(args.block_hidden),
            lora_rank=int(args.lora_rank),
            dropout=float(args.dropout),
            paired_actions=True,
            include_seq_len=True,
            backbone_route_by=str(args.backbone_route_by),
            head_route_by=str(args.head_route_by),
        ).to(device)
    else:
        agent = NeuralUCBAgent(
            state_dim=state_dim,
            a_values=mc.a_values,
            b_values=mc.b_values,
            num_metric_types=num_metric_types,
            num_task_types=num_task_types,
            side_dim=side_dim,
            num_heads=num_heads,
            num_hidden_pool=num_hidden_pool,
            backbone_depth=2,
            dropout=float(args.dropout),
            paired_actions=True,
            task_cond_head=True,
            task_head_mlp=False,
            output_activation="sigmoid",
            num_views=(1 if args.extra_view == "none" else 2),
            include_seq_len=True,
        ).to(device)
    print(f"agent params={sum(p.numel() for p in agent.parameters())}", flush=True)

    opt = optim.AdamW(
        [{"params": list(agent.parameters()), "lr": args.lr, "weight_decay": args.weight_decay}]
    )

    best_val = -1.0
    best_ckpt = os.path.join(args.save_dir, "policy_best.pt")
    final_ckpt = os.path.join(args.save_dir, "policy_final.pt")
    log = []

    def eval_val():
        agent.eval()
        per_task_stats: Dict[str, list] = {}
        argmax_r, oracle_r = [], []
        pred_mean_all, pred_at_argmax_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                st = batch["state"].to(device)
                r = batch["rewards"].to(device)
                out = agent.forward(st, metric_type=batch["metric_types"])
                scores = out["reward_pred"]
                am_idx = scores.argmax(-1)
                rr = r.gather(1, am_idx.unsqueeze(1)).squeeze(1)
                argmax_r.extend(rr.cpu().tolist())
                oracle_r.extend(r.max(-1).values.cpu().tolist())
                pred_mean_all.extend(scores.mean(-1).cpu().tolist())
                pred_at_argmax_all.extend(
                    scores.gather(1, am_idx.unsqueeze(1)).squeeze(1).cpu().tolist()
                )
                for i, t in enumerate(batch["task_types"]):
                    s = per_task_stats.setdefault(t, [[], []])
                    s[0].append(float(rr[i]))
                    s[1].append(float(r[i].max().item()))
        agent.train()
        return {
            "r_argmax": float(np.mean(argmax_r)),
            "r_oracle": float(np.mean(oracle_r)),
            "pred_mean": float(np.mean(pred_mean_all)),
            "pred_at_argmax": float(np.mean(pred_at_argmax_all)),
            "per_task": {t: (float(np.mean(rs[0])), float(np.mean(rs[1])), len(rs[0]))
                         for t, rs in per_task_stats.items()},
        }

    for epoch in range(1, args.epochs + 1):
        losses = []
        for batch in tr_loader:
            st = batch["state"].to(device)
            r = batch["rewards"].to(device)
            metric_types_batch = batch["metric_types"]
            out = agent.forward(st, metric_type=metric_types_batch)
            reward_pred = out["reward_pred"]      # (B, A), sigmoid output
            feat = out["feature_vector"]           # (B, H)

            # UCB top-K selection — per-sample head-route-aware Σ⁻¹.
            with torch.no_grad():
                feat_d = feat.detach()
                head_idx = agent._head_idx_for_state(st, metric_types_batch)
                invs_batch = agent.inverse_lambdas[head_idx]    # (B, A, H, H)
                unc = torch.einsum("bi,baij,bj->ba", feat_d, invs_batch, feat_d)
                ucb_score = reward_pred.detach() + args.ucb_beta * torch.sqrt(unc + 1e-6)
                K = min(args.ucb_topk, reward_pred.size(-1))
                _, topk_idx = ucb_score.topk(K, dim=-1)

            pred_k = reward_pred.gather(1, topk_idx)   # (B, K) with grad
            true_k = r.gather(1, topk_idx)
            if args.loss == "listwise":
                # ListNet CE over the top-K UCB-selected actions; temperature on the
                # TARGET only (predictions unscaled). Reconstruction of the original lost
                # uncommitted at history #54 - validated BIT-EXACT against
                # runs/repro_2694_seed42/train.log.json (seed 42: losses and r_argmax
                # identical to full float precision over the first 3 epochs, 2026-08-26).
                target = F.softmax(true_k / args.loss_temp, dim=-1)
                logp = F.log_softmax(pred_k, dim=-1)
                loss = -(target * logp).sum(dim=-1).mean()
            else:
                loss = F.mse_loss(pred_k, true_k)

            opt.zero_grad(); loss.backward(); opt.step()

            # Σ⁻¹ rank-1 update — broadcast head_idx over K, pass per-sample tensor.
            with torch.no_grad():
                feat_d = feat.detach()
                feat_exp = feat_d.unsqueeze(1).expand(-1, K, -1).reshape(-1, feat_d.size(-1))
                act_flat = topk_idx.reshape(-1)
                head_idx_kth = head_idx.unsqueeze(1).expand(-1, K).reshape(-1)
                agent._update_inverse_covariances(feat_exp, act_flat, head_idx_kth)

            losses.append(float(loss.item()))

        vm = eval_val()
        pt = "  ".join(f"{t[:8]}={vm['per_task'][t][0]:.3f}/{vm['per_task'][t][1]:.3f}"
                       for t in sorted(vm["per_task"].keys()))
        print(f"Epoch {epoch:3d}/{args.epochs}  L={np.mean(losses):.4f}  "
              f"r_arg={vm['r_argmax']:.4f}  oracle={vm['r_oracle']:.4f}  "
              f"p_mean={vm['pred_mean']:.4f}  p_arg={vm['pred_at_argmax']:.4f}  "
              f"| {pt}", flush=True)
        log.append({"epoch": epoch, "loss": float(np.mean(losses)), **vm})

        if vm["r_argmax"] > best_val:
            best_val = vm["r_argmax"]
            torch.save({"iteration": epoch,
                        "agent_state_dict": agent.state_dict(),
                        "arch_config": _arch(args, agent)}, best_ckpt)

    torch.save({"iteration": args.epochs,
                "agent_state_dict": agent.state_dict(),
                "arch_config": _arch(args, agent)}, final_ckpt)
    with open(os.path.join(args.save_dir, "train.log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nBest val r_argmax: {best_val:.4f}  →  {best_ckpt}")


def _arch(args, agent):
    """Save arch_config for ckpt — variant-aware so loader can rebuild correctly."""
    common = {
        "state_dim": int(agent.state_dim),
        "num_metric_types": int(agent.num_metric_types),
        "num_task_types": int(agent.num_task_types),
        "dropout": float(args.dropout),
        "a_values": agent.a_values.detach().cpu(),
        "b_values": agent.b_values.detach().cpu(),
        "num_actions": int(agent.num_actions),
        "paired_actions": bool(getattr(agent, "paired_actions", True)),
        "include_seq_len": bool(getattr(agent, "include_seq_len", True)),
        # Encoder reproducibility
        "compression_method": "sigmoid",
        "encoder_topk": int(args.encoder_topk),
        "mini_attn_ckpt": str(args.mini_attn_ckpt or ""),
        "encoder_max_input_length": int(args.encoder_max_input_length),
        "encoder_feature_mode": str(args.encoder_feature_mode),
        "single_view": (args.extra_view != "none"),  # True when extra_view is used
        "encoder_views": "both",
        "encoder_include_hidden_pool": bool(args.encoder_include_hidden_pool),
        "encoder_hidden_pool_window": int(args.encoder_hidden_pool_window),
        "extra_view": str(args.extra_view),
        "pre_rope_query_window": int(args.pre_rope_query_window),
        "ucb_topk": int(args.ucb_topk),
        "ucb_beta": float(args.ucb_beta),
    }
    if args.agent_variant == "lora_ucb":
        common.update({
            "variant": "lora_ucb",
            "metric_heads": list(agent.metric_heads),
            "hidden": int(agent.hidden),
            "block_hidden": int(agent.block_hidden),
            "lora_rank": int(agent.lora_rank),
            "backbone_route_by": str(agent.backbone_route_by),
            "head_route_by": str(agent.head_route_by),
        })
    else:
        common.update({
            "variant": "ucb",
            "side_dim": int(agent.side_dim),
            "num_heads": int(agent.num_heads),
            "num_views": int(agent.num_views),
            "num_hidden_pool": int(agent.num_hidden_pool),
            "backbone_depth": 2,
        })
    return common


if __name__ == "__main__":
    main()
