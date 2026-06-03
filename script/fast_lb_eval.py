"""Fast LongBench eval using precomputed (sample, action) score index.

Skips model inference entirely. The agent selects an action from precomputed
LB states; scores are looked up from index.pt (precomputed per-sample per-action).
Output format is identical to longbench_RL.py + longbench_eval.py:
  result_txt/pred/<budget>/<run_name>/<dataset>.jsonl  — per-sample records
  result_txt/pred/<budget>/<run_name>/result.json       — aggregate scores

Usage:
    python script/fast_lb_eval.py \\
        --rl_checkpoint runs/exp_pre_rope_mean/policy_best.pt \\
        --run_name exp_pre_rope_mean \\
        --budget 128

Meta-only agent (state_dim=19):
    python script/fast_lb_eval.py \\
        --rl_checkpoint runs/meta_only_v0/policy_best.pt \\
        --run_name meta_only_v0 \\
        --meta_only \\
        --budget 128
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL.a2sf_model import SIGMOID_A_VALUES, SIGMOID_B_VALUES
from RL.agent.neural_ucb_agent import NeuralUCBAgent
from RL.env.encoder import METRIC_TYPE_ORDER, TASK_TYPE_ORDER

STATES_PATH = "runs/fast_lb_eval/lb_states_pre_rope_mean.pt"
INDEX_PATH  = "runs/fast_lb_eval/index.pt"
META_DIM    = 1 + len(METRIC_TYPE_ORDER) + len(TASK_TYPE_ORDER)  # 18


def load_agent(ckpt_path: str, meta_only: bool) -> NeuralUCBAgent:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd   = ckpt["agent_state_dict"]

    if meta_only:
        a_vals = torch.tensor(SIGMOID_A_VALUES, dtype=torch.float32)
        b_vals = torch.tensor(SIGMOID_B_VALUES, dtype=torch.float32)
        agent = NeuralUCBAgent(
            state_dim=META_DIM + 1,
            a_values=a_vals, b_values=b_vals,
            num_metric_types=len(METRIC_TYPE_ORDER),
            num_task_types=len(TASK_TYPE_ORDER),
            side_dim=1, num_heads=1, backbone_depth=2, dropout=0.0,
            paired_actions=True, num_hidden_pool=0,
            task_cond_head=True, num_views=1,
        )
    else:
        cfg = ckpt["arch_config"]
        agent = NeuralUCBAgent(
            state_dim=cfg["state_dim"],
            a_values=cfg["a_values"], b_values=cfg["b_values"],
            num_metric_types=cfg["num_metric_types"],
            num_task_types=cfg["num_task_types"],
            side_dim=cfg["side_dim"], num_heads=cfg["num_heads"],
            backbone_depth=cfg["backbone_depth"], dropout=cfg["dropout"],
            paired_actions=cfg.get("paired_actions", True),
            num_hidden_pool=cfg.get("num_hidden_pool", 0),
            task_cond_head=True,
            num_views=cfg.get("num_views", 2),
        )

    agent.load_state_dict(sd, strict=True)
    agent.eval()
    return agent



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rl_checkpoint", required=True)
    p.add_argument("--run_name",      required=True)
    p.add_argument("--budget",        type=int, default=128)
    p.add_argument("--meta_only",     action="store_true",
                   help="Use meta-only mode (state_dim=19, strip mini_attn+pre_rope).")
    p.add_argument("--states_path",   default=STATES_PATH)
    p.add_argument("--index_path",    default=INDEX_PATH)
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = f"result_txt/pred/{args.budget}/{args.run_name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading agent: {args.rl_checkpoint}  meta_only={args.meta_only}")
    agent = load_agent(args.rl_checkpoint, args.meta_only)

    print(f"Loading states: {args.states_path}")
    states_d = torch.load(args.states_path, map_location="cpu", weights_only=False)
    print(f"Loading index:  {args.index_path}")
    index_d  = torch.load(args.index_path,  map_location="cpu", weights_only=False)
    datasets = index_d["datasets"]

    individual_scores = {}

    with torch.no_grad():
        for ds in datasets:
            states_key = f"{ds}/states"
            if states_key not in states_d:
                print(f"  [skip] {ds}: no states")
                continue

            lb_states  = states_d[states_key].float()   # (N, 88)
            scores_mat = index_d[f"{ds}/scores"].float() # (N, 13)  — 0-100 scale
            preds_mat  = index_d[f"{ds}/preds"]          # list[N][13] — actual text
            answers    = index_d[f"{ds}/answers"]        # list[list[str]]
            all_cls    = index_d.get(f"{ds}/all_classes", [[] for _ in range(lb_states.size(0))])
            lengths    = index_d.get(f"{ds}/lengths",    [None] * lb_states.size(0))

            # Build states for agent
            if args.meta_only:
                meta  = lb_states[:, :META_DIM]
                dummy = torch.zeros(meta.size(0), 1)
                s     = torch.cat([meta, dummy], dim=-1)  # (N, 19)
            else:
                s = lb_states  # (N, 88)

            pred_rewards = agent.forward(s)["reward_pred"]  # (N, 13)
            actions      = pred_rewards.argmax(dim=-1)       # (N,)

            N = lb_states.size(0)
            selected_scores = scores_mat[torch.arange(N), actions]  # (N,) in 0-100

            # Write per-dataset JSONL
            out_path = os.path.join(output_dir, f"{ds}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for i in range(N):
                    act_idx = int(actions[i].item())
                    a_val   = float(SIGMOID_A_VALUES[act_idx])
                    b_val   = float(SIGMOID_B_VALUES[act_idx])
                    score_i = float(selected_scores[i].item())
                    ans_i   = answers[i] if i < len(answers) else []
                    cls_i   = all_cls[i] if i < len(all_cls) else []
                    len_i   = int(lengths[i]) if (i < len(lengths) and lengths[i] is not None) else None
                    pred_text = str(preds_mat[i][act_idx]) if preds_mat[i][act_idx] is not None else ""
                    record  = {
                        "pred":        pred_text,
                        "answers":     ans_i,
                        "all_classes": cls_i,
                        "length":      len_i,
                        "a":           a_val,
                        "b":           int(round(b_val)),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            ds_avg = float(selected_scores.mean().item())
            individual_scores[ds] = round(ds_avg, 2)
            print(f"  {ds:25s}  {ds_avg:.2f}  (action dist: "
                  f"top={SIGMOID_A_VALUES[int(actions.mode().values)]:.2g}/"
                  f"b{int(SIGMOID_B_VALUES[int(actions.mode().values)])}  "
                  f"n={int((actions == actions.mode().values).sum())}/{N})",
                  flush=True)

    # Score with longbench_eval (uses actual pred text)
    from longbench_eval import evaluate_results
    evaluate_results(output_dir)


if __name__ == "__main__":
    main()
