"""Stage A (model-dependent) of fast LongBench eval: pick one action per sample.

Runs a trained RL agent over precomputed LB states and writes the chosen action
index for every sample, per dataset. This is the ONLY half that depends on the
agent / encoder architecture: when you change the model structure, copy or edit
just this script (or point --states_path at the matching precomputed states).
Its output is a plain actions.json that fast_lb_score.py (model-independent)
turns into result.json.

The actions for a dataset are aligned with the sample order of that dataset's
states (which matches index.pt), so the scorer can look them up positionally.

Usage:
    python script/fast_lb_select.py \\
        --rl_checkpoint runs/<run>/policy_best.pt \\
        --states_path   runs/fast_lb_eval/lb_states_v5maxo.pt \\
        --out           runs/fast_lb_eval/actions/<run>.json

Meta-only agent (state_dim=19):
    python script/fast_lb_select.py --rl_checkpoint runs/meta_only_v0/policy_best.pt \\
        --states_path runs/fast_lb_eval/lb_states_pre_rope_mean.pt \\
        --meta_only --out runs/fast_lb_eval/actions/meta_only_v0.json
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL.a2sf_model import SIGMOID_A_VALUES, SIGMOID_B_VALUES
from RL.agent.neural_ucb_agent import NeuralUCBAgent
from RL.env.encoder import METRIC_TYPE_ORDER, TASK_TYPE_ORDER

DEFAULT_STATES = "runs/fast_lb_eval/lb_states_pre_rope_mean.pt"
META_DIM = 1 + len(METRIC_TYPE_ORDER) + len(TASK_TYPE_ORDER)  # 18


def load_agent(ckpt_path: str, meta_only: bool) -> NeuralUCBAgent:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["agent_state_dict"]
    if meta_only:
        agent = NeuralUCBAgent(
            state_dim=META_DIM + 1,
            a_values=torch.tensor(SIGMOID_A_VALUES, dtype=torch.float32),
            b_values=torch.tensor(SIGMOID_B_VALUES, dtype=torch.float32),
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


def select_actions(ckpt_path: str, states_path: str, meta_only: bool = False) -> dict:
    """Return {dataset -> list[int] of chosen action indices} for the agent."""
    agent = load_agent(ckpt_path, meta_only)
    states_d = torch.load(states_path, map_location="cpu", weights_only=False)
    datasets = sorted(k[:-len("/states")] for k in states_d if k.endswith("/states"))

    actions_by_ds = {}
    with torch.no_grad():
        for ds in datasets:
            lb_states = states_d[f"{ds}/states"].float()        # (N, state_dim)
            if meta_only:
                meta = lb_states[:, :META_DIM]
                s = torch.cat([meta, torch.zeros(meta.size(0), 1)], dim=-1)  # (N, 19)
            else:
                s = lb_states
            actions = agent.forward(s)["reward_pred"].argmax(dim=-1)   # (N,)
            actions_by_ds[ds] = [int(a) for a in actions.tolist()]
            m = int(actions.mode().values)
            n_mode = int((actions == m).sum())
            print(f"  {ds:25s}  n={len(actions):4d}  mode=action{m:02d} "
                  f"(a={SIGMOID_A_VALUES[m]:.2g}, b={int(SIGMOID_B_VALUES[m])})  "
                  f"{n_mode}/{len(actions)}", flush=True)
    return actions_by_ds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rl_checkpoint", required=True)
    p.add_argument("--states_path", default=DEFAULT_STATES,
                   help="precomputed LB states matching the agent's encoder")
    p.add_argument("--meta_only", action="store_true",
                   help="meta-only mode (state_dim=19, strip mini_attn+pre_rope)")
    p.add_argument("--out", required=True, help="output actions.json path")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"agent : {args.rl_checkpoint}  meta_only={args.meta_only}")
    print(f"states: {args.states_path}")
    actions_by_ds = select_actions(args.rl_checkpoint, args.states_path, args.meta_only)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "checkpoint":  args.rl_checkpoint,
            "states_path": args.states_path,
            "meta_only":   args.meta_only,
            "actions":     actions_by_ds,
        }, f)
    print(f"saved → {args.out}  ({len(actions_by_ds)} datasets)")


if __name__ == "__main__":
    main()
