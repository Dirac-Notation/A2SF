"""Fast LongBench eval (one-shot wrapper).

Convenience wrapper that chains the two stages:
  1. fast_lb_select.py  — run the agent over LB states -> action index per sample
                          (model-dependent: change this when the architecture changes)
  2. fast_lb_score.py   — look up the chosen actions in index.pt -> result.json
                          (model-independent)

For an iterate-on-architecture workflow, call the two stages separately so the
scoring half stays fixed:
    python script/fast_lb_select.py --rl_checkpoint ... --states_path ... --out a.json
    python script/fast_lb_score.py  --actions a.json --run_name ... --budget 128

Usage (one-shot):
    python script/fast_lb_eval.py \\
        --rl_checkpoint runs/<run>/policy_best.pt \\
        --run_name <run> --budget 128 \\
        --states_path runs/fast_lb_eval/lb_states_v5maxo.pt
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # script/ dir
from fast_lb_select import select_actions, DEFAULT_STATES
from fast_lb_score import score_actions, DEFAULT_INDEX


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rl_checkpoint", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--budget", type=int, default=128)
    p.add_argument("--meta_only", action="store_true")
    p.add_argument("--states_path", default=DEFAULT_STATES)
    p.add_argument("--index_path", default=DEFAULT_INDEX)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[1/2] select actions  (agent={args.rl_checkpoint}, states={args.states_path})")
    actions = select_actions(args.rl_checkpoint, args.states_path, args.meta_only)
    print(f"[2/2] score actions   (index={args.index_path})")
    score_actions(actions, args.run_name, args.budget, args.index_path)


if __name__ == "__main__":
    main()
