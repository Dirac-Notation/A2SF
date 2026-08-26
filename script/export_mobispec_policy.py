"""Export a trained WAITS routing selector to a MobiSpec (llama.cpp) policy JSON.

The selector artifact (runs/selectors/<m>_u5b.npz, RoutingNeuralUCB) maps
(task, metric) -> greedy (a, b) action. MobiSpec's llama-simple-snapkv consumes the
exported file via --waits-policy FILE --waits-task NAME --waits-metric NAME:
    { "Single-doc QA|qa_f1_score": [10.0, 16.0], ..., "default": [a, b] }
"default" = the ("unknown", "unknown") greedy action, used when the key is absent.

Usage:
    python script/export_mobispec_policy.py --agent runs/selectors/llama3-8b_u5b.npz \
        --out experiments/MobiSpec/policies/waits_llama3-8b_u5b.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL.model import RoutingNeuralUCB
from RL.metadata import TASK_TYPE_ORDER, METRIC_TYPE_ORDER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="selector artifact (runs/selectors/<m>_u5b.npz)")
    ap.add_argument("--out", required=True, help="output policy JSON path")
    args = ap.parse_args()

    agent = RoutingNeuralUCB.load(args.agent)

    policy = {}
    for task in TASK_TYPE_ORDER:
        for metric in METRIC_TYPE_ORDER:
            a, b = agent.action_ab(agent.greedy_action(task, metric))
            policy[f"{task}|{metric}"] = [float(a), float(b)]
    policy["default"] = policy["unknown|unknown"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(policy, f, indent=2, ensure_ascii=False)

    n_cells = len(TASK_TYPE_ORDER) * len(METRIC_TYPE_ORDER)
    print(f"wrote {args.out}: {n_cells} (task, metric) cells + default = {policy['default']}")


if __name__ == "__main__":
    main()
