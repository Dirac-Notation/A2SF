"""Train the WAITS routing policy (L) — the deployed (task, metric) NeuralUCB.

D-I-A-R-L:
  D  recipe jsonl (non-LongBench): task_type, metric_type, action_scores_gt_by_budget[budget]
  I  phi(s) = [task_oh | metric_oh]                       (RL/metadata.py)
  A  RoutingNeuralUCB = per-arm LinUCB over phi           (RL/model.py)
  R  TRUE bandit feedback: play one arm, observe only r[a*]  (GT reward)
  L  closed-form Sherman-Morrison rank-1 update of A_{a*}^-1

Greedy eval (no bonus) on the real-LB per-action index; reports the policy's mean accuracy
vs the (task, metric) lookup table. `--export_table` writes a `longbench.py --waits_table`
block so the learned policy is deployed through the standard eval path.

  python RL/train.py --model llama3-1b \
      --recipe datasets/training/raw/recipe_v3_1b/train.jsonl \
      --index runs/fast_lb_eval/index_llama3-1b_128.pt --budget 128 \
      --export_table runs/waits_tables/waits_llama3-1b.json
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from RL.model import RoutingNeuralUCB
from RL.metadata import dataset_metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--budget", default="128")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=0.0, help="epsilon-greedy explore prob")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export_table", type=str, default=None,
                    help="If set, write the learned (dataset->(a,b)) waits_table JSON here.")
    ap.add_argument("--actions", type=str, default=None,
                    help="Arm subset 'a:b,a:b,...' (e.g. the U5b champion set). Rewards for "
                         "a=1.0 curves come from --ext_scores.")
    ap.add_argument("--ext_scores", type=str, default=None,
                    help="extA budget jsonl with action_scores_gt for the a=1.0 curves.")
    ap.add_argument("--save_agent", type=str, default=None,
                    help="Save the TRAINED SELECTOR (model artifact .npz) here.")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.recipe)]

    ACT17 = [(0.0,1),(0.01,1),(0.01,16),(0.01,32),(0.01,128),(0.1,1),(0.1,16),(0.1,32),(0.1,128),
             (10.0,1),(10.0,16),(10.0,32),(10.0,128),(1.0,1),(1.0,16),(1.0,32),(1.0,128)]
    ext = None
    if a.ext_scores:
        ext = {int(json.loads(l)["sample_id"]): json.loads(l)["action_scores_gt"]
               for l in open(a.ext_scores)}
    arm_actions = None
    cols = None
    if a.actions:
        arm_actions = [(float(p.split(":")[0]), float(p.split(":")[1])) for p in a.actions.split(",")]
        cols = [ACT17.index(ab) for ab in arm_actions]

    def reward(r):
        sc = r["action_scores_gt_by_budget"]
        base = list(map(float, sc[a.budget] if isinstance(sc, dict) else sc))
        if ext is not None:
            e = ext.get(int(r["sample_id"]))
            base = base + (list(map(float, e)) if e is not None else [np.nan] * 4)
        v = np.asarray(base, dtype=float)
        return v[cols] if cols is not None else v

    data = [(r["task_type"], r.get("metric_type", "?"), reward(r)) for r in rows]
    data = [d for d in data if not np.isnan(d[2]).any()]

    # ── train: TRUE bandit feedback + Sherman-Morrison ────────────────────────
    agent = RoutingNeuralUCB(lam=a.lam, beta=a.beta, seed=a.seed, actions=arm_actions)
    rng = np.random.RandomState(a.seed)
    for _ in range(a.epochs):
        for idx in rng.permutation(len(data)):
            task, met, r = data[idx]
            p = agent.phi(task, met)
            astar = agent.select(p, epsilon=a.epsilon)
            agent.update(p, astar, float(r[astar]))

    if a.save_agent:
        os.makedirs(os.path.dirname(a.save_agent) or ".", exist_ok=True)
        agent.save(a.save_agent)
        print(f"selector (trained agent) saved -> {a.save_agent}")

    # ── eval: greedy policy vs lookup table on the real-LB per-action index ────
    import torch
    d2t = json.load(open(f"{REPO}/config/task2dataset.json"))
    ds2task = {d: t for t, ds in d2t.items() for d in ds}
    from collections import defaultdict
    by_s, by_t = defaultdict(list), defaultdict(list)
    for r in rows:
        by_s[(r["task_type"], r.get("metric_type", "?"))].append(reward(r))
        by_t[r["task_type"]].append(reward(r))
    s_arg = {k: int(np.mean(v, 0).argmax()) for k, v in by_s.items()}
    t_arg = {k: int(np.mean(v, 0).argmax()) for k, v in by_t.items()}

    idx = torch.load(a.index, map_location="cpu", weights_only=False)
    v_pol, v_tab, match, n = [], [], 0, 0
    for k in idx:
        if not k.endswith("/scores"):
            continue
        ds = k[:-7]; sc = np.asarray(idx[k]); t = ds2task.get(ds)
        if sc.ndim != 2 or sc.shape[1] != agent.A or t is None:
            continue
        met = dataset_metric(ds, t)
        a_pol = agent.greedy_action(t, met)
        a_tab = s_arg.get((t, met), t_arg.get(t, 0))
        v_pol.append(sc[:, a_pol].mean()); v_tab.append(sc[:, a_tab].mean())
        match += (a_pol == a_tab); n += 1
    print(f"{a.model:11} routing(beta={a.beta},ep={a.epochs},seed={a.seed}) = "
          f"{np.mean(v_pol):.2f}  table={np.mean(v_tab):.2f}  "
          f"(action match {match}/{n}, arms played {int(agent.plays.sum())})")

    # ── deploy: export the learned policy as a longbench --waits_table block ───
    if a.export_table:
        datasets, n_samples = [], {}
        for k in idx:
            if not k.endswith("/scores"):
                continue
            ds = k[:-7]; sc = np.asarray(idx[k])
            if sc.ndim != 2 or ds2task.get(ds) is None:
                continue
            datasets.append(ds); n_samples[ds] = sc.shape[0]
        table = agent.export_table(datasets, ds2task, n_samples)
        os.makedirs(os.path.dirname(a.export_table) or ".", exist_ok=True)
        key = f"{a.model}_{int(a.budget)}"
        existing = {}
        if os.path.exists(a.export_table):
            existing = json.load(open(a.export_table))
        existing[key] = table
        json.dump(existing, open(a.export_table, "w"))
        print(f"exported waits_table key={key} ({len(table)} datasets) -> {a.export_table}")


if __name__ == "__main__":
    main()
