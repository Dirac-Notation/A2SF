"""Per-task LOOKUP meta-policy eval: compute each task's argmax action from a model's
scored TRAIN data (NO LongBench), apply per-task to that model's LB index, report LB Avg.
This is the simplest meta-only policy (= the task-fixed the recipe can reach without LB).

  python script/lookup_eval.py --train datasets/training/raw/recipe_v2_1b/train.jsonl \
      --index runs/fast_lb_eval/index_llama3-1b_128.pt
"""
import argparse, json, os, sys
import numpy as np, torch
from collections import defaultdict
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACT = ['H2O', '0.01/1', '0.01/16', '0.01/32', '0.01/128', '0.1/1', '0.1/16',
       '0.1/32', '0.1/128', '10/1', '10/16', '10/32', '10/128']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="scored train.jsonl (action_scores_gt_by_budget)")
    ap.add_argument("--index", required=True)
    ap.add_argument("--budget", default="128")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.train)]
    by = defaultdict(list)
    for r in rows:
        sc = r["action_scores_gt_by_budget"]
        sc = sc[a.budget] if isinstance(sc, dict) else sc
        by[r["task_type"]].append(sc)
    task_arg = {t: int(np.mean(np.asarray(v), 0).argmax()) for t, v in by.items()}
    idx = torch.load(a.index, map_location="cpu", weights_only=False)
    d2t = json.load(open(f"{REPO}/config/task2dataset.json"))
    ds2task = {ds: t for t, dss in d2t.items() for ds in dss}
    per_ds = {}
    for k in idx:
        if not k.endswith("/scores"): continue
        ds = k[:-len("/scores")]; sc = np.asarray(idx[k]); t = ds2task.get(ds)
        if sc.ndim != 2 or sc.shape[1] != 13 or t not in task_arg: continue
        per_ds[ds] = sc[:, task_arg[t]].mean()
    print("per-task argmax (TRAIN, no LB): " +
          ", ".join(f"{t}={ACT[task_arg[t]]}" for t in sorted(task_arg)))
    print("per-dataset LB: " + ", ".join(f"{ds}={v:.1f}" for ds, v in sorted(per_ds.items())))
    print(f"\n★ per-task LOOKUP meta-policy LB Avg = {np.mean(list(per_ds.values())):.2f}  "
          f"(over {len(per_ds)} datasets)")

if __name__ == "__main__":
    main()
