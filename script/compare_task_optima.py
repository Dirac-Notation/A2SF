"""Compare per-task optimal (a,b) action between a TRAINING corpus (scored) and LongBench.
Stage-1 validation: does the training data's per-task argmax action match LongBench's?

  python script/compare_task_optima.py --scored_dirs datasets/training/raw/synth_v1_1b,datasets/training/raw/clean_v1_1b \
      --index runs/fast_lb_eval/index_llama3-1b_128.pt
"""
import argparse, json, os, sys
import numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACT = ["(0,1)H2O", "0.01/1", "0.01/16", "0.01/32", "0.01/128", "0.1/1", "0.1/16",
       "0.1/32", "0.1/128", "10/1", "10/16", "10/32", "10/128"]

def load_train(scored_dirs):
    by_task = {}  # task -> list of 13-vectors
    fc_by_task = {}
    for d in scored_dirs:
        common = {json.loads(l)["sample_id"]: json.loads(l)
                  for l in open(os.path.join(d, "common.jsonl"))}
        for l in open(os.path.join(d, "budget_128.jsonl")):
            r = json.loads(l); c = common.get(r["sample_id"])
            if not c: continue
            t = c["task_type"]; sc = r["action_scores_gt"]
            if len(sc) != 13: continue
            by_task.setdefault(t, []).append(sc)
            fc_by_task.setdefault(t, []).append(c.get("full_cache_score", 0.0))
    return by_task, fc_by_task

def load_lb(index_path):
    idx = torch.load(index_path, map_location="cpu", weights_only=False)
    d2t = json.load(open(f"{REPO}/config/task2dataset.json"))
    ds2task = {}
    for task, dss in d2t.items():
        for ds in dss: ds2task[ds] = task
    by_task = {}; fc_by_task = {}
    for k in list(idx.keys()):
        if not k.endswith("/scores"): continue
        ds = k[:-len("/scores")]; task = ds2task.get(ds)
        if task is None: continue
        sc = np.asarray(idx[k])  # (N, 13)
        if sc.ndim != 2 or sc.shape[1] != 13: continue
        by_task.setdefault(task, []).append(sc)
        # full-cache proxy: best single action mean is not fc; index has no fc -> skip
    return {t: np.concatenate(v, 0) for t, v in by_task.items()}

def load_lb_perds(index_path):
    """per-DATASET (N,13) score arrays + dataset->task map, for transfer scoring."""
    idx = torch.load(index_path, map_location="cpu", weights_only=False)
    d2t = json.load(open(f"{REPO}/config/task2dataset.json"))
    ds2task = {ds: t for t, dss in d2t.items() for ds in dss}
    out = {}
    for k in idx:
        if not k.endswith("/scores"): continue
        ds = k[:-len("/scores")]; sc = np.asarray(idx[k])
        if sc.ndim == 2 and sc.shape[1] == 13:
            out[ds] = (sc, ds2task.get(ds, "?"))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_dirs", required=True)
    ap.add_argument("--index", required=True)
    a = ap.parse_args()
    train, fc = load_train(a.scored_dirs.split(","))
    lb = load_lb(a.index)
    perds = load_lb_perds(a.index)
    # per-task TRAINING argmax action
    tr_arg = {t: int(np.asarray(v).mean(0).argmax()) for t, v in train.items() if np.asarray(v).size}
    lb_task_arg = {t: int(v.mean(0).argmax()) for t, v in lb.items()}
    # transfer: each LB dataset uses its TASK's training-argmax; mean over all datasets (= LB Avg)
    tr_scores, tf_scores, orc_scores = [], [], []
    for ds, (sc, task) in perds.items():
        m = sc.mean(0)
        if task in tr_arg: tr_scores.append(m[tr_arg[task]] * 100)
        if task in lb_task_arg: tf_scores.append(m[lb_task_arg[task]] * 100)
        orc_scores.append(m.max() * 100)
    print("\n=== TRANSFER (LB Avg over datasets) ===")
    print(f"  training-argmax per task -> LB: {np.mean(tr_scores):.2f}  (covered {len(tr_scores)}/{len(perds)} ds)")
    print(f"  LB task-fixed (ceiling for task head): {np.mean(tf_scores):.2f}")
    print(f"  per-dataset oracle: {np.mean(orc_scores):.2f}")
    print(f"{'TASK':<20} {'train_best(a/b)':<16} {'LB_best(a/b)':<16} {'match':<6} {'train_fc':<9} {'n_tr':<6} {'n_lb'}")
    print("-" * 90)
    match = 0; total = 0
    for task in sorted(set(train) | set(lb)):
        tr = np.asarray(train.get(task, []))
        lbv = lb.get(task)
        tr_best = ACT[int(tr.mean(0).argmax())] if tr.size else "-"
        lb_best = ACT[int(lbv.mean(0).argmax())] if lbv is not None else "-"
        tr_fc = float(np.mean(fc.get(task, [0]))) if task in fc else 0.0
        m = "YES" if (tr.size and lbv is not None and tr_best == lb_best) else ""
        if tr.size and lbv is not None:
            total += 1; match += (tr_best == lb_best)
        print(f"{task:<20} {tr_best:<16} {lb_best:<16} {m:<6} {tr_fc:<9.3f} "
              f"{len(tr) if tr.size else 0:<6} {len(lbv) if lbv is not None else 0}")
    print("-" * 90)
    print(f"per-task argmax match: {match}/{total}")
    # also show train per-action means for degenerate diagnosis
    print("\n=== train per-action GT mean by task (×100) ===")
    for task in sorted(train):
        tr = np.asarray(train[task])
        print(f"{task:<20} " + " ".join(f"{x*100:4.1f}" for x in tr.mean(0)))
    print("idx:                 " + " ".join(f"{i:>4}" for i in range(13)))

if __name__ == "__main__":
    main()
