#!/usr/bin/env python3
"""Assemble RL training data with FC reward from a scored pool.

Merges the two outputs of generate_sigmoid_dataset.py:
  <scored_dir>/common.jsonl      — metadata + full_cache_pred/full_cache_score
  <scored_dir>/budget_<B>.jsonl  — per-sample action_outputs/action_scores_gt/action_scores_fc

into the record format RL/train.py reads (it indexes r[score_field][str(budget)]).
The bandit reward is FC (action_scores_fc_by_budget); GT is kept alongside for
eval/comparison. MaxO is intentionally dropped (see reward-design discussion).

Then a deterministic, per-dataset-stratified split → train.jsonl + validation.jsonl
in the same dir.

  python datasets/assemble_fc_training.py \
      --scored_dir datasets/training/scored/faithful_v1 --budget 128 --val_frac 0.1

Train with:
  python RL/train.py --model llama3-1b --budget 128 \
      --data_file      datasets/training/scored/faithful_v1/train.jsonl \
      --val_data_file  datasets/training/scored/faithful_v1/validation.jsonl \
      --score_field action_scores_fc_by_budget \
      --val_score_field action_scores_fc_by_budget ...
"""
import argparse, json, os, random
from collections import defaultdict


def _as_list(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return [x]
    return x if isinstance(x, list) else ([] if x is None else [x])


def load_common(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "full_cache_pred" not in r:
                continue
            rows[int(r["sample_id"])] = r
    return rows


def load_budget(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[int(r["sample_id"])] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_dir", required=True,
                    help="dir with common.jsonl + budget_<B>.jsonl")
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_positive", action="store_true",
                    help="keep only rows whose FC scores have at least one >0 "
                         "(train.py filters these out anyway).")
    ap.add_argument("--exclude_datasets", default="",
                    help="comma-separated datasets to drop (e.g. unlearnable synthetic "
                         "tasks where 1B yields ~0/empty regardless of action).")
    a = ap.parse_args()
    exclude = {s.strip() for s in a.exclude_datasets.split(",") if s.strip()}
    bkey = str(a.budget)

    common = load_common(os.path.join(a.scored_dir, "common.jsonl"))
    budget = load_budget(os.path.join(a.scored_dir, f"budget_{a.budget}.jsonl"))
    sids = sorted(set(common) & set(budget))
    print(f"common={len(common)}  budget_{a.budget}={len(budget)}  merged={len(sids)}")

    records, skipped = [], 0
    for sid in sids:
        c = common[sid]; b = budget[sid]
        if c.get("dataset") in exclude:
            skipped += 1; continue
        sc_gt = [float(x) for x in b.get("action_scores_gt", [])]
        sc_fc = [float(x) for x in b.get("action_scores_fc", [])]
        if not sc_fc:
            skipped += 1; continue
        if a.min_positive and not any(x > 0 for x in sc_fc):
            skipped += 1; continue
        records.append({
            "sample_id": str(sid),
            "input_prompt": c["input_prompt"],
            "answers": _as_list(c.get("answers")),
            "all_classes": _as_list(c.get("all_classes")),
            "metric_type": c.get("metric_type", "qa_f1_score"),
            "task_type": c.get("task_type", "unknown"),
            "dataset": c.get("dataset"),
            "subtype": c.get("subtype", "_"),
            "length": int(c.get("length", 0)),
            "generation_length": int(c.get("generation_length", 64)),
            "full_cache_pred": c.get("full_cache_pred", ""),
            "full_cache_score": float(c.get("full_cache_score", 0.0)),
            "action_outputs": b.get("action_outputs", []),
            "action_scores_gt": sc_gt,
            "action_scores_fc": sc_fc,
            # per-budget dicts that RL/train.py indexes via score_field[str(budget)]
            "action_scores_fc_by_budget": {bkey: sc_fc},
            "action_scores_gt_by_budget": {bkey: sc_gt},
            "token_budget": int(a.budget),
        })
    print(f"assembled {len(records)} records ({skipped} skipped)")

    # deterministic per-dataset stratified split
    by_ds = defaultdict(list)
    for r in records:
        by_ds[r["dataset"]].append(r)
    rng = random.Random(a.seed)
    train, val = [], []
    for ds, rs in sorted(by_ds.items()):
        rs = rs[:]; rng.shuffle(rs)
        n_val = max(1, int(round(len(rs) * a.val_frac))) if len(rs) > 1 else 0
        val.extend(rs[:n_val]); train.extend(rs[n_val:])
    rng.shuffle(train); rng.shuffle(val)

    tr_path = os.path.join(a.scored_dir, "train.jsonl")
    va_path = os.path.join(a.scored_dir, "validation.jsonl")
    for path, rs in [(tr_path, train), (va_path, val)]:
        with open(path, "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train={len(train)} -> {tr_path}")
    print(f"val  ={len(val)} -> {va_path}")

    # per-dataset/task report
    def dist(rs, key):
        c = defaultdict(int)
        for r in rs: c[r[key]] += 1
        return dict(sorted(c.items()))
    print("train per-task:", dist(train, "task_type"))
    print("val   per-task:", dist(val, "task_type"))


if __name__ == "__main__":
    main()
