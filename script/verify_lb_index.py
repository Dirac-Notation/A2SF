"""Verify index.pt against the ALL backup directory.

Uses the `idx` field present in ALL/llama3-1b_actionXX_128/*.jsonl for direct
sample alignment — no fuzzy (answers, length) key matching needed.

For every (dataset, action, sample) triple:
  1. pred text  — index[ds/preds][idx][action] must equal ALL pred
  2. score      — index[ds/scores][idx, action] must match re-scored pred

Usage:
    python script/verify_lb_index.py \\
        --index runs/fast_lb_eval/index.pt \\
        --all_dir result_txt/backup/llama3-1b/128/ALL
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL.a2sf_model import SIGMOID_A_VALUES, SIGMOID_B_VALUES
from longbench_eval import scorer as lb_scorer

N_ACTIONS = len(SIGMOID_A_VALUES)

DATASETS = [
    "lcc", "repobench-p",
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news",
    "samsum", "trec", "triviaqa",
    "passage_count", "passage_retrieval_en",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index",   default="runs/fast_lb_eval/index.pt")
    p.add_argument("--all_dir", default="result_txt/backup/llama3-1b/128/ALL")
    p.add_argument("--max_per_ds", type=int, default=0,
                   help="Limit samples per dataset (0=all).")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading index: {args.index}")
    index = torch.load(args.index, map_location="cpu", weights_only=False)
    print(f"Datasets in index: {index.get('datasets', [])}")

    action_dirs = sorted(
        d for d in os.listdir(args.all_dir)
        if os.path.isdir(os.path.join(args.all_dir, d))
    )
    assert len(action_dirs) == N_ACTIONS, \
        f"Expected {N_ACTIONS} action dirs, got {len(action_dirs)}"
    print(f"Action dirs: {len(action_dirs)}")

    total   = 0
    text_ok = 0
    text_bad= 0
    score_ok = 0
    score_bad= 0
    mismatch_examples = []

    for ds in DATASETS:
        preds_mat  = index.get(f"{ds}/preds")
        scores_mat = index.get(f"{ds}/scores")
        if preds_mat is None:
            print(f"  [{ds}] not in index — skip")
            continue

        ds_text_ok = ds_text_bad = ds_score_ok = ds_score_bad = 0

        for a_idx, adir in enumerate(action_dirs):
            path = os.path.join(args.all_dir, adir, f"{ds}.jsonl")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            if args.max_per_ds > 0:
                lines = lines[:args.max_per_ds]

            for r in lines:
                sample_idx = int(r["idx"])
                if sample_idx >= len(preds_mat):
                    continue

                pred_all = str(r.get("pred", ""))
                pred_idx = str(preds_mat[sample_idx][a_idx])
                total += 1

                # 1. Text check
                if pred_all == pred_idx:
                    text_ok += 1; ds_text_ok += 1
                else:
                    text_bad += 1; ds_text_bad += 1
                    if len(mismatch_examples) < 5:
                        mismatch_examples.append({
                            "ds": ds, "idx": sample_idx, "action": a_idx,
                            "all_pred":   pred_all[:80],
                            "index_pred": pred_idx[:80],
                        })

                # 2. Score check
                if scores_mat is not None:
                    idx_score = float(scores_mat[sample_idx, a_idx].item())
                    answers   = r.get("answers",    []) or []
                    all_cls   = r.get("all_classes") or []
                    recomputed= lb_scorer(ds, [pred_all], [answers], all_cls)
                    if abs(recomputed - idx_score) < 1.0:
                        score_ok += 1; ds_score_ok += 1
                    else:
                        score_bad += 1; ds_score_bad += 1

        n_checked = ds_text_ok + ds_text_bad
        print(f"  {ds:25s}  text OK={ds_text_ok}/{n_checked}  bad={ds_text_bad}"
              f"  score OK={ds_score_ok}/{n_checked}  bad={ds_score_bad}")

    print(f"\n{'─'*55}")
    print(f"Total checked:    {total}")
    print(f"Text  match:      {text_ok} / {total}"
          f"  ({100*text_ok/max(1,total):.1f}%)")
    print(f"Text  mismatch:   {text_bad}")
    print(f"Score match:      {score_ok} / {total}"
          f"  ({100*score_ok/max(1,total):.1f}%)")
    print(f"Score mismatch:   {score_bad}")

    if mismatch_examples:
        print(f"\nFirst {len(mismatch_examples)} text mismatches:")
        for ex in mismatch_examples:
            print(f"  [{ex['ds']}#{ex['idx']}] action={ex['action']}")
            print(f"    ALL:   {ex['all_pred']!r}")
            print(f"    index: {ex['index_pred']!r}")

    if text_bad == 0 and score_bad == 0:
        print("\n✓ PASS: index.pt matches ALL backup perfectly.")
    else:
        print("\n✗ FAIL: mismatches detected.")


if __name__ == "__main__":
    main()
