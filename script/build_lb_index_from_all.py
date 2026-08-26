"""Build index.pt from pre-existing ALL/ backup directory.

Reads result_txt/backup/llama3-1b/128/ALL/llama3-1b_actionXX_128/ (13 dirs),
aligns samples by the `idx` field, and produces:
  {ds}/preds:       list[N][13]  — actual generated text
  {ds}/scores:      Tensor[N,13] — float32 metric scores (0-100)
  {ds}/answers:     list[N]
  {ds}/all_classes: list[N]
  {ds}/lengths:     list[N]
  datasets:         list[str]

Usage:
    python script/build_lb_index_from_all.py \\
        --all_dir result_txt/backup/llama3-1b/128/ALL \\
        --out runs/fast_lb_eval/index.pt
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL.action_grid import SIGMOID_A_VALUES, SIGMOID_B_VALUES
from longbench_eval import scorer as lb_scorer

DATASETS = [
    "lcc", "repobench-p",
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news",
    "samsum", "trec", "triviaqa",
    "passage_count", "passage_retrieval_en",
]

N_ACTIONS = len(SIGMOID_A_VALUES)  # 13


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--all_dir", default="result_txt/backup/llama3-1b/128/ALL")
    p.add_argument("--out",     default="runs/fast_lb_eval/index.pt")
    return p.parse_args()


def main():
    args = parse_args()

    # Enumerate action dirs sorted by action index
    action_dirs = sorted(
        d for d in os.listdir(args.all_dir)
        if os.path.isdir(os.path.join(args.all_dir, d))
    )
    assert len(action_dirs) == N_ACTIONS, \
        f"Expected {N_ACTIONS} action dirs, got {len(action_dirs)}: {action_dirs}"
    print(f"Action dirs ({len(action_dirs)}): {action_dirs}")

    out = {}
    ds_list = []

    for ds in DATASETS:
        # Load per-action pred files
        # Each file: lines with {idx, pred, answers, all_classes, length}
        action_data = {}   # action_idx -> {sample_idx -> record}
        N = 0

        for a_idx, adir in enumerate(action_dirs):
            path = os.path.join(args.all_dir, adir, f"{ds}.jsonl")
            if not os.path.exists(path):
                print(f"  [skip] {ds} action {a_idx}: file not found")
                continue
            by_idx = {}
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    sample_idx = int(r["idx"])
                    by_idx[sample_idx] = r
                    N = max(N, sample_idx + 1)
            action_data[a_idx] = by_idx

        if not action_data:
            print(f"  [skip] {ds}: no data")
            continue

        print(f"  {ds}: N={N} samples, {len(action_data)} actions loaded")

        # Build arrays aligned by idx
        preds_mat  = [[None] * N_ACTIONS for _ in range(N)]
        scores_mat = torch.zeros(N, N_ACTIONS, dtype=torch.float32)
        answers_l    = [None] * N
        all_classes_l= [None] * N
        lengths_l    = [None] * N

        for a_idx, by_idx in action_data.items():
            for sample_idx, r in by_idx.items():
                pred      = str(r.get("pred", ""))
                answers   = r.get("answers",    []) or []
                all_cls   = r.get("all_classes", None)
                length    = r.get("length")

                preds_mat[sample_idx][a_idx] = pred

                # Store meta from any action (same for all)
                if answers_l[sample_idx] is None:
                    answers_l[sample_idx]     = answers
                    all_classes_l[sample_idx] = all_cls
                    lengths_l[sample_idx]     = length

                # Compute score
                try:
                    s = lb_scorer(ds, [pred], [answers], all_cls or [])
                except Exception:
                    s = 0.0
                scores_mat[sample_idx, a_idx] = float(s)

        out[f"{ds}/preds"]       = preds_mat
        out[f"{ds}/scores"]      = scores_mat
        out[f"{ds}/answers"]     = answers_l
        out[f"{ds}/all_classes"] = all_classes_l
        out[f"{ds}/lengths"]     = lengths_l
        ds_list.append(ds)

        # Quick sanity: mean score per action
        mean_scores = scores_mat.mean(0).tolist()
        best_a = int(scores_mat.mean(0).argmax())
        print(f"    action scores range: {min(mean_scores):.2f}–{max(mean_scores):.2f}"
              f"  best=action{best_a:02d}"
              f"  (a={SIGMOID_A_VALUES[best_a]:.3g}, b={int(SIGMOID_B_VALUES[best_a])})")

    out["datasets"] = ds_list
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out, args.out)
    print(f"\nSaved → {args.out}  ({len(ds_list)} datasets)")


if __name__ == "__main__":
    main()
