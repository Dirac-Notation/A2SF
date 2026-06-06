"""Stage B (model-independent) of fast LongBench eval: score the chosen actions.

Reads an actions.json (from fast_lb_select.py) and the precomputed per-sample
per-action score index, looks up each sample's score, and writes predictions +
result.json in the standard longbench format. This half has NO RL / encoder /
model code, so it never changes when you swap the agent architecture.

Output (identical to longbench_RL.py + longbench_eval.py):
  result_txt/pred/<budget>/<run_name>/<dataset>.jsonl  — per-sample records
  result_txt/pred/<budget>/<run_name>/result.json       — aggregate scores

Usage:
    python script/fast_lb_score.py \\
        --actions   runs/fast_lb_eval/actions/<run>.json \\
        --run_name  <run> --budget 128
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Action grid constants only (the (a, b) definitions written into each record);
# this is the action space, not model architecture.
from RL.a2sf_model import SIGMOID_A_VALUES, SIGMOID_B_VALUES

DEFAULT_INDEX = "runs/fast_lb_eval/index.pt"


def score_actions(actions_by_ds: dict, run_name: str, budget: int,
                  index_path: str = DEFAULT_INDEX) -> dict:
    """Look up scores for chosen actions, write predictions, run LB scorer.

    Returns the per-dataset mean of the looked-up index scores (pre-scorer
    sanity numbers; the authoritative scores come from result.json).
    """
    index_d = torch.load(index_path, map_location="cpu", weights_only=False)
    output_dir = f"result_txt/pred/{budget}/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    individual_scores = {}
    for ds in index_d["datasets"]:
        if ds not in actions_by_ds:
            print(f"  [skip] {ds}: no actions in file")
            continue

        scores_mat = index_d[f"{ds}/scores"].float()   # (N, 13) — 0-100
        preds_mat  = index_d[f"{ds}/preds"]            # list[N][13]
        answers    = index_d[f"{ds}/answers"]
        all_cls    = index_d.get(f"{ds}/all_classes", None)
        lengths    = index_d.get(f"{ds}/lengths", None)
        N = scores_mat.size(0)

        actions = torch.tensor(actions_by_ds[ds], dtype=torch.long)
        if actions.numel() != N:
            raise ValueError(
                f"{ds}: actions length {actions.numel()} != index samples {N}. "
                "The states used in fast_lb_select.py must match index.pt.")

        selected = scores_mat[torch.arange(N), actions]   # (N,)

        out_path = os.path.join(output_dir, f"{ds}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for i in range(N):
                a_idx = int(actions[i])
                pred_text = preds_mat[i][a_idx]
                rec = {
                    "pred":        str(pred_text) if pred_text is not None else "",
                    "answers":     answers[i] if i < len(answers) else [],
                    "all_classes": all_cls[i] if (all_cls and i < len(all_cls)) else [],
                    "length":      int(lengths[i]) if (lengths and i < len(lengths)
                                                       and lengths[i] is not None) else None,
                    "a":           float(SIGMOID_A_VALUES[a_idx]),
                    "b":           int(round(float(SIGMOID_B_VALUES[a_idx]))),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        ds_avg = float(selected.mean())
        individual_scores[ds] = round(ds_avg, 2)
        print(f"  {ds:25s}  {ds_avg:.2f}", flush=True)

    # Authoritative scoring from the written pred text (writes result.json).
    from longbench_eval import evaluate_results
    evaluate_results(output_dir)
    print(f"saved → {output_dir}/result.json")
    return individual_scores


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--actions", required=True, help="actions.json from fast_lb_select.py")
    p.add_argument("--run_name", required=True)
    p.add_argument("--budget", type=int, default=128)
    p.add_argument("--index_path", default=DEFAULT_INDEX)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.actions) as f:
        act = json.load(f)
    print(f"actions: {args.actions}  ({len(act['actions'])} datasets, "
          f"ckpt={act.get('checkpoint')})")
    score_actions(act["actions"], args.run_name, args.budget, args.index_path)


if __name__ == "__main__":
    main()
