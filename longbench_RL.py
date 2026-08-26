"""LongBench evaluation for the WAITS routing policy.

The (task, metric) routing model is a lookup table (RL/model.py), so RL eval is just:
  1. train the routing policy + export a `--waits_table` block (RL/train.py)
  2. run the standard LongBench eval applying that table (longbench.py --waits_table)

This thin orchestrator is the RL-eval entry point. (The legacy per-prompt deep-agent eval
harness was removed; recover from git / logs/history if a per-prompt RL model is revived.)

  python longbench_RL.py --model llama3-1b --budget 128 \
      --recipe datasets/training/raw/recipe_v3_1b/train.jsonl \
      --index runs/fast_lb_eval/index_llama3-1b_128.pt \
      --run_name WAITS_llama3-1b --gpus_per_model 1
"""
import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="LongBench eval of the WAITS routing policy.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--recipe", required=True, help="recipe jsonl with action_scores_gt_by_budget")
    ap.add_argument("--index", required=True, help="per-action LB index for routing train/eval")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--gpus_per_model", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--table_dir", default="runs/waits_tables")
    a = ap.parse_args()

    py = sys.executable
    table = os.path.join(a.table_dir, f"waits_{a.model}.json")
    run_name = a.run_name or f"WAITS_{a.model}"

    # 1) train routing policy + export the (dataset -> (a,b)) waits_table
    train_cmd = [py, "RL/train.py", "--model", a.model, "--recipe", a.recipe,
                 "--index", a.index, "--budget", str(a.budget),
                 "--seed", str(a.seed), "--export_table", table]
    print("[longbench_RL] train+export:", " ".join(train_cmd), flush=True)
    subprocess.run(train_cmd, cwd=REPO, check=True)

    # 2) standard LongBench eval applying the learned table
    eval_cmd = [py, "longbench.py", "--model", a.model, "--budget", str(a.budget),
                "--gpus_per_model", str(a.gpus_per_model), "--run_name", run_name,
                "--waits_table", table]
    print("[longbench_RL] eval:", " ".join(eval_cmd), flush=True)
    subprocess.run(eval_cmd, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
