"""Merge per-worker oracle results and score with longbench_eval.py logic."""
import os, json, glob, argparse
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--n_workers", type=int, default=16)
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = args.out_dir
    merged_dir = os.path.join(out_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    # Collect all worker shards per dataset
    dataset_records = defaultdict(dict)
    for fpath in glob.glob(os.path.join(out_dir, "*_w*.jsonl")):
        with open(fpath) as f:
            for line in f:
                r = json.loads(line)
                dataset = r["dataset"]
                idx = r["idx"]
                dataset_records[dataset][idx] = r

    # Write merged files
    for dataset, records in dataset_records.items():
        merged_path = os.path.join(merged_dir, f"{dataset}.jsonl")
        with open(merged_path, "w") as f:
            for idx in sorted(records):
                f.write(json.dumps(records[idx]) + "\n")
        print(f"{dataset}: {len(records)} examples → {merged_path}")

    print("\nMerge done. Run longbench_eval.py on:", merged_dir)

if __name__ == "__main__":
    main()
