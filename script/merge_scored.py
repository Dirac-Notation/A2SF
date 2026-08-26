"""Merge multiple scored dirs into one, keeping only chosen task_types from each, with
unique sample_ids. Produces a combined <out>/common.jsonl + budget_<B>.jsonl for assemble.

  python script/merge_scored.py --out datasets/training/raw/recipe_v1_1b --budget 128 \
      --src datasets/training/raw/synth_v1_1b:Passage_Retrieval \
      --src datasets/training/raw/qa_v1_1b:Single-doc_QA,Multi-doc_QA \
      --src datasets/training/raw/clean_v1_1b:Code_Complete,Few_Shot,Summarization
(task names: underscores -> spaces)
"""
import argparse, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--src", action="append", required=True,
                    help="dir:Task1,Task2 (underscores become spaces)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    com_out = open(os.path.join(a.out, "common.jsonl"), "w")
    bud_out = open(os.path.join(a.out, f"budget_{a.budget}.jsonl"), "w")
    nid = 0; counts = {}
    for spec in a.src:
        d, tasks = spec.split(":")
        keep = set(t.replace("_", " ") for t in tasks.split(","))
        com = {json.loads(l)["sample_id"]: json.loads(l)
               for l in open(os.path.join(d, "common.jsonl"))}
        for l in open(os.path.join(d, f"budget_{a.budget}.jsonl")):
            r = json.loads(l); c = com.get(r["sample_id"])
            if not c or c.get("task_type") not in keep: continue
            c = dict(c); r = dict(r)
            c["sample_id"] = nid; r["sample_id"] = nid; nid += 1
            counts[c["task_type"]] = counts.get(c["task_type"], 0) + 1
            com_out.write(json.dumps(c) + "\n"); bud_out.write(json.dumps(r) + "\n")
    com_out.close(); bud_out.close()
    print(f"merged {nid} rows -> {a.out}; by task: {counts}")

if __name__ == "__main__":
    main()
