"""Assemble baseline-format prediction .jsonl files for WAITS(task) / WAITS(sub_type)
routing, by selecting each dataset's routed-action prediction from the fast-LB index
(which stores per-prompt x per-action preds). Then score with longbench_eval so each
backup dir matches a normal baseline dir (per-dataset .jsonl + result.json).

  python script/make_waits_preds.py --model llama3-1b \
      --recipe datasets/training/raw/recipe_v3_1b/train.jsonl \
      --index runs/fast_lb_eval/index_llama3-1b_128.pt --budget 128
"""
import argparse, json, os, subprocess, sys
import numpy as np, torch
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LB_SUBTYPE = {
    "trec": "classification", "triviaqa": "qa", "samsum": "dialogue",
    "narrativeqa": "qa", "qasper": "qa", "multifieldqa_en": "qa",
    "hotpotqa": "qa", "2wikimqa": "qa", "musique": "qa",
    "gov_report": "summary", "multi_news": "summary", "qmsum": "summary",
    "lcc": "code", "repobench-p": "code",
    "passage_retrieval_en": "retrieval", "passage_count": "retrieval",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--budget", default="128")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.recipe)]
    by_ts, by_t = defaultdict(list), defaultdict(list)
    for r in rows:
        sc = r["action_scores_gt_by_budget"]; sc = sc[a.budget] if isinstance(sc, dict) else sc
        st = r.get("subtype", "_")
        by_ts[(r["task_type"], st)].append(sc); by_t[r["task_type"]].append(sc)
    ts_arg = {k: int(np.mean(np.asarray(v), 0).argmax()) for k, v in by_ts.items()}
    t_arg = {k: int(np.mean(np.asarray(v), 0).argmax()) for k, v in by_t.items()}

    d2t = json.load(open(f"{REPO}/config/task2dataset.json"))
    ds2task = {d: t for t, ds in d2t.items() for d in ds}
    idx = torch.load(a.index, map_location="cpu", weights_only=False)
    datasets = [k[:-7] for k in idx if k.endswith("/scores") and k[:-7] in ds2task]

    def act_task(ds): return t_arg.get(ds2task[ds], 0)
    def act_sub(ds):
        t = ds2task[ds]; st = LB_SUBTYPE.get(ds, "_")
        return ts_arg.get((t, st), t_arg.get(t, 0))

    for tag, fn in [("WAITS(task)", act_task), ("WAITS(sub_type)", act_sub)]:
        out_dir = f"{REPO}/result_txt/backup/{a.model}/{a.budget}/{a.model}_{tag}_{a.budget}"
        os.makedirs(out_dir, exist_ok=True)
        for ds in datasets:
            ai = fn(ds)
            preds = idx[f"{ds}/preds"]; answers = idx[f"{ds}/answers"]
            allc = idx[f"{ds}/all_classes"]; lens = idx[f"{ds}/lengths"]
            with open(f"{out_dir}/{ds}.jsonl", "w", encoding="utf-8") as f:
                for i in range(len(preds)):
                    rec = {"pred": preds[i][ai], "answers": list(answers[i]),
                           "all_classes": allc[i], "length": int(lens[i])}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # score from the actual assembled preds (overwrites the index-derived result.json)
        subprocess.run([sys.executable, f"{REPO}/longbench_eval.py", out_dir],
                       cwd=REPO, capture_output=True)
        res = json.load(open(f"{out_dir}/result.json"))
        print(f"{a.model} {tag:16}: overall={res['overall_average']:.2f}  ({len(datasets)} datasets) -> {out_dir}")


if __name__ == "__main__":
    main()
