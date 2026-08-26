"""Emit longbench_eval-format result.json for WAITS(task) and WAITS(sub_type) routing,
from the real-LB per-action index + a recipe's per-(task[,subtype]) argmax action.

  python script/make_waits_result.py --model llama3-1b \
      --recipe datasets/training/raw/recipe_v3_1b/train.jsonl \
      --index runs/fast_lb_eval/index_llama3-1b_128.pt --budget 128
"""
import argparse, json, os
import numpy as np, torch
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys; sys.path.insert(0, REPO)
from longbench_eval import calculate_group_averages

LB_SUBTYPE = {
    "trec": "classification", "triviaqa": "qa", "samsum": "dialogue",
    "narrativeqa": "qa", "qasper": "qa", "multifieldqa_en": "qa",
    "hotpotqa": "qa", "2wikimqa": "qa", "musique": "qa",
    "gov_report": "summary", "multi_news": "summary", "qmsum": "summary",
    "lcc": "code", "repobench-p": "code",
    "passage_retrieval_en": "retrieval", "passage_count": "retrieval",
}


def build(scores_by_ds, action_of):
    ind = {ds: round(float(sc[:, action_of(ds)].mean()), 2) for ds, sc in scores_by_ds.items()}
    grp = calculate_group_averages(ind)
    overall = round(sum(ind.values()) / len(ind), 2)
    return {"individual_scores": ind, "group_averages": grp, "overall_average": overall}


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
    scores_by_ds = {}
    for k in idx:
        if not k.endswith("/scores"): continue
        ds = k[:-7]; sc = np.asarray(idx[k])
        if sc.ndim == 2 and sc.shape[1] == 13 and ds in ds2task:
            scores_by_ds[ds] = sc

    def act_task(ds): return t_arg.get(ds2task[ds], 0)
    def act_sub(ds):
        t = ds2task[ds]; st = LB_SUBTYPE.get(ds, "_")
        return ts_arg.get((t, st), t_arg.get(t, 0))

    for tag, fn in [("WAITS(task)", act_task), ("WAITS(sub_type)", act_sub)]:
        res = build(scores_by_ds, fn)
        out_dir = f"{REPO}/result_txt/backup/{a.model}/{a.budget}/{a.model}_{tag}_{a.budget}"
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/result.json", "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
        print(f"{a.model} {tag:16}: overall={res['overall_average']:.2f}  -> {out_dir}/result.json")


if __name__ == "__main__":
    main()
