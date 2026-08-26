"""Sub-type routing meta-policy: per-(task, SUBTYPE) argmax from a diversified recipe, applied
to LB by mapping each LB dataset to its observable sub-type. Tests whether sub-type metadata
(generalizable, observable from prompt format) lets the meta-policy EXCEED task-fixed.
Falls back to per-task argmax when a task has no sub-type split in the recipe.

  python script/subtype_lookup_eval.py --train datasets/training/raw/recipe_v3_1b/train.jsonl \
      --index runs/fast_lb_eval/index_llama3-1b_128.pt
"""
import argparse, json, os
import numpy as np, torch
from collections import defaultdict
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACT = ['H2O','.01/1','.01/16','.01/32','.01/128','.1/1','.1/16','.1/32','.1/128','10/1','10/16','10/32','10/128']
# LB dataset -> observable sub-type (the category a prompt-detector would output)
LB_SUBTYPE = {
    "trec": "classification", "triviaqa": "qa", "samsum": "dialogue",          # Few Shot
    "narrativeqa": "qa", "qasper": "qa", "multifieldqa_en": "qa",              # Single-doc QA
    "hotpotqa": "qa", "2wikimqa": "qa", "musique": "qa",                       # Multi-doc QA
    "gov_report": "summary", "multi_news": "summary", "qmsum": "summary",      # Summarization
    "lcc": "code", "repobench-p": "code",                                      # Code
    "passage_retrieval_en": "retrieval", "passage_count": "retrieval",         # Passage Retrieval
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--index", required=True); ap.add_argument("--budget", default="128")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.train)]
    # per-(task, subtype) and per-task argmax
    by_ts = defaultdict(list); by_t = defaultdict(list)
    for r in rows:
        sc = r["action_scores_gt_by_budget"]; sc = sc[a.budget] if isinstance(sc, dict) else sc
        st = r.get("subtype", "_")
        by_ts[(r["task_type"], st)].append(sc); by_t[r["task_type"]].append(sc)
    ts_arg = {k: int(np.mean(np.asarray(v), 0).argmax()) for k, v in by_ts.items()}
    t_arg = {k: int(np.mean(np.asarray(v), 0).argmax()) for k, v in by_t.items()}
    d2t = json.load(open(f"{REPO}/config/task2dataset.json")); ds2task = {d: t for t, ds in d2t.items() for d in ds}
    idx = torch.load(a.index, map_location="cpu", weights_only=False)
    vals_st = []; vals_tf = []
    for k in idx:
        if not k.endswith("/scores"): continue
        ds = k[:-7]; sc = np.asarray(idx[k]); t = ds2task.get(ds)
        if sc.ndim != 2 or sc.shape[1] != 13 or t is None: continue
        st = LB_SUBTYPE.get(ds, "_")
        a_st = ts_arg.get((t, st), t_arg.get(t, 0))   # subtype action, fall back to task
        a_tf = t_arg.get(t, 0)
        vals_st.append(sc[:, a_st].mean()); vals_tf.append(sc[:, a_tf].mean())
    print(f"task-fixed (per-task)      : {np.mean(vals_tf):.2f}")
    print(f"SUB-TYPE routing           : {np.mean(vals_st):.2f}   (gain {(np.mean(vals_st)-np.mean(vals_tf)):+.2f})")
    print("\nrecipe per-(task,subtype) argmax:")
    for (t, st), ai in sorted(ts_arg.items()):
        print(f"  {t:18} {st:14}: {ACT[ai]}")

if __name__ == "__main__":
    main()
