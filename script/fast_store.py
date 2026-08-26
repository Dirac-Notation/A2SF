"""Unified fast-eval store: per-sample x per-action {pred, score} for LongBench.

File: result_txt/backup/fast_store/store_<model>_<budget>.jsonl.gz (canonical, see store_path())
  one line per LB sample:
  {"dataset": ds, "idx": i, "answers": [...], "all_classes": ..., "length": L,
   "actions": {"<a>:<b>": {"pred": str, "score": float}, ..., "full": {...}}}

Scores are FINAL at build time (official longbench_eval post-processing applied:
first-line truncation for trec/triviaqa/samsum/lsht). All downstream analyses must
LOOK UP from this store — never rescore preds (memory feedback_fast_eval_index_ext).

API:
  load_store(model, budget=128)                    -> {(ds, idx): row}
  add_action(model, action_key, pred_dir, budget)  -> scores + merges a new action in
  routing_value(store, table_key_fn)               -> overall (mean over datasets)
  action_matrix(store, action_keys)                -> {(ds,idx): np.array of scores}

CLI:
  python script/fast_store.py build --model llama3-1b            # initial build
  python script/fast_store.py add --model llama3-1b \
      --action 1.0:64 --pred_dir result_txt/pred/128/fix_a1_b64_llama3-1b_128
  python script/fast_store.py table --model llama3-1b --waits_table runs/waits_tables/x.json
"""
import argparse, gzip, json, os, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np

from longbench_eval import (qa_f1_score, rouge_score, classification_score,
                            retrieval_score, count_score, code_sim_score)

METRIC_FN = {"qa_f1_score": qa_f1_score, "rouge_score": rouge_score,
             "classification_score": classification_score, "retrieval_score": retrieval_score,
             "count_score": count_score, "code_sim_score": code_sim_score}
DS_METRIC = {"narrativeqa": "qa_f1_score", "qasper": "qa_f1_score", "multifieldqa_en": "qa_f1_score",
             "hotpotqa": "qa_f1_score", "2wikimqa": "qa_f1_score", "musique": "qa_f1_score",
             "gov_report": "rouge_score", "qmsum": "rouge_score", "multi_news": "rouge_score",
             "samsum": "rouge_score", "trec": "classification_score", "triviaqa": "qa_f1_score",
             "lcc": "code_sim_score", "repobench-p": "code_sim_score",
             "passage_count": "count_score", "passage_retrieval_en": "retrieval_score"}
FIRSTLINE = {"trec", "triviaqa", "samsum", "lsht"}
GRID13 = [(0.0, 1), (0.01, 1), (0.01, 16), (0.01, 32), (0.01, 128), (0.1, 1), (0.1, 16),
          (0.1, 32), (0.1, 128), (10.0, 1), (10.0, 16), (10.0, 32), (10.0, 128)]

def akey(a, b): return f"{float(a):g}:{int(b)}"

def store_path(model, budget=128):
    return f"{ROOT}/result_txt/backup/fast_store/store_{model}_{budget}.jsonl.gz"

def official_score(ds, pred, answers, all_classes):
    if ds in FIRSTLINE:
        pred = pred.lstrip("\n").split("\n")[0]
    fn = METRIC_FN[DS_METRIC[ds]]
    if not answers:
        return 0.0
    return float(max(fn(pred, a, all_classes=all_classes) for a in answers) * 100)

def load_store(model, budget=128):
    st = {}
    with gzip.open(store_path(model, budget), "rt") as f:
        for line in f:
            r = json.loads(line)
            st[(r["dataset"], r["idx"])] = r
    return st

def save_store(st, model, budget=128):
    tmp = store_path(model, budget) + ".tmp"
    with gzip.open(tmp, "wt") as f:
        for k in sorted(st):
            f.write(json.dumps(st[k], ensure_ascii=False) + "\n")
    os.replace(tmp, store_path(model, budget))

def add_action(model, action_key, pred_dir, budget=128, st=None):
    """Merge a fixed-action pred dir into the store (scores computed here, once)."""
    own = st is None
    if own: st = load_store(model, budget)
    n_add = 0
    for ds in DS_METRIC:
        f = os.path.join(pred_dir, f"{ds}.jsonl")
        if not os.path.exists(f): continue
        for j, r in enumerate(map(json.loads, open(f))):
            i = int(r.get("idx", j))
            key = (ds, i)
            if key not in st: continue
            sc = official_score(ds, r["pred"], st[key]["answers"], st[key].get("all_classes"))
            st[key]["actions"][action_key] = {"pred": r["pred"], "score": round(sc, 4)}
            n_add += 1
    if own: save_store(st, model, budget)
    print(f"[store] {model}: action '{action_key}' merged ({n_add} samples)")
    return st

def action_matrix(store, action_keys):
    out = {}
    for k, r in store.items():
        out[k] = np.array([r["actions"][a]["score"] if a in r["actions"] else np.nan
                           for a in action_keys])
    return out

def routing_value(store, choose_fn):
    """choose_fn(ds) -> action_key. Returns overall (mean over datasets of per-ds mean)."""
    per = defaultdict(list)
    for (ds, i), r in store.items():
        a = choose_fn(ds)
        if a in r["actions"]:
            per[ds].append(r["actions"][a]["score"])
    return float(np.mean([np.mean(v) for v in per.values() if v]))

def build(model, budget=128):
    import torch
    idx = torch.load(f"{ROOT}/runs/fast_lb_eval/index_{model}_{budget}.pt",
                     map_location="cpu", weights_only=False)
    st = {}
    for kk in idx:
        if not kk.endswith("/scores"): continue
        ds = kk[:-7]
        if ds not in DS_METRIC: continue
        sc = np.asarray(idx[kk], float)
        preds = idx[f"{ds}/preds"]; answers = idx[f"{ds}/answers"]
        acls = idx[f"{ds}/all_classes"]; lens = idx[f"{ds}/lengths"]
        for i in range(len(sc)):
            acts = {}
            if isinstance(preds[i], (list, tuple)):
                for ai, ab in enumerate(GRID13):
                    acts[akey(*ab)] = {"pred": preds[i][ai], "score": round(float(sc[i, ai]), 4)}
            st[(ds, i)] = {"dataset": ds, "idx": i, "answers": answers[i],
                           "all_classes": acls[i], "length": int(lens[i]), "actions": acts}
    # a=1.0 fixed runs (whatever exists)
    for tag, b in [("b1", 1), ("b16", 16), ("b32", 32), ("b128", 128)]:
        pd = f"{ROOT}/result_txt/pred/128/fix_a1_{tag}_{model}_128"
        if os.path.isdir(pd):
            add_action(model, akey(1.0, b), pd, budget, st=st)
    # full cache (backup rows are SHARD-SHUFFLED -> match by (answers, length) fingerprint)
    fdir = f"{ROOT}/result_txt/backup/{model}/{model}_full"
    if os.path.isdir(fdir):
        n_add = 0
        for ds in DS_METRIC:
            f = os.path.join(fdir, f"{ds}.jsonl")
            if not os.path.exists(f): continue
            pool = defaultdict(list)
            for r in map(json.loads, open(f)):
                pool[(json.dumps(r["answers"]), int(r["length"]))].append(r["pred"])
            for i in range(len([1 for k in st if k[0] == ds])):
                key = (ds, i)
                if key not in st: continue
                fp = (json.dumps(st[key]["answers"]), int(st[key]["length"]))
                if pool.get(fp):
                    pred = pool[fp].pop(0)
                    sc = official_score(ds, pred, st[key]["answers"], st[key].get("all_classes"))
                    st[key]["actions"]["full"] = {"pred": pred, "score": round(sc, 4)}
                    n_add += 1
        print(f"[store] {model}: 'full' merged ({n_add} samples, fingerprint-matched)")
    save_store(st, model, budget)
    n_act = len(next(iter(st.values()))["actions"])
    print(f"[store] built {store_path(model, budget)}: {len(st)} samples, ~{n_act} actions each")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "add", "table", "eval"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--action", help="add: action key 'a:b'")
    ap.add_argument("--pred_dir", help="add: fixed-action pred dir")
    ap.add_argument("--waits_table", help="table: waits_table json to evaluate")
    ap.add_argument("--agent", help="eval: trained selector artifact (.npz, RoutingNeuralUCB.save)")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.model, a.budget)
    elif a.cmd == "add":
        add_action(a.model, a.action, a.pred_dir, a.budget)
    elif a.cmd == "eval":
        # RUN the trained selector on each input: agent(input) -> action -> store lookup
        from RL.metadata import dataset_metric
        from RL.model import RoutingNeuralUCB
        d2t = json.load(open(f"{ROOT}/config/task2dataset.json"))
        ds2task = {d: t for t, ds in d2t.items() for d in ds}
        agent = RoutingNeuralUCB.load(a.agent)
        st = load_store(a.model, a.budget)
        def choose(ds):
            t = ds2task[ds]
            ab = agent.action_ab(agent.greedy_action(t, dataset_metric(ds, t)))
            return akey(*ab)
        print(f"selector eval = {routing_value(st, choose):.2f}")
    elif a.cmd == "table":
        st = load_store(a.model, a.budget)
        tab = json.load(open(a.waits_table))[f"{a.model}_{a.budget}"]
        v = routing_value(st, lambda ds: akey(*tab[ds][0]) if ds in tab else None)
        print(f"routing value = {v:.2f}")

if __name__ == "__main__":
    main()
