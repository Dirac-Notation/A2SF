"""Cell labels taken from the recipe's own task type instead of the A1 classifier.

The A1 classifier misfires on part of the corpus, which shows up as label noise in the cell
table. The recipe is data we generated, so its task types are known and can be used directly;
LongBench is still labelled by A1 at evaluation time. Clean protocol is preserved: no LB
information enters training.
"""
import argparse, json, os, sys
import numpy as np

TRACES = os.environ.get("ICLR_TRACES", "/data2/smp9898/iclr_traces")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0,'.')
from iclr.proxy_gate import tanimoto, load_store
from iclr.recipe_table import KEYS13, bucket
d2m=json.load(open("config/dataset2maxlen.json"))

# metric -> cell class. For recipe-v5 rows the scoring metric already identifies the task,
# so it takes precedence.
_METRIC_CLS={"qa_f1_score":0,"rouge_score":1,"code_sim_score":2,
             "classification_score":3,"retrieval_score":4,"count_score":4}

def true_cls(ds, tt, metric=None):
    if metric in _METRIC_CLS: return _METRIC_CLS[metric]
    if "QA" in tt or ds=="div_qa": return 0
    if tt=="Summarization" or ds=="div_dialog": return 1
    if tt=="Code Complete": return 2
    if ds=="div_class": return 3
    if tt=="Passage Retrieval": return 4
    return 0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True)
    ap.add_argument("--pool",default="poolx"); ap.add_argument("--min_cell",type=int,default=10)
    a=ap.parse_args()
    za=np.load(f"{TRACES}/a1_recipe/{a.pool}_{a.model}.npz")
    a1r={str(za["idx"][i]):int(za["cls"][i]) for i in range(len(za["idx"]))}
    rows=[]
    for obj in (json.loads(l) for l in open(f"datasets/training/raw/recipe_{a.pool}_{a.model}.jsonl")):
        sid=str(obj["sample_id"]); fc=(obj.get("full_cache_pred") or "")
        if not fc.strip() or len(obj.get("action_outputs",[]))!=13: continue
        p0=np.array([tanimoto(o or "",fc) for o in obj["action_outputs"]])
        rows.append({"b":bucket(obj["generation_length"]),
                     "a1":a1r.get(sid), "tc":true_cls(obj["dataset"],obj["task_type"]),"p0":p0})
    def table(keyfn):
        cells={}
        for r in rows:
            c=keyfn(r)
            if c is None: continue
            cells.setdefault(c,[]).append(r["p0"])
        g=int(np.argmax(np.stack([r["p0"] for r in rows]).mean(0)))
        return {c:(int(np.argmax(np.stack(v).mean(0))) if len(v)>=a.min_cell else g) for c,v in cells.items()}, g
    store=load_store(a.model); gt={}
    for (ds,i),r in store.items():
        if all(x in r["actions"] for x in KEYS13): gt[(ds,i)]=np.array([r["actions"][x]["score"] for x in KEYS13])
    zl=np.load(f"{TRACES}/a1/{a.model}.npz")
    a1={(str(zl["ds"][i]),int(zl["idx"][i])):int(zl["cls"][i]) for i in range(len(zl["idx"]))}
    ks=sorted(set(gt)&set(a1))
    def lb(tab,g):
        per,pg={},{}
        for k in ks:
            c=(bucket(d2m[k[0]]),a1[k])
            per.setdefault(k[0],[]).append(gt[k][tab.get(c,g)]); pg.setdefault(k[0],[]).append(gt[k][g])
        mac=lambda d: float(np.mean([np.mean(v) for v in d.values()]))
        return mac(per),mac(pg)
    t1,g1=table(lambda r:(r["b"],r["a1"]) if r["a1"] is not None else None)
    t2,g2=table(lambda r:(r["b"],r["tc"]))
    l1,gl=lb(t1,g1); l2,_=lb(t2,g2)
    print(f"{a.model:12s} A1 labels {l1:6.2f} | task-type labels {l2:6.2f} ({l2-l1:+.2f}) | global {gl:6.2f}")
    json.dump({"a1_label":l1,"true_label":l2,"glob":gl},
              open(f"result_txt/analysis/proxy_gate/cleancell_{a.model}.json","w"),indent=1)
if __name__=="__main__": main()
