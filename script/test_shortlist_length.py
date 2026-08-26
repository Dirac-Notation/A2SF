"""Coarse-to-fine test: (task,metric) -> top-K candidate actions (floor), then per-prompt
pick ONE among the K using LENGTH. All shortlist/length-rules learned on the RECIPE (no LB
training); evaluated on the real-LB index. Reports:
  WAITS (K=1 metadata-fixed, floor) | shortlist-ORACLE@K (best-of-K per prompt, upper bound)
  | LENGTH-rule@K (realizable: length-bin -> best-of-K from recipe).
"""
import json, numpy as np, torch, os
from collections import defaultdict
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEWSHOT_METRIC={"trec":"classification_score","triviaqa":"qa_f1_score","samsum":"rouge_score"}
TASK_METRIC={"Single-doc QA":"qa_f1_score","Multi-doc QA":"qa_f1_score","Passage Retrieval":"qa_f1_score","Code Complete":"code_sim_score","Summarization":"rouge_score"}
LBINS=[0,2000,4000,8000,16000,1e9]   # context-length bins
def lbin(x):
    for i in range(len(LBINS)-1):
        if LBINS[i]<=x<LBINS[i+1]: return i
    return len(LBINS)-2
REC={'llama3-1b':'recipe_v3_1b','llama3-8b':'recipe_v3_8b','qwen2':'recipe_v3_qwen','mistral-7b':'recipe_v3_mistral'}
d2t=json.load(open(f"{REPO}/config/task2dataset.json")); ds2task={d:t for t,ds in d2t.items() for d in ds}

def run(model, K):
    rows=[json.loads(l) for l in open(f"{REPO}/datasets/training/raw/{REC[model]}/train.jsonl")]
    def rew(r): sc=r["action_scores_gt_by_budget"]; return np.asarray(sc["128"] if isinstance(sc,dict) else sc)
    def length(r): return float(r.get("length") or r.get("token_budget") or 4000)
    cell=defaultdict(list)              # (task,metric) -> list of (rewardvec, length)
    for r in rows: cell[(r["task_type"], r.get("metric_type","?"))].append((rew(r), length(r)))
    topK={}; lenrule={}                 # cell -> top-K action idx ; cell -> {lbin: action}
    for c,lst in cell.items():
        R=np.array([x[0] for x in lst]); L=np.array([x[1] for x in lst])
        meanR=R.mean(0); cand=list(np.argsort(meanR)[-K:])    # top-K actions
        topK[c]=cand
        rule={}
        for b in range(len(LBINS)-1):
            m=np.array([lbin(x)==b for x in L])
            if m.sum()>=5:
                # among candidates, best mean reward in this length-bin
                rule[b]=cand[int(np.argmax(R[m][:,cand].mean(0)))]
        rule_default=cand[int(np.argmax(meanR[cand]))]   # = WAITS top-1
        lenrule[c]=(rule, rule_default)

    idx=torch.load(f"{REPO}/runs/fast_lb_eval/index_{model}_128.pt", map_location="cpu", weights_only=False)
    v_waits=[]; v_oracle=[]; v_len=[]
    for k in idx:
        if not k.endswith("/scores"): continue
        ds=k[:-7]; t=ds2task.get(ds)
        if t is None: continue
        sc=np.asarray(idx[k]); lens=np.asarray(idx.get(f"{ds}/lengths",[4000]*len(sc)))
        met=FEWSHOT_METRIC.get(ds, TASK_METRIC.get(t,"qa_f1_score"))
        c=(t,met); cand=topK.get(c)
        if cand is None: continue
        a_waits=cand[int(np.argmax([sc[:,a].mean() for a in cand]))]   # cell top-1 (=WAITS)
        v_waits.append(sc[:,a_waits].mean())
        v_oracle.append(sc[:,cand].max(1).mean())                       # per-prompt best-of-K
        rule,dflt=lenrule[c]
        per=[sc[i, rule.get(lbin(lens[i]) if i<len(lens) else 0, dflt)] for i in range(len(sc))]
        v_len.append(np.mean(per))
    return np.mean(v_waits), np.mean(v_oracle), np.mean(v_len)

for model in ['llama3-1b','llama3-8b','qwen2','mistral-7b']:
    line=f"{model:11}"
    for K in [2,3]:
        w,o,l=run(model,K)
        line+=f" | K={K}: WAITS={w:.2f} oracle@K={o:.2f}(+{o-w:.2f}) LENrule={l:.2f}(+{l-w:.2f})"
    print(line)
