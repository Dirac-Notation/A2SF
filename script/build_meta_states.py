"""Build MODEL-AGNOSTIC meta-only LB states from the index (no model forward).
state = [seq_len_norm(1), metric_one_hot(10), task_one_hot(7)] = 18-d. The per-task head
reads task_oh at [11:18]; with CV (representative) data it learns the per-task best action
(= task-fixed, which beats SnapKV). Works for ANY model (qwen2/mistral) since the LLaMA-only
encoders fail there.

  python script/build_meta_states.py --model qwen2 --out runs/fast_lb_eval/lb_states_qwen2_meta.pt
"""
import argparse, json, os, sys
import torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from RL.metadata import METRIC_TYPE_ORDER, TASK_TYPE_ORDER

DS_METRIC = {"hotpotqa":"qa_f1_score","2wikimqa":"qa_f1_score","multifieldqa_en":"qa_f1_score",
    "qasper":"qa_f1_score","narrativeqa":"qa_f1_score","triviaqa":"qa_f1_score","musique":"qa_f1_score",
    "gov_report":"rouge_score","qmsum":"rouge_score","multi_news":"rouge_score","samsum":"rouge_score",
    "trec":"classification_score","passage_retrieval_en":"retrieval_score","passage_count":"count_score",
    "lcc":"code_sim_score","repobench-p":"code_sim_score"}
MAXSEQ = 32768

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
t2d = json.load(open(f"{REPO}/config/task2dataset.json")); ds2task = {d: t for t, dl in t2d.items() for d in dl}
ix = torch.load(f"{REPO}/runs/fast_lb_eval/index_{a.model}_128.pt", map_location="cpu")
nm, nt = len(METRIC_TYPE_ORDER), len(TASK_TYPE_ORDER)
def midx(ds): m = DS_METRIC.get(ds, "unknown"); return METRIC_TYPE_ORDER.index(m) if m in METRIC_TYPE_ORDER else nm - 1
def tidx(ds): t = ds2task.get(ds, "unknown"); return TASK_TYPE_ORDER.index(t) if t in TASK_TYPE_ORDER else nt - 1
# NeuralUCBAgent meta-only layout: state = [meta(1+nm+nt) | side(1)] = 19-d, side_dim=1,num_heads=1
SD = 1 + nm + nt + 1
out = {"state_dim": SD, "num_metric_types": nm, "num_task_types": nt,
       "side_dim": 1, "num_heads": 1, "num_hidden_pool": 0, "config": {}}
dss = sorted(k[:-7] for k in ix if k.endswith("/scores"))
for ds in dss:
    N = ix[ds + "/scores"].shape[0]
    lengths = ix.get(ds + "/lengths", None)
    states = torch.zeros(N, SD)
    for i in range(N):
        L = float(lengths[i]) if lengths is not None and i < len(lengths) and lengths[i] is not None else 4000.0
        states[i, 0] = min(L, MAXSEQ) / MAXSEQ
        states[i, 1 + midx(ds)] = 1.0
        states[i, 1 + nm + tidx(ds)] = 1.0
    out[ds + "/states"] = states
    out[ds + "/order"] = torch.arange(N)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
torch.save(out, a.out)
print(f"saved {a.out}: state_dim={out['state_dim']}, {len(dss)} datasets, num_task={nt}")
