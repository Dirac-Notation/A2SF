"""Build meta-only states for a TRAINING jsonl (not LongBench). Same 19-d layout as
script/build_meta_states.py (seq_len_norm + metric_oh + task_oh + side dummy), keyed by
line index = train.py's _prompt_id. Lets meta-only train on the clean synthetic/clean corpus.

  python script/build_train_meta_states.py --train datasets/training/scored/cleanrecipe_1b/train.jsonl \
      --val datasets/training/scored/cleanrecipe_1b/validation.jsonl --out runs/states/cleanrecipe_1b_meta.pt
"""
import argparse, json, os, sys
import torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from RL.metadata import METRIC_TYPE_ORDER, TASK_TYPE_ORDER
MAXSEQ = 32768
nm, nt = len(METRIC_TYPE_ORDER), len(TASK_TYPE_ORDER)
SD = 1 + nm + nt + 1

def midx(m): return METRIC_TYPE_ORDER.index(m) if m in METRIC_TYPE_ORDER else nm - 1
def tidx(t): return TASK_TYPE_ORDER.index(t) if t in TASK_TYPE_ORDER else nt - 1

def state_for(row):
    s = torch.zeros(SD)
    L = float(row.get("length", 4000) or 4000)
    s[0] = min(L, MAXSEQ) / MAXSEQ
    s[1 + midx(row.get("metric_type", "unknown"))] = 1.0
    s[1 + nm + tidx(row.get("task_type", "unknown"))] = 1.0
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    tr = [json.loads(l) for l in open(a.train)]
    vl = [json.loads(l) for l in open(a.val)]
    out = {"state_dim": SD, "num_metric_types": nm, "num_task_types": nt,
           "side_dim": 1, "num_heads": 1, "num_hidden_pool": 0, "config": {}}
    N = len(tr)
    for i, r in enumerate(tr): out[i] = state_for(r)
    for j, r in enumerate(vl): out[N + j] = state_for(r)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(out, a.out)
    print(f"saved {a.out}: {N} train + {len(vl)} val states, state_dim={SD}, num_task={nt}")

if __name__ == "__main__":
    main()
