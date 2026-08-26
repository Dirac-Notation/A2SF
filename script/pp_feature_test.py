"""Stage-3 decisive test: does a per-prompt FEATURE (e.g. answer_pos) let a policy beat the
per-task lookup? k-fold CV on a scored dataset. Compares, on held-out prompts:
  (a) per-task lookup (train-fold per-task argmax)  -- the meta-only ceiling
  (b) MLP with [task_oh | pp_feature]               -- per-prompt policy
Reports mean held-out reward of each (the action's GT score). If (b) > (a), the feature carries
usable per-prompt signal beyond task-fixed.

  python script/pp_feature_test.py --scored datasets/training/raw/postest_qa_1b --feat answer_pos
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn as nn
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from RL.metadata import TASK_TYPE_ORDER
NT = len(TASK_TYPE_ORDER)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True); ap.add_argument("--feat", default="answer_pos")
    ap.add_argument("--budget", default="128"); ap.add_argument("--K", type=int, default=5)
    a = ap.parse_args()
    com = {json.loads(l)["sample_id"]: json.loads(l) for l in open(f"{a.scored}/common.jsonl")}
    rows = []
    for l in open(f"{a.scored}/budget_{a.budget}.jsonl"):
        r = json.loads(l); c = com.get(r["sample_id"])
        if not c or len(r["action_scores_gt"]) != 13: continue
        rows.append((c.get("task_type"), float(c.get(a.feat, -1)), np.array(r["action_scores_gt"], np.float32)))
    tasks = sorted(set(t for t, _, _ in rows))
    tidx = {t: i for i, t in enumerate(tasks)}
    Y = np.stack([y for _, _, y in rows]); F = np.array([f for _, f, _ in rows], np.float32)
    T = np.array([tidx[t] for t, _, _ in rows])
    n = len(rows); rng = np.random.RandomState(42); perm = rng.permutation(n); folds = np.array_split(perm, a.K)
    lookup_r, mlp_r, fc_best = [], [], []
    for i in range(a.K):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(a.K) if j != i])
        # (a) lookup: per-task argmax on train fold
        la = {}
        for t in range(len(tasks)):
            m = tr[T[tr] == t]
            la[t] = int(Y[m].mean(0).argmax()) if len(m) else 0
        lookup_r += [Y[k, la[T[k]]] for k in te]
        fc_best += [Y[k].max() for k in te]  # per-prompt oracle
        # (b) MLP [task_oh | feat]
        def feat(idx):
            X = np.zeros((len(idx), NT + 1), np.float32)
            for j, k in enumerate(idx): X[j, T[k]] = 1.0; X[j, NT] = F[k]
            return torch.tensor(X)
        Xtr, Ytr = feat(tr), torch.tensor(Y[tr]); Xte = feat(te)
        pol = nn.Sequential(nn.Linear(NT + 1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 13))
        opt = torch.optim.Adam(pol.parameters(), lr=3e-3, weight_decay=1e-4)
        for ep in range(600):
            opt.zero_grad(); nn.functional.mse_loss(pol(Xtr), Ytr).backward(); opt.step()
        with torch.no_grad():
            acts = pol(Xte).argmax(1).numpy()
        mlp_r += [Y[te[j], acts[j]] for j in range(len(te))]
    print(f"dataset={a.scored}  feat={a.feat}  n={n}  tasks={tasks}")
    print(f"  per-task LOOKUP (meta ceiling) : {np.mean(lookup_r)*100:.2f}")
    print(f"  MLP [task | {a.feat}]          : {np.mean(mlp_r)*100:.2f}   (gain {(np.mean(mlp_r)-np.mean(lookup_r))*100:+.2f})")
    print(f"  per-prompt ORACLE (upper bound): {np.mean(fc_best)*100:.2f}")

if __name__ == "__main__":
    main()
