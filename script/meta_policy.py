"""Clean, fast LEARNABLE meta-policy (replaces the slow NeuralUCB that collapses on
meta-only due to shared-backbone cross-task interference). Predicts the 13-action reward
vector from a per-prompt feature vector; argmax = chosen action.

Stage 2 (meta-only): features = [task_oh | metric_oh].  Since these are constant per task,
MSE regression converges to the per-task mean reward -> argmax = per-task lookup (= task-fixed).
Stage 3: append model-agnostic per-prompt features -> can beat the lookup.

  train: python script/meta_policy.py train --train <train.jsonl> --val <val.jsonl> --out runs/policy/meta.pt
  eval : python script/meta_policy.py eval  --ckpt runs/policy/meta.pt --states <lb_states>.pt --index <index>.pt
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn as nn
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from RL.metadata import METRIC_TYPE_ORDER, TASK_TYPE_ORDER
NM, NT = len(METRIC_TYPE_ORDER), len(TASK_TYPE_ORDER)

class MetaPolicy(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 13))
    def forward(self, x): return self.net(x)

SUBTYPES = ["_", "classification", "qa", "dialogue", "summary", "code", "retrieval"]
def sub_oh(st):
    f = np.zeros(len(SUBTYPES), np.float32); f[SUBTYPES.index(st) if st in SUBTYPES else 0] = 1.0; return f

def meta_feat(task_type, metric_type, extra=None, subtype=None):
    # task one-hot (+ observable SUB-TYPE one-hot — generalizable metadata that beats task-fixed).
    f = np.zeros(NT, np.float32)
    if task_type in TASK_TYPE_ORDER: f[TASK_TYPE_ORDER.index(task_type)] = 1.0
    if subtype is not None: f = np.concatenate([f, sub_oh(subtype)])
    if extra is not None: f = np.concatenate([f, np.asarray(extra, np.float32)])
    return f

def load_xy(path, budget="128"):
    X, Y = [], []
    for l in open(path):
        r = json.loads(l)
        sc = r["action_scores_gt_by_budget"]; sc = sc[budget] if isinstance(sc, dict) else sc
        if len(sc) != 13: continue
        X.append(meta_feat(r.get("task_type"), r.get("metric_type"), r.get("pp_features")))
        Y.append(sc)
    return torch.tensor(np.array(X)), torch.tensor(np.array(Y), dtype=torch.float32)

def train(a):
    Xtr, Ytr = load_xy(a.train); Xv, Yv = load_xy(a.val)
    torch.manual_seed(0)
    pol = MetaPolicy(Xtr.shape[1], a.hidden)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3, weight_decay=1e-5)
    best = 1e9; best_sd = None
    for ep in range(a.epochs):
        pol.train(); opt.zero_grad()
        loss = nn.functional.mse_loss(pol(Xtr), Ytr); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == a.epochs - 1:
            pol.eval()
            with torch.no_grad():
                vl = nn.functional.mse_loss(pol(Xv), Yv).item()
                # val argmax reward
                arg = pol(Xv).argmax(1); rarg = Yv[torch.arange(len(Yv)), arg].mean().item()
            if vl < best: best = vl; best_sd = {k: v.clone() for k, v in pol.state_dict().items()}
            print(f"ep{ep} mse={loss.item():.4f} vmse={vl:.4f} v_rarg={rarg*100:.2f}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"sd": best_sd or pol.state_dict(), "in_dim": Xtr.shape[1], "hidden": a.hidden}, a.out)
    print(f"saved {a.out}")

def evl(a):
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    pol = MetaPolicy(ck["in_dim"], ck["hidden"]); pol.load_state_dict(ck["sd"]); pol.eval()
    st = torch.load(a.states, map_location="cpu", weights_only=False)
    idx = torch.load(a.index, map_location="cpu", weights_only=False)
    d2t = json.load(open(f"{REPO}/config/task2dataset.json")); ds2task = {d: t for t, ds in d2t.items() for d in ds}
    vals = []
    for k in idx:
        if not k.endswith("/scores"): continue
        ds = k[:-7]; sc = np.asarray(idx[k])
        if sc.ndim != 2 or sc.shape[1] != 13: continue
        states = st[f"{ds}/states"].float()  # (N,19): [seq|metric10|task7|side]
        # rebuild meta feat [task7|metric10] from the LB state one-hots
        task_oh = states[:, 1 + NM:1 + NM + NT]
        X = task_oh
        with torch.no_grad():
            acts = pol(X).argmax(1).numpy()
        n = min(len(acts), sc.shape[0])
        vals.append(np.mean([sc[i, acts[i]] for i in range(n)]))
    print(f"★ learned meta-policy LB Avg = {np.mean(vals):.2f}  (over {len(vals)} datasets)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train"); t.add_argument("--train", required=True); t.add_argument("--val", required=True)
    t.add_argument("--out", required=True); t.add_argument("--epochs", type=int, default=500); t.add_argument("--hidden", type=int, default=128)
    e = sub.add_parser("eval"); e.add_argument("--ckpt", required=True); e.add_argument("--states", required=True); e.add_argument("--index", required=True)
    a = ap.parse_args()
    (train if a.cmd == "train" else evl)(a)
