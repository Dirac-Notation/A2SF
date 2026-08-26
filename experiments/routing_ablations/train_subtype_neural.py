"""TEST: learn a neural contextual-bandit Q-function over [task | subtype] one-hot.

This makes WAITS sub-type routing a LEARNED RL policy instead of a hardcoded argmax
lookup table:
    state  s = [task_onehot | subtype_onehot]              (observable metadata)
    reward r = action_scores_gt_by_budget[budget]  (13-dim, the bandit reward)
    Q_theta(s) -> 13   (MLP), trained by MSE (fitted-Q) and listwise softmax-CE
    policy   pi(s) = argmax_a Q_theta(s)
Evaluated on the real-LB per-action index (same protocol as the table). Compares the
learned neural policy vs the hardcoded table (subtype) vs task-fixed.

  python script/train_subtype_neural.py --model llama3-1b \
      --recipe datasets/training/raw/recipe_v3_1b/train.jsonl \
      --index runs/fast_lb_eval/index_llama3-1b_128.pt --budget 128
"""
import argparse, json, os
import numpy as np, torch, torch.nn as nn
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# recipe-consistent subtype: only Few-Shot datasets carry a sub-type; everything else "_"
FEWSHOT_SUBTYPE = {"trec": "classification", "triviaqa": "qa", "samsum": "dialogue"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--budget", default="128")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--loss", choices=["mse", "listwise"], default="mse")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    rows = [json.loads(l) for l in open(a.recipe)]
    tasks = sorted({r["task_type"] for r in rows})
    subs = sorted({r.get("subtype", "_") for r in rows})
    ti = {t: i for i, t in enumerate(tasks)}; si = {s: i for i, s in enumerate(subs)}
    D = len(tasks) + len(subs)

    def encode(task, sub):
        v = np.zeros(D, np.float32)
        v[ti[task]] = 1.0
        v[len(tasks) + si.get(sub, si.get("_", 0))] = 1.0
        return v

    X, R = [], []
    for r in rows:
        sc = r["action_scores_gt_by_budget"]; sc = sc[a.budget] if isinstance(sc, dict) else sc
        X.append(encode(r["task_type"], r.get("subtype", "_"))); R.append(sc)
    X = torch.tensor(np.array(X)); R = torch.tensor(np.array(R, np.float32))

    # --- model: small MLP Q(s) -> 13 ---
    net = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 13))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for ep in range(a.epochs):
        opt.zero_grad()
        q = net(X)
        if a.loss == "mse":
            loss = ((q - R) ** 2).mean()
        else:  # listwise softmax-CE, temp 0.1 (champion)
            tgt = torch.softmax(R / 0.1, dim=1)
            loss = -(tgt * torch.log_softmax(q / 0.1, dim=1)).sum(1).mean()
        loss.backward(); opt.step()

    # --- eval on real-LB index ---
    d2t = json.load(open(f"{REPO}/config/task2dataset.json"))
    ds2task = {d: t for t, ds in d2t.items() for d in ds}
    idx = torch.load(a.index, map_location="cpu", weights_only=False)
    # table baselines (recompute for reference)
    by_ts, by_t = defaultdict(list), defaultdict(list)
    for r in rows:
        sc = r["action_scores_gt_by_budget"]; sc = sc[a.budget] if isinstance(sc, dict) else sc
        by_ts[(r["task_type"], r.get("subtype", "_"))].append(sc); by_t[r["task_type"]].append(sc)
    ts_arg = {k: int(np.mean(v, 0).argmax()) for k, v in by_ts.items()}
    t_arg = {k: int(np.mean(v, 0).argmax()) for k, v in by_t.items()}

    net.eval()
    v_nn, v_st, v_tf = [], [], []
    with torch.no_grad():
        for k in idx:
            if not k.endswith("/scores"): continue
            ds = k[:-7]; sc = np.asarray(idx[k]); t = ds2task.get(ds)
            if sc.ndim != 2 or sc.shape[1] != 13 or t is None: continue
            sub = FEWSHOT_SUBTYPE.get(ds, "_")
            a_nn = int(net(torch.tensor(encode(t, sub))[None]).argmax())
            a_st = ts_arg.get((t, sub), t_arg.get(t, 0))
            a_tf = t_arg.get(t, 0)
            v_nn.append(sc[:, a_nn].mean()); v_st.append(sc[:, a_st].mean()); v_tf.append(sc[:, a_tf].mean())
    print(f"{a.model} [{a.loss}] task-fixed={np.mean(v_tf):.2f}  table-subtype={np.mean(v_st):.2f}  "
          f"NEURAL-Q={np.mean(v_nn):.2f}  (NN-table {np.mean(v_nn)-np.mean(v_st):+.2f})")


if __name__ == "__main__":
    main()
