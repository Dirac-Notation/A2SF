"""Bandit router variants, compared against the supervised router on identical inputs.

RL/model.py notes that a categorical state makes the NeuralUCB Sherman-Morrison core reduce
exactly to LinUCB. That held when the state was [task one-hot | metric one-hot], where a
per-arm linear model is already fully expressive (one coefficient per cell). The current input
is continuous -- generation-length bucket, probe softmax and hidden-state PCA -- so the
reduction no longer applies and a nonlinear arm model is worth testing.

NeuralUCB is implemented in its practical Neural-LinUCB form. The full parameter gradient is
about 40k dimensions, which makes Z intractable; with a linear output head the gradient with
respect to that head is exactly the hidden representation phi(x), so phi is used as the UCB
feature:

    f_a(x) = w_a . phi(x) + c_a          phi = last hidden layer of the shared trunk
    Z_a    = lambda I + sum phi phi^T    over rounds that played arm a
    UCB_a  = f_a(x) + beta sqrt(phi^T Z_a^-1 phi)

Each round reveals only the reward of the arm actually played; the trunk is refit from the
accumulated buffer every --refit_every epochs.

The comparison separates two axes that are easy to confuse:

                    linear            nonlinear
  bandit            --algo linucb     --algo neuralucb
  full information  --algo ridge      iclr/mlp_v3.py

  python iclr/ucb_v3.py --model llama3-8b --algo neuralucb --beta 0.1 --device cuda
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '.')
# The heavy dependencies (fast_store, reward, probe paths) are needed only while building the
# cache. Once it exists this module must run on numpy and torch alone, so that it also works on
# a remote GPU host that only has part of the repository synced.
KEYS13 = ["0:1", "0.01:1", "0.01:16", "0.01:32", "0.01:128", "0.1:1", "0.1:16",
          "0.1:32", "0.1:128", "10:1", "10:16", "10:32", "10:128"]

NA = 13


CACHE = os.environ.get("UCB_CACHE", os.environ.get("ICLR_TRACES", "/data2/smp9898/iclr_traces") + "/ucb_cache")


def load_data(a):
    """Row filter, features and rewards identical to mlp_v3, so only the algorithm differs.

    load_store and the hidden-state SVD dominate the runtime, so the result is cached as npz
    per (model, pool, variant, feats, hid_dim, reward); a sweep would otherwise rebuild the
    same arrays dozens of times.
    """
    os.makedirs(CACHE, exist_ok=True)
    cp = f"{CACHE}/{a.model}_{a.pool}_{a.variant}_{a.feats}_h{a.hid_dim}_{a.reward}.npz"
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=True)
        return z["X"], z["R"], z["XL"], [tuple(x) for x in z["ks"]], z["GT"]
    from iclr.mlp_v3 import build_feat, PROBE_ROOT, D2M
    from iclr.proxy_gate import load_store
    from iclr.reward import action_rewards, has_gold
    from iclr.clean_cell_table import true_cls
    PROBE = f"{PROBE_ROOT}/probe_{a.variant}"
    PROBE_H = f"{PROBE_ROOT}/probe_{a.hid_var}"
    HR = HL = None
    if a.hid_dim > 0:
        zhr = np.load(f"{PROBE_H}/{a.model}_recipe.npz")
        zhl = np.load(f"{PROBE_H}/{a.model}_lb.npz")
        HR = {str(zhr["key"][i]): zhr["hid"][i].astype(np.float32) for i in range(len(zhr["key"]))}
        HL = {str(zhl["key"][i]): zhl["hid"][i].astype(np.float32) for i in range(len(zhl["key"]))}
    zr = np.load(f"{PROBE}/{a.model}_recipe.npz")
    P = {str(zr["key"][i]): (zr["prob"][i], zr["surf"][i]) for i in range(len(zr["key"]))}
    rows = []
    for obj in (json.loads(l) for l in open(f"datasets/training/raw/recipe_{a.pool}_{a.model}.jsonl")):
        sid = str(obj["sample_id"])
        if not (obj.get("full_cache_pred") or "").strip(): continue
        if len(obj.get("action_outputs", [])) != NA or sid not in P: continue
        if HR is not None and sid not in HR: continue
        pr, sf = P[sid]
        g_ok = a.reward in ("gold", "mix") and has_gold(obj)
        rows.append((int(obj["generation_length"]),
                     true_cls(obj["dataset"], obj["task_type"], obj.get("metric")), pr, sf,
                     action_rewards(obj, "gold" if g_ok else "p0"),
                     HR[sid] if HR is not None else None))
    R = np.stack([r[4] for r in rows])
    R = R / 100.0 if R.max() > 1.5 else R                      # rewards to [0, 1]
    PCA = None
    if a.hid_dim > 0:                                          # PCA fitted on the recipe only
        Hm = np.stack([r[5] for r in rows]); mu = Hm.mean(0)
        _, _, Vt = np.linalg.svd(Hm - mu, full_matrices=False); W = Vt[:a.hid_dim].T
        PCA = (mu, W, ((Hm - mu) @ W).std(0) + 1e-6)
    def feats(items, hid):
        Xl = [build_feat(b, pr, sf, a.feats) for b, pr, sf in items]
        if PCA is not None:
            Z = ((np.stack(hid) - PCA[0]) @ PCA[1]) / PCA[2]
            Xl = [x + list(z) for x, z in zip(Xl, Z)]
        return np.array(Xl, dtype=np.float32)
    X = feats([(r[0], r[2], r[3]) for r in rows], [r[5] for r in rows] if PCA else None)
    # LongBench, used purely as a test set
    zl = np.load(f"{PROBE}/{a.model}_lb.npz")
    PL = {str(zl["key"][i]): (zl["prob"][i], zl["surf"][i]) for i in range(len(zl["key"]))}
    store = load_store(a.model); gt = {}
    for (ds, i), r in store.items():
        if all(x in r["actions"] for x in KEYS13):
            gt[(ds, i)] = np.array([r["actions"][x]["score"] for x in KEYS13])
    ks = [k for k in sorted(gt) if f"{k[0]}|{k[1]}" in PL]
    if HL is not None: ks = [k for k in ks if f"{k[0]}|{k[1]}" in HL]
    XL = feats([(int(D2M[k[0]]), *PL[f"{k[0]}|{k[1]}"]) for k in ks],
               [HL[f"{k[0]}|{k[1]}"] for k in ks] if PCA else None)
    GT = np.stack([gt[k] for k in ks])
    np.savez(cp, X=X, R=R, XL=XL, ks=np.array(ks, dtype=object), GT=GT)
    return X, R, XL, ks, GT


def trunk(d_in, hidden, depth, seed):
    torch.manual_seed(seed)
    L, d = [], d_in
    for _ in range(depth):
        L += [nn.Linear(d, hidden), nn.ReLU()]; d = hidden
    return nn.Sequential(*L)


def refit(net, head, buf_x, buf_a, buf_r, epochs=120, lr=1e-3, dev="cpu"):  # noqa: D401
    """Fit on observed (x, played arm, reward) only; rewards of unplayed arms are never used."""
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=lr)
    X = torch.tensor(np.array(buf_x, dtype=np.float32), device=dev)
    A = torch.tensor(np.array(buf_a), dtype=torch.long, device=dev)
    Rr = torch.tensor(np.array(buf_r, dtype=np.float32), device=dev)
    n = len(X)
    for _ in range(epochs):
        p = torch.randperm(n, device=X.device)
        for i in range(0, n, 256):
            b = p[i:i + 256]
            pred = head(net(X[b])).gather(1, A[b, None]).squeeze(1)
            loss = ((pred - Rr[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", default="poolz")
    ap.add_argument("--variant", default="v1t64e")
    ap.add_argument("--hid_var", default="v1t64h")
    ap.add_argument("--feats", default="prob")
    ap.add_argument("--hid_dim", type=int, default=16)
    ap.add_argument("--reward", default="p0")
    ap.add_argument("--algo", default="neuralucb", choices=["neuralucb", "linucb", "ridge"])
    ap.add_argument("--beta", type=float, default=0.1, help="exploration coefficient")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=32)
    ap.add_argument("--refit_every", type=int, default=4, help="refit the network every N epochs")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cpu", help="cpu or cuda; moves the refit and forward passes to the GPU")
    ap.add_argument("--cache_only", action="store_true", help="build the data cache and exit")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    X, R, XL, ks, GT = load_data(a)
    if a.cache_only:
        print("  cache built, exiting"); return
    n, d = X.shape
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if dev != a.device: print("  CUDA unavailable, falling back to cpu")
    print(f"  {n} rows, {d} features, {NA} actions, algo={a.algo}")

    sel_seeds = []
    for sd in [int(s) for s in a.seeds.split(",")]:
        rng = np.random.RandomState(sd)
        if a.algo == "ridge":
            # full-information linear control: ridge over all 13 rewards per arm, not a bandit
            Xb = np.hstack([X, np.ones((n, 1))])
            Wm = np.linalg.solve(Xb.T @ Xb + a.lam * np.eye(d + 1), Xb.T @ R)
            XLb = np.hstack([XL, np.ones((len(XL), 1))])
            sel_seeds.append((XLb @ Wm).argmax(1))
            continue
        if a.algo == "linucb":
            Ainv = [np.eye(d) / a.lam for _ in range(NA)]
            b = [np.zeros(d) for _ in range(NA)]
            th = [np.zeros(d) for _ in range(NA)]
            for ep in range(a.epochs):
                for i in rng.permutation(n):
                    p = X[i].astype(np.float64)
                    u = [th[k] @ p + a.beta * np.sqrt(max(p @ Ainv[k] @ p, 0.0)) for k in range(NA)]
                    k = int(np.argmax(u))
                    r = float(R[i, k])                       # only the played arm's reward is observed
                    Ap = Ainv[k] @ p
                    Ainv[k] -= np.outer(Ap, Ap) / (1.0 + p @ Ap)
                    b[k] += r * p; th[k] = Ainv[k] @ b[k]
            sel_seeds.append(np.array([int(np.argmax([th[k] @ x for k in range(NA)])) for x in XL.astype(np.float64)]))
            continue
        # ---- NeuralUCB (Neural-LinUCB: the last-layer gradient is the hidden representation) ----
        net = trunk(d, a.hidden, a.depth, sd).to(dev)
        head = nn.Linear(a.hidden, NA).to(dev)
        with torch.no_grad():
            nn.init.zeros_(head.bias); nn.init.normal_(head.weight, 0, 0.01)
        Zinv = np.stack([np.eye(a.hidden) / a.lam for _ in range(NA)])
        seen = {}          # (row, arm) -> reward; replaying the same pair carries no new information
        for ep in range(a.epochs):
            order = rng.permutation(n)
            with torch.no_grad():
                Phi = net(torch.tensor(X[order], device=dev)).cpu().numpy().astype(np.float64)
                Fv = head(torch.tensor(Phi, dtype=torch.float32, device=dev)).cpu().numpy()
            for j, i in enumerate(order):
                ph = Phi[j]
                bonus = np.sqrt(np.maximum(np.einsum("i,kij,j->k", ph, Zinv, ph), 0.0))
                k = int(np.argmax(Fv[j] + a.beta * bonus))
                seen[(int(i), k)] = float(R[i, k])          # only the played arm's reward is observed
                Zp = Zinv[k] @ ph
                Zinv[k] -= np.outer(Zp, Zp) / (1.0 + ph @ Zp)
            if ep % a.refit_every == 0 or ep == a.epochs - 1:
                idx = np.array([p[0] for p in seen]); arm = np.array([p[1] for p in seen])
                rew = np.array(list(seen.values()))
                refit(net, head, X[idx], arm, rew, epochs=(120 if ep == 0 else 40), dev=dev)
        with torch.no_grad():
            sel_seeds.append(head(net(torch.tensor(XL, device=dev))).cpu().numpy().argmax(1))

    # seed ensemble by majority vote over the chosen action
    S = np.stack(sel_seeds)
    sel = np.array([np.bincount(S[:, j], minlength=NA).argmax() for j in range(S.shape[1])])
    glob = int(R.mean(0).argmax())
    per, pg = {}, {}
    for j, kk in enumerate(ks):
        per.setdefault(kk[0], []).append(GT[j][int(sel[j])])
        pg.setdefault(kk[0], []).append(GT[j][glob])
    mac = lambda dd: float(np.mean([np.mean(v) for v in dd.values()]))
    t2d = json.load(open("config/task2dataset.json")); f2 = {x: t for t, ds in t2d.items() for x in ds}
    fam = {}
    for ds, v in per.items(): fam.setdefault(f2.get(ds, "?"), []).append(float(np.mean(v)))
    fam = {t: float(np.mean(v)) for t, v in fam.items()}
    import collections
    c = collections.Counter(int(x) for x in sel)
    print("  per task: " + "  ".join(f"{t[:12]}={fam[t]:.2f}" for t in sorted(fam)))
    print("  action mix: " + "  ".join(f"{KEYS13[k]}:{v}" for k, v in c.most_common(5)))
    print(f"{a.model:12s} algo={a.algo:9s} beta={a.beta:<5g} LB={mac(per):6.2f} "
          f"(global {mac(pg):.2f}, Δ{mac(per)-mac(pg):+.2f})")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"model": a.model, "algo": a.algo, "beta": a.beta, "epochs": a.epochs,
                   "lb": mac(per), "glob": mac(pg), "fam": fam,
                   "picks": {KEYS13[k]: v for k, v in c.most_common()}},
                  open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
