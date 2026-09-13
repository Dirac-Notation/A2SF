"""Search the router architecture against its own ceiling, by training on LongBench itself.

Every number this script prints is diagnostic. Training on LongBench violates the clean
protocol, so none of it can be claimed as performance; the point is to learn how much accuracy
a given feature set and architecture could reach at all. Once an architecture is fixed here,
it is frozen and the training corpus is rebuilt without LongBench to match that regime.

Two protocols are reported side by side:

  rand5  random 5-fold over samples. Other samples of the same dataset appear in training, so
         this bounds dataset-level routing, which is reachable at deployment whenever the
         features identify the dataset.
  lodo   leave-one-dataset-out. Tests generalization to an unseen task family; stricter, and
         closer to deployment.

  python iclr/struct_search.py --model llama3-8b --stage C
"""
import argparse
import itertools
import json
import math
import os
import sys

import numpy as np
import torch

torch.set_num_threads(6)
import torch.nn as nn
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from iclr.proxy_gate import load_store  # noqa: E402
from iclr.recipe_table import KEYS13  # noqa: E402

D2M = json.load(open("config/dataset2maxlen.json"))
TRACES = os.environ.get("ICLR_TRACES", "/data2/smp9898/iclr_traces")
PROBE = f"{TRACES}/probe_v1t64"
PROBE_H = f"{TRACES}/probe_v1t64h"


def len_feat(g, enc):
    b = [0.] * 3
    b[0 if g <= 32 else (1 if g <= 128 else 2)] = 1.
    sc = [math.log(max(g, 1)) / 6.5]
    if enc == "bucket":
        return b
    if enc == "scalar":
        return sc
    if enc == "both":
        return b + sc
    if enc == "fine":
        f = [0.] * 4
        f[0 if g <= 32 else (1 if g <= 64 else (2 if g <= 128 else 3))] = 1.
        return f
    raise ValueError(enc)


def load(model):
    z = np.load(f"{PROBE}/{model}_lb.npz")
    P = {str(z["key"][i]): (z["prob"][i], z["surf"][i]) for i in range(len(z["key"]))}
    # last hidden state of the same probe forward, at no extra cost
    H = {}
    hp = f"{PROBE_H}/{model}_lb.npz"
    if os.path.exists(hp):
        zh = np.load(hp)
        if "hid" in zh.files:
            H = {str(zh["key"][i]): zh["hid"][i].astype(np.float32) for i in range(len(zh["key"]))}
    S = load_store(model)
    ds_l, A, prob, surf, gen, hid = [], [], [], [], [], []
    for (d, i), r in S.items():
        k = f"{d}|{i}"
        if k not in P or not all(x in r["actions"] for x in KEYS13):
            continue
        if H and k not in H:
            continue
        ds_l.append(d); A.append([r["actions"][x]["score"] for x in KEYS13])
        prob.append(P[k][0]); surf.append(P[k][1]); gen.append(D2M[d])
        if H:
            hid.append(H[k])
    return (np.array(ds_l), np.array(A, dtype=np.float64),
            np.array(prob, dtype=np.float32), np.array(surf, dtype=np.float32),
            np.array(gen), np.array(hid, dtype=np.float32) if hid else None)


def base_feats(enc, fs, prob, surf, gen):
    out = [np.array([len_feat(g, enc) for g in gen], dtype=np.float32)]
    if "prob" in fs:
        out.append(prob)
    if "surf" in fs:
        out.append(surf)
    return np.hstack(out).astype(np.float32)


def hid_dim(fs):
    """Return K when fs contains 'hidK', else 0."""
    for tok_ in fs.split("+"):
        if tok_.startswith("hid"):
            return int(tok_[3:])
    return 0


def add_hid(Xb, hid, tr, te, k):
    """Fit the PCA on the training fold only, to avoid leakage."""
    if k == 0 or hid is None:
        return torch.tensor(Xb, dtype=torch.float32)
    mu = hid[tr].mean(0)
    U, S_, Vt = np.linalg.svd(hid[tr] - mu, full_matrices=False)
    W = Vt[:k].T
    Z = (hid - mu) @ W
    Z = Z / (Z[tr].std(0) + 1e-6)
    return torch.tensor(np.hstack([Xb, Z.astype(np.float32)]), dtype=torch.float32)


def make_target(A, mu, kind):
    Y = A[:, mu]
    if kind == "rank":
        return torch.tensor(np.stack([rankdata(y) / len(mu) for y in Y]), dtype=torch.float32)
    if kind == "raw":
        return torch.tensor(Y / 100.0, dtype=torch.float32)
    if kind == "z":
        s = Y.std(1, keepdims=True); s[s < 1e-6] = 1.0
        return torch.tensor((Y - Y.mean(1, keepdims=True)) / s, dtype=torch.float32)
    if kind == "top1":
        T = np.zeros_like(Y); T[np.arange(len(Y)), Y.argmax(1)] = 1.0
        return torch.tensor(T, dtype=torch.float32)
    raise ValueError(kind)


def fit(X, Y, loss, hid, depth, sd, epochs=250, patience=25):
    torch.manual_seed(sd)
    layers, d = [], X.shape[1]
    for _ in range(depth):
        layers += [nn.Linear(d, hid), nn.ReLU()]; d = hid
    layers += [nn.Linear(d, Y.shape[1])]
    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(sd)
    n = len(X); idx = np.random.RandomState(sd).permutation(n); nv = max(64, n // 10)
    va, tr = idx[:nv], idx[nv:]
    best, bs, pat = 1e9, None, 0
    for _ in range(epochs):
        net.train()
        p = tr[torch.randperm(len(tr), generator=g).numpy()]
        for i in range(0, len(p), 256):
            b = p[i:i + 256]
            o = net(X[b])
            l = ((o - Y[b]) ** 2).mean() if loss == "mse" else \
                -(Y[b] * torch.log_softmax(o, -1)).sum(-1).mean()
            opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            o = net(X[va])
            vl = float(((o - Y[va]) ** 2).mean() if loss == "mse" else
                       -(Y[va] * torch.log_softmax(o, -1)).sum(-1).mean())
        if vl < best - 1e-6:
            best, bs, pat = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= patience:
                break
    net.load_state_dict(bs); net.eval()
    return net


def evaluate(ds_l, A, Xb, hidmat, kdim, mu, tgt, loss, hidn, depth, seeds, proto):
    """proto is 'rand5' or 'lodo'; returns the dataset-macro score."""
    Y = make_target(A, mu, tgt)
    sel = np.zeros(len(A), dtype=int)
    if proto == "rand5":
        rng = np.random.RandomState(0); idx = rng.permutation(len(A))
        folds = [idx[f::5] for f in range(5)]
    else:
        folds = [np.where(ds_l == d)[0] for d in sorted(set(ds_l))]
    for te in folds:
        tr = np.setdiff1d(np.arange(len(A)), te)
        X = add_hid(Xb, hidmat, tr, te, kdim)
        nets = [fit(X[tr], Y[tr], loss, hidn, depth, s) for s in seeds]
        with torch.no_grad():
            sel[te] = (sum(n(X[te]) for n in nets) / len(nets)).argmax(1).numpy()
    per = {}
    for j, d in enumerate(ds_l):
        per.setdefault(d, []).append(A[j, mu[sel[j]]])
    return float(np.mean([np.mean(v) for v in per.values()]))


def greedy_menu(A, tr, ds_l, k):
    """Pick the k-action menu by greedy cell-oracle inside the training fold, to avoid leakage."""
    cells = {}
    for j in tr:
        cells.setdefault(ds_l[j], []).append(A[j])
    Cm = {c: np.stack(v).mean(0) for c, v in cells.items()}
    W = {c: len(v) for c, v in cells.items()}
    tot = sum(W.values())
    menu = []
    for _ in range(k):
        best, bi = None, None
        for x in range(A.shape[1]):
            if x in menu:
                continue
            v = sum(W[c] * max(Cm[c][y] for y in menu + [x]) for c in Cm) / tot
            if best is None or v > best:
                best, bi = v, x
        menu.append(bi)
    return sorted(menu)


def evaluate_k(ds_l, A, Xb, hidmat, kdim, k, tgt, loss, hidn, depth, seeds):
    """rand5 evaluation with the menu size chosen inside each fold."""
    sel = np.zeros(len(A), dtype=int); mus = {}
    rng = np.random.RandomState(0); idx = rng.permutation(len(A))
    for f in range(5):
        te = idx[f::5]; tr = np.setdiff1d(np.arange(len(A)), te)
        mu = greedy_menu(A, tr, ds_l, k) if k < A.shape[1] else list(range(A.shape[1]))
        Y = make_target(A, mu, tgt)
        X = add_hid(Xb, hidmat, tr, te, kdim)
        nets = [fit(X[tr], Y[tr], loss, hidn, depth, s) for s in seeds]
        with torch.no_grad():
            p = (sum(n(X[te]) for n in nets) / len(nets)).argmax(1).numpy()
        for a_, j in enumerate(te):
            sel[j] = mu[p[a_]]
    per = {}
    for j, d in enumerate(ds_l):
        per.setdefault(d, []).append(A[j, sel[j]])
    return float(np.mean([np.mean(v) for v in per.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", default="A")
    ap.add_argument("--proto", default="rand5", choices=["rand5", "lodo", "both"])
    ap.add_argument("--feats", default="prob+hid16")
    ap.add_argument("--enc", default="bucket")
    ap.add_argument("--tgt", default="raw")
    ap.add_argument("--loss", default="ce")
    a = ap.parse_args()
    ds_l, A, prob, surf, gen, hidmat = load(a.model)
    mu13 = list(range(13))
    # baselines
    per = {}
    for j, d in enumerate(ds_l):
        per.setdefault(d, []).append(A[j])
    G = {d: np.array(v) for d, v in per.items()}
    gi = int(np.mean([g.mean(0) for g in G.values()], axis=0).argmax())
    glob = float(np.mean([g[:, gi].mean() for g in G.values()]))
    dso = float(np.mean([g.mean(0).max() for g in G.values()]))
    ppo = float(np.mean([g.max(1).mean() for g in G.values()]))
    print(f"=== {a.model}  baselines: global {glob:.2f} | dataset oracle {dso:.2f} | prompt oracle {ppo:.2f}")

    protos = ["rand5", "lodo"] if a.proto == "both" else [a.proto]
    if a.stage == "A":
        FEATS = ["prob", "prob+surf", "hid16", "hid32", "prob+hid16", "prob+hid32",
                 "prob+surf+hid32", "prob+hid64"]
        grid = list(itertools.product(
            ["bucket", "fine"],                            # generation-length encoding
            FEATS,                                          # input features
            ["rank", "raw"],                               # target (z and top1 measured worse on llama)
            ["mse", "ce"]))                                # loss
        res = []
        for enc, fs, tgt, loss in grid:
            Xb = base_feats(enc, fs, prob, surf, gen)
            k = hid_dim(fs)
            if k and hidmat is None:
                continue
            row = {"enc": enc, "feats": fs, "tgt": tgt, "loss": loss}
            for p in protos:
                row[p] = evaluate(ds_l, A, Xb, hidmat, k, mu13, tgt, loss, 64, 2, (0,), p)
            res.append(row)
            print(f"  {enc:7s} {fs:16s} {tgt:5s} {loss:4s}  " +
                  "  ".join(f"{p}={row[p]:6.2f}" for p in protos), flush=True)
        res.sort(key=lambda r: -r[protos[0]])
        os.makedirs("result_txt/analysis/struct", exist_ok=True)
        json.dump({"model": a.model, "glob": glob, "dso": dso, "ppo": ppo, "res": res},
                  open(f"result_txt/analysis/struct/stageA_{a.model}.json", "w"), indent=1)
        print(f"\nbest: {res[0]}")
    elif a.stage == "C":
        # Stage C: freeze the action set to the full 13-grid and settle the rest of the design.
        # Picking a different architecture per model would be ad hoc, so selection is by the
        # mean over the three models; this script emits per-model values and the mean is taken
        # outside.
        res = []
        for fs, tl, hidn, depth in itertools.product(
                ["prob", "prob+surf", "prob+hid16"],
                [("rank", "ce"), ("raw", "ce")],
                [64, 128], [2, 3]):
            tgt, loss = tl
            Xb = base_feats("bucket", fs, prob, surf, gen)
            v = evaluate(ds_l, A, Xb, hidmat, hid_dim(fs), mu13, tgt, loss, hidn, depth, (0, 1, 2), "rand5")
            res.append({"feats": fs, "tgt": tgt, "loss": loss, "hid": hidn, "depth": depth, "rand5": v})
            print(f"  {fs:12s} {tgt:5s}/{loss:3s} h={hidn:3d} d={depth}  rand5={v:6.2f}", flush=True)
        res.sort(key=lambda r: -r["rand5"])
        json.dump({"model": a.model, "glob": glob, "dso": dso, "ppo": ppo, "res": res},
                  open(f"result_txt/analysis/struct/stageC_{a.model}.json", "w"), indent=1)
        print(f"\nbest: {res[0]}")
    else:  # Stage B: capacity, menu size and ensembling on top of the Stage A winner
        fs = a.feats; kdim = hid_dim(fs)
        Xb = base_feats(a.enc, fs, prob, surf, gen)
        res = []
        for hidn, depth, k, ns in itertools.product([32, 64, 128], [2, 3], [5, 8, 13], [1, 3]):
            v = evaluate_k(ds_l, A, Xb, hidmat, kdim, k, a.tgt, a.loss, hidn, depth, tuple(range(ns)))
            res.append({"hid": hidn, "depth": depth, "k": k, "seeds": ns, "rand5": v})
            print(f"  h={hidn:3d} d={depth} k={k:2d} seeds={ns}  rand5={v:6.2f}", flush=True)
        res.sort(key=lambda r: -r["rand5"])
        json.dump({"model": a.model, "feats": fs, "enc": a.enc, "tgt": a.tgt, "loss": a.loss,
                   "glob": glob, "dso": dso, "ppo": ppo, "res": res},
                  open(f"result_txt/analysis/struct/stageB_{a.model}.json", "w"), indent=1)
        print(f"\nbest: {res[0]}")


if __name__ == "__main__":
    main()
