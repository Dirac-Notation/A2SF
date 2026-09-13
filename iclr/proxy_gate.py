"""Label-free proxy screening: does a per-prompt proxy carry usable action signal?

Runs without a GPU by reusing the cached per-action outputs.

  Spread   are the per-prompt optimal actions actually spread out within a dataset?
  Gate 1   synonym insensitivity: variance inside the four a=0.01 actions (kept-token
           distance 0.000) over total variance
  Gate 2   rank agreement: per-sample Spearman(proxy, GT), and how often argmax-proxy lands
           in the GT top-tie set
  Gate 3   ceiling: LongBench score of per-sample argmax-proxy routing (macro over datasets)

Proxies: P3 = -mean over (layer, head) of the future-attention AUC cost ratio;
P0 = Tanimoto(action prediction tokens, full-cache prediction tokens).
GT = the official fast_store score.
"""
import argparse
import glob
import gzip
import json
import os
import re
import sys

TRACES = os.environ.get("ICLR_TRACES", "/data2/smp9898/iclr_traces")

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

A_VALUES = [0.01, 0.1, 1.0, 10.0]
B_VALUES = [1.0, 16.0, 32.0, 128.0]
CANDIDATES = [(0.0, 1.0)] + [(a, b) for a in A_VALUES for b in B_VALUES]  # 17
KEYS = [f"{a:g}:{b:g}" for a, b in CANDIDATES]
SYN_IDX = [KEYS.index(k) for k in ["0.01:1", "0.01:16", "0.01:32", "0.01:128"]]


def tanimoto(a, b):
    A = set(re.findall(r"\w+", a.lower()))
    B = set(re.findall(r"\w+", b.lower()))
    if not A and not B:
        return 1.0
    return len(A & B) / max(len(A | B), 1)


def load_store(model):
    store = {}
    with gzip.open(f"result_txt/backup/fast_store/store_{model}_128.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            store[(r["dataset"], r["idx"])] = r
    return store


def load_p3(model):
    """(ds, idx) -> proxy vector [17] (higher better). File position i maps to the jsonl idx."""
    out = {}
    for d in sorted(glob.glob(f"{TRACES}/lb_pass1/{model}/*/")):
        ds = os.path.basename(d.rstrip("/"))
        files = sorted(glob.glob(os.path.join(d, "s*_rewards.npz")))
        if not files:
            continue
        idxs = [json.loads(l)["idx"] for l in open(f"datasets/longbench/{ds}.jsonl")]
        for f in files:
            i = int(os.path.basename(f)[1:4])
            ratio = np.load(f)["ratio"]              # [17, L, H], lower better
            out[(ds, idxs[i])] = -ratio.reshape(17, -1).mean(1)
    return out


def gate1(vec_by_sample):
    """Within-synonym-group std over total std, averaged per sample."""
    ratios = []
    for v in vec_by_sample:
        tot = np.std(v)
        if tot < 1e-12:
            continue
        ratios.append(np.std(v[SYN_IDX]) / tot)
    return float(np.mean(ratios)) if ratios else float("nan")


def gate2(pairs):
    """pairs: list of (proxy[17], gt[17]). Returns Spearman over non-ties, top-set hit rate and the random baseline."""
    rhos, hits, rand = [], [], []
    for p, g in pairs:
        top = set(np.where(g == g.max())[0])
        hits.append(int(np.argmax(p)) in top)
        rand.append(len(top) / 17)
        if len(np.unique(g)) > 1:
            r = spearmanr(p, g).correlation
            if np.isfinite(r):
                rhos.append(r)
    return (float(np.mean(rhos)) if rhos else float("nan"), len(rhos),
            float(np.mean(hits)), float(np.mean(rand)))


def lb_mean(per_ds_scores):
    return float(np.mean([np.mean(v) for v in per_ds_scores.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3-1b")
    args = ap.parse_args()

    store = load_store(args.model)
    p3 = load_p3(args.model)
    datasets = sorted({ds for ds, _ in store})

    # ---- build per-sample vectors ----
    gt_all, p0_all = {}, {}                    # (ds, idx) -> [17]
    full_score = {}
    for (ds, idx), r in store.items():
        acts = r["actions"]
        if any(k not in acts for k in KEYS):
            continue
        gt_all[(ds, idx)] = np.array([acts[k]["score"] for k in KEYS])
        fp = acts["full"]["pred"]
        p0_all[(ds, idx)] = np.array([tanimoto(acts[k]["pred"], fp) for k in KEYS])
        full_score[(ds, idx)] = acts["full"]["score"]

    sub = sorted(set(p3) & set(gt_all))        # subset that has P3
    print(f"[pgate] {args.model}: store {len(gt_all)} samples / {len(datasets)} ds, "
          f"P3 subset {len(sub)} samples / {len({d for d, _ in sub})} ds")

    res = {"model": args.model, "n_store": len(gt_all), "n_p3": len(sub)}

    # ---- check 1: spread of the GT-optimal action ----
    var = {}
    for ds in datasets:
        keys = [k for k in gt_all if k[0] == ds]
        tops = [set(np.where(gt_all[k] == gt_all[k].max())[0]) for k in keys]
        cover = max(np.mean([a in t for t in tops]) for a in range(17))
        var[ds] = {"n": len(keys),
                   "mean_topset": float(np.mean([len(t) for t in tops])),
                   "all_tied_frac": float(np.mean([len(t) == 17 for t in tops])),
                   "best_fixed_cover": float(cover)}
    res["variance_gt"] = var
    print("\n== check 1: spread of the GT-optimal action, per dataset ==")
    print(f"{'dataset':24s} {'n':>4s} {'tie set':>8s} {'all-tie%':>8s} {'fixed cover%':>12s}")
    for ds, v in var.items():
        print(f"{ds:24s} {v['n']:4d} {v['mean_topset']:8.1f} "
              f"{100*v['all_tied_frac']:7.1f} {100*v['best_fixed_cover']:10.1f}")

    # ---- check 2: the three proxy gates ----
    print("\n== check 2: the three proxy gates ==")
    rows = {}
    for name, vec, keys in [("P0(text-fid)", p0_all, sorted(gt_all)),
                            ("P0 (P3 subset)", p0_all, sub),
                            ("P3(fut-att)", p3, sub)]:
        g1 = gate1([vec[k] for k in keys])
        rho, n_nt, hit, rand = gate2([(vec[k], gt_all[k]) for k in keys])
        # gate 3: argmax-proxy routing
        per_ds = {}
        for k in keys:
            per_ds.setdefault(k[0], []).append(gt_all[k][int(np.argmax(vec[k]))])
        g3 = lb_mean(per_ds)
        rows[name] = {"gate1_syn_ratio": g1, "gate2_rho": rho, "gate2_n": n_nt,
                      "gate2_hit": hit, "gate2_rand": rand, "gate3_lb": g3}
        print(f"{name:14s} gate1 syn-var ratio={g1:.3f}  gate2 rho={rho:+.3f} (n={n_nt}) "
              f"hit={100*hit:.1f}% (random {100*rand:.1f}%)  gate3 LB={g3:.2f}")
    # GT's own synonym jitter, i.e. the noise floor
    rows["GT (self)"] = {"gate1_syn_ratio": gate1([gt_all[k] for k in sorted(gt_all)])}
    print(f"{'GT (self)':14s} gate1 syn-var ratio={rows['GT (self)']['gate1_syn_ratio']:.3f}")
    res["gates"] = rows

    # ---- baselines for the gate-3 comparison ----
    def ref(keys):
        per_ds_best, per_ds_or, per_ds_full = {}, {}, {}
        ds_set = sorted({d for d, _ in keys})
        for ds in ds_set:
            kk = [k for k in keys if k[0] == ds]
            mat = np.stack([gt_all[k] for k in kk])          # [n, 17]
            per_ds_best[ds] = [mat[:, int(np.argmax(mat.mean(0)))].mean()]
            per_ds_or[ds] = [mat.max(1).mean()]
            per_ds_full[ds] = [np.mean([full_score[k] for k in kk])]
        return lb_mean(per_ds_best), lb_mean(per_ds_or), lb_mean(per_ds_full)

    for tag, keys in [("all samples", sorted(gt_all)), ("P3 subset", sub)]:
        b, o, fl = ref(keys)
        res[f"ref_{tag}"] = {"dataset_best_fixed": b, "gt_oracle": o, "full": fl}
        print(f"\n[baseline {tag}] dataset-best-fixed={b:.2f}  GT-oracle (incl. spurious)={o:.2f}  full={fl:.2f}")

    os.makedirs("result_txt/analysis/proxy_gate", exist_ok=True)
    out = f"result_txt/analysis/proxy_gate/pgate_{args.model}.json"
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"\n[pgate] saved -> {out}")


if __name__ == "__main__":
    main()
