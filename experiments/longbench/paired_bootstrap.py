"""Paired bootstrap CI of (run A - run B) on LongBench: samples joined by (answers, length), resampled within each
dataset, averaged equally over datasets (4,000 resamples, seed 0).

    python experiments/longbench/paired_bootstrap.py result_txt/pred/128/<qtrust_run> result_txt/pred/128/<snapkv_run>
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import longbench_eval as E


def load(d):
    out = {}
    for f in glob.glob(f"{d}/*.jsonl"):
        ds = os.path.basename(f)[:-6]
        seen = defaultdict(int); rows = {}
        for l in open(f):
            r = json.loads(l)
            k = (json.dumps(r["answers"], ensure_ascii=False), r.get("length"))
            seen[k] += 1; rows[k + (seen[k],)] = r
        out[ds] = rows
    return out


def score(ds, r):
    p = r["pred"]
    if ds in ("trec", "triviaqa", "samsum", "lsht"):
        p = p.lstrip("\n").split("\n")[0]
    return 100 * max(E.dataset2metric[ds](p, g, all_classes=r.get("all_classes")) for g in r["answers"])


def main(a, b):
    rng = np.random.default_rng(0)
    A, B = load(a), load(b)
    diffs, ma, mb = {}, {}, {}
    for ds in sorted(set(A) & set(B)):
        ks = sorted(set(A[ds]) & set(B[ds]))
        xa = np.array([score(ds, A[ds][k]) for k in ks]); xb = np.array([score(ds, B[ds][k]) for k in ks])
        diffs[ds] = xa - xb; ma[ds] = xa.mean(); mb[ds] = xb.mean()
    n = sum(len(v) for v in diffs.values())
    point = np.mean([v.mean() for v in diffs.values()])
    bs = [np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in diffs.values()]) for _ in range(4000)]
    lo, hi = np.quantile(bs, [.025, .975])
    print(f"datasets {len(diffs)} samples {n}  A {np.mean(list(ma.values())):.2f}  B {np.mean(list(mb.values())):.2f}  "
          f"A-B {point:+.2f} [{lo:+.2f}, {hi:+.2f}]  P(A-B<=0) {np.mean(np.array(bs) <= 0):.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
