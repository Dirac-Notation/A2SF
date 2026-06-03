"""For each prompt in obs2 data, decide which sigmoid-family method (TOVA / SnapKV / H2O)
gives the best per-key Jaccard against the decode-attention top-B "answer set",
then characterize the winning groups.

Mapping (a→∞ family is the sigmoid-band step at b = window size):
  TOVA      = step at b=1     -> br[:, 0]
  SnapKV-16 = step at b=16    -> br[:, 15]
  SnapKV-32 = step at b=32    -> br[:, 31]
  H2O       = a→0 (uniform)   -> br[:, W-1]   (all queries equal weight; with W=256 max)

Per-prompt features collected:
  - dataset, task, seq_len, br vector, winner method, margin over runner-up
"""
import os, sys, json, argparse
import numpy as np
from collections import defaultdict, Counter

WORKPATH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(WORKPATH))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--budget",  type=int, default=128)
    p.add_argument("--datasets", default="samsum,qasper,hotpotqa,gov_report")
    p.add_argument("--print_n", type=int, default=3,
                   help="show top-N prompt previews per winner group")
    return p.parse_args()


METHODS = {"TOVA": 0, "SnapKV-16": 15, "SnapKV-32": 31, "H2O": -1}
TASK_OF = {
    "samsum": "Few_Shot",
    "qasper": "Single-doc_QA",
    "hotpotqa": "Multi-doc_QA",
    "gov_report": "Summarization",
}


def main():
    args = parse_args()
    plot_dir = os.path.join(WORKPATH, "plots")
    if args.budget != 128:
        plot_dir = os.path.join(plot_dir, f"b{args.budget}")

    rows = []  # one row per prompt
    for ds in args.datasets.split(","):
        ds = ds.strip()
        ds_dir = os.path.join(plot_dir, TASK_OF[ds], ds)
        m_path = os.path.join(ds_dir, "metrics.npz")
        if not os.path.isfile(m_path):
            print(f"miss: {m_path}")
            continue
        m = np.load(m_path)
        br = m["br"]                 # (N, W)
        seq_len = m["seq_len"]       # (N,)
        sample_idx = m["sample_idx"] # (N,)
        N, W = br.shape

        # method scores
        scores = {}
        for name, idx in METHODS.items():
            ix = idx if idx >= 0 else W - 1
            scores[name] = br[:, ix]

        # winner
        for n in range(N):
            vec = {k: scores[k][n] for k in METHODS}
            ordered = sorted(vec.items(), key=lambda kv: -kv[1])
            winner, w_score = ordered[0]
            runner, r_score = ordered[1]
            margin = w_score - r_score
            rows.append({
                "dataset": ds,
                "task": TASK_OF[ds],
                "seq_len": int(seq_len[n]),
                "sample_idx": int(sample_idx[n]),
                "br_vec": br[n],
                "TOVA":      float(scores["TOVA"][n]),
                "SnapKV-16": float(scores["SnapKV-16"][n]),
                "SnapKV-32": float(scores["SnapKV-32"][n]),
                "H2O":       float(scores["H2O"][n]),
                "winner": winner,
                "margin": float(margin),
            })

    print(f"\nbudget = {args.budget}    total prompts = {len(rows)}")
    print(f"window  W = {br.shape[1]}\n")

    # Overall mean per method
    print("== mean br per method (over all prompts) ==")
    for name in METHODS:
        vals = [r[name] for r in rows]
        print(f"  {name:10s}  mean={np.mean(vals):.3f}   median={np.median(vals):.3f}")

    # Winner counts
    counts = Counter(r["winner"] for r in rows)
    print("\n== winner counts ==")
    for k in METHODS:
        c = counts.get(k, 0)
        print(f"  {k:10s}  {c:4d}  ({100*c/len(rows):.1f}%)")

    # Winner × dataset
    print("\n== winner × dataset (count) ==")
    by_ds = defaultdict(Counter)
    for r in rows:
        by_ds[r["dataset"]][r["winner"]] += 1
    head = ["dataset"] + list(METHODS.keys())
    print("  " + "  ".join(f"{h:>10s}" for h in head))
    for ds in sorted(by_ds):
        line = [ds] + [str(by_ds[ds].get(k, 0)) for k in METHODS]
        print("  " + "  ".join(f"{x:>10s}" for x in line))

    # Winner × seq_len bucket
    print("\n== winner × seq_len bucket ==")
    buckets = [(0, 2500), (2500, 4000), (4000, 5500), (5500, 99999)]
    bk_counts = defaultdict(Counter)
    for r in rows:
        for lo, hi in buckets:
            if lo <= r["seq_len"] < hi:
                bk_counts[(lo, hi)][r["winner"]] += 1
                break
    head = ["seq_len"] + list(METHODS.keys()) + ["N"]
    print("  " + "  ".join(f"{h:>12s}" for h in head))
    for lo, hi in buckets:
        n = sum(bk_counts[(lo, hi)].values())
        line = [f"{lo}-{hi}"] + [str(bk_counts[(lo, hi)].get(k, 0)) for k in METHODS] + [str(n)]
        print("  " + "  ".join(f"{x:>12s}" for x in line))

    # Margin stats per winner
    print("\n== avg margin (winner - runner-up) per group ==")
    for k in METHODS:
        margs = [r["margin"] for r in rows if r["winner"] == k]
        if margs:
            print(f"  {k:10s}  mean_margin={np.mean(margs):.3f}   max={np.max(margs):.3f}   n={len(margs)}")

    # br shape per group: where does br peak on average?
    print("\n== mean br[d] curve, per winner group ==")
    Wmax = rows[0]["br_vec"].shape[0]
    print(f"  d:  {'  '.join(f'{d:5d}' for d in [1,4,16,32,64,128,192,Wmax])}")
    for k in METHODS:
        sub = [r["br_vec"] for r in rows if r["winner"] == k]
        if not sub: continue
        mean = np.stack(sub, axis=0).mean(axis=0)
        sample_ds = [1, 4, 16, 32, 64, 128, 192, Wmax]
        sample_vals = [mean[d-1] for d in sample_ds]
        print(f"  {k:10s}  {'  '.join(f'{v:5.3f}' for v in sample_vals)}")

    # Show example prompts (largest margin) per winner
    print("\n== top-margin prompt previews per winner ==")
    longbench_dir = os.path.join(ROOT, "datasets", "longbench")
    pool = defaultdict(list)
    for fname in os.listdir(longbench_dir):
        with open(os.path.join(longbench_dir, fname)) as f:
            for line in f:
                it = json.loads(line)
                pool[it["dataset"]].append(it.get("input_prompt", ""))

    # NOTE: parallel_obs2 shuffles with random.Random(SEED) and slices first n_items.
    # For preview alone we just print dataset, seq_len, margin, and a tag.
    for k in METHODS:
        sub = sorted([r for r in rows if r["winner"] == k], key=lambda r: -r["margin"])[:args.print_n]
        if not sub: continue
        print(f"\n  [{k}] top by margin:")
        for r in sub:
            print(f"    - {r['dataset']:11s}  L={r['seq_len']:5d}  "
                  f"TOVA={r['TOVA']:.3f}  SnapKV-16={r['SnapKV-16']:.3f}  "
                  f"SnapKV-32={r['SnapKV-32']:.3f}  H2O={r['H2O']:.3f}  margin={r['margin']:+.3f}")


if __name__ == "__main__":
    main()
