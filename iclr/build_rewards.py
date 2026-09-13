"""Step 2: offline reward matrix from trace dumps.

For each (doc, layer, head, action): KVP-style all-budget AUC cost of the ranking
induced by that action's scores, normalized by the oracle ranking's cost.
  cost(sigma) = sum_b rank_b * u_{sigma_b};  ratio = cost(sigma_c) / cost(sigma*)
ratio >= 1, lower is better. reward = -ratio.

Ties are broken randomly (seeded per doc) - grid-order artifacts otherwise decide.

Output: <dump_dir>/rewards.npz  { ratio: [D, C, L, H] f32, doc_ids: [D] }
Usage:  python iclr/build_rewards.py --dump_dir $ICLR_TRACES/llama3-1b
"""
import argparse
import glob
import json
import os

import numpy as np


def auc_cost(order, u):
    """order: [N] indices best->worst; u: [N]. cost = sum (rank+1) * u[order[rank]]"""
    ranks = np.arange(1, len(order) + 1, dtype=np.float64)
    return float((ranks * u[order].astype(np.float64)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.dump_dir, "meta.json")))
    files = sorted(glob.glob(os.path.join(args.dump_dir, "doc_*.npz")))
    print(f"[rewards] {len(files)} docs")

    ratios, doc_ids = [], []
    for f in files:
        z = np.load(f)
        u = z["u"].astype(np.float32)        # [L, H, N]
        cand = z["cand"].astype(np.float32)  # [C, L, H, N]
        C, L, H, N = cand.shape
        d = int(os.path.basename(f)[4:8])
        rng = np.random.RandomState(args.seed * 100003 + d)
        noise = rng.uniform(0, 1e-6, size=N).astype(np.float32)
        r = np.zeros((C, L, H), dtype=np.float32)
        for l in range(L):
            for h in range(H):
                uv = u[l, h]
                oracle = np.argsort(-(uv + noise), kind="stable")
                c_star = auc_cost(oracle, uv)
                if c_star <= 0:
                    r[:, l, h] = 1.0
                    continue
                for c in range(C):
                    order = np.argsort(-(cand[c, l, h] + noise), kind="stable")
                    r[c, l, h] = auc_cost(order, uv) / c_star
        ratios.append(r)
        doc_ids.append(d)
        if len(doc_ids) % 25 == 0:
            print(f"[rewards] {len(doc_ids)}/{len(files)}", flush=True)

    out = os.path.join(args.dump_dir, "rewards.npz")
    np.savez_compressed(out, ratio=np.stack(ratios), doc_ids=np.array(doc_ids),
                        candidates=np.array(meta["candidates"]))
    print(f"[rewards] saved -> {out}")


if __name__ == "__main__":
    main()
