"""Recipe-P0 lookup table -> LongBench transfer (clean protocol: LB stays a pure test set).

Table: (length bucket x A1 class) -> one of the 13 grid actions, scored by
P0 = Tanimoto(action_output, full_cache_pred).
Buckets: S (<=32) / M (33-128) / L (>128), so LB's 128-token QA lands in M and its 512-token
summarization in L.
Variants: V1 = plain P0 argmax, V2 = abstain (A1 in {D, E} falls back to the global action).
Baselines: recipe-global-fixed (label-free), LB in-menu oracle-fixed and LB-fold A0xA1, both
diagnostic only.

  python iclr/recipe_table.py --model llama3-1b \
      --recipe datasets/training/raw/recipe_v3_1b/train.jsonl
"""
import argparse
import json
import os
import sys

TRACES = os.environ.get("ICLR_TRACES", "/data2/smp9898/iclr_traces")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from iclr.proxy_gate import tanimoto, load_store

GRID13 = [(0.0, 1.0)] + [(a, b) for a in [0.01, 0.1, 10.0] for b in [1.0, 16.0, 32.0, 128.0]]
KEYS13 = [f"{a:g}:{b:g}" for a, b in GRID13]
MIN_CELL = 8


def bucket(g):
    return "S" if g <= 32 else ("M" if g <= 128 else "L")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--a1", default=None, help="path to the a1_recipe npz (default a1_recipe/<model>.npz)")
    args = ap.parse_args()

    # ---- recipe P0 and table fitting ----
    za = np.load(args.a1 or f"{TRACES}/a1_recipe/{args.model}.npz")
    a1_recipe = {str(za["idx"][i]): int(za["cls"][i]) for i in range(len(za["idx"]))}
    cells, glob_p0 = {}, []
    n_used = 0
    for obj in (json.loads(l) for l in open(args.recipe)):
        fc = (obj.get("full_cache_pred") or "")
        if not fc.strip() or str(obj["sample_id"]) not in a1_recipe:
            continue
        p0 = np.array([tanimoto(o or "", fc) for o in obj["action_outputs"]])
        if len(p0) != 13:
            continue
        cell = (bucket(obj["generation_length"]), a1_recipe[str(obj["sample_id"])])
        cells.setdefault(cell, []).append(p0)
        glob_p0.append(p0)
        n_used += 1
    g_star = int(np.argmax(np.stack(glob_p0).mean(0)))
    table = {}
    for cell, v in cells.items():
        table[cell] = int(np.argmax(np.stack(v).mean(0))) if len(v) >= MIN_CELL else g_star
    print(f"[rt] {args.model}: recipe {n_used} samples, {len(cells)} cells, global={KEYS13[g_star]}")
    for cell in sorted(table):
        print(f"  {cell[0]}x{'ABCDE'[cell[1]]}: {KEYS13[table[cell]]:9s} (n={len(cells[cell])})")

    # ---- LongBench transfer (pure test) ----
    store = load_store(args.model)
    gt = {}
    for (ds, idx), r in store.items():
        acts = r["actions"]
        if any(k not in acts for k in KEYS13):
            continue
        gt[(ds, idx)] = np.array([acts[k]["score"] for k in KEYS13])
    zl = np.load(f"{TRACES}/a1/{args.model}.npz")
    a1_lb = {(str(zl["ds"][i]), int(zl["idx"][i])): int(zl["cls"][i])
             for i in range(len(zl["idx"]))}
    d2m = json.load(open("config/dataset2maxlen.json"))
    ks = sorted(set(gt) & set(a1_lb))

    def apply(select_fn):
        per = {}
        for k in ks:
            per.setdefault(k[0], []).append(gt[k][select_fn(k)])
        return per

    def macro(per):
        return float(np.mean([np.mean(v) for v in per.values()]))

    v1 = apply(lambda k: table.get((bucket(d2m[k[0]]), a1_lb[k]), g_star))
    v2 = apply(lambda k: g_star if a1_lb[k] >= 3
               else table.get((bucket(d2m[k[0]]), a1_lb[k]), g_star))
    gfix = apply(lambda k: g_star)
    # diagnostic reference: LB in-menu oracle-fixed (per-dataset best, in-sample)
    ofix = {}
    for ds in sorted({d for d, _ in ks}):
        kk = [k for k in ks if k[0] == ds]
        mat = np.stack([gt[k] for k in kk])
        ofix[ds] = list(mat[:, int(np.argmax(mat.mean(0)))])
    print(f"\n[LB transfer, 13-grid menu, {len(ks)} samples]")
    print(f"  recipe-global-fixed (label-free): {macro(gfix):.2f}")
    print(f"  recipe table V1 (plain):          {macro(v1):.2f} ({macro(v1)-macro(gfix):+.2f})")
    print(f"  recipe table V2 (abstain D/E):    {macro(v2):.2f} ({macro(v2)-macro(gfix):+.2f})")
    print(f"  [diag] LB dataset-fixed (in-sample): {macro(ofix):.2f}")
    per_ds = {d: round(float(np.mean(v1[d]) - np.mean(gfix[d])), 2) for d in sorted(v1)}
    print(f"  V1 per-dataset delta: {per_ds}")
    out = {"model": args.model, "global": KEYS13[g_star],
           "table": {f"{c[0]}x{c[1]}": KEYS13[a] for c, a in table.items()},
           "lb": {"global_fixed": macro(gfix), "v1": macro(v1), "v2": macro(v2),
                  "lb_dataset_fixed": macro(ofix)}}
    json.dump(out, open(f"result_txt/analysis/proxy_gate/recipe_table_{args.model}.json", "w"),
              indent=1, ensure_ascii=False)
    print(f"[rt] saved -> result_txt/analysis/proxy_gate/recipe_table_{args.model}.json")


if __name__ == "__main__":
    main()
