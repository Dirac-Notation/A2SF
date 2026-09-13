"""Merge a doc-QA supplement into an existing recipe pool and its probe features.

  recipe_<base>_<model>.jsonl + generated rows  ->  recipe_<newpool>_<model>.jsonl
  probe_<basevar>/<model>_recipe.npz + probe_<docvar>/...  ->  probe_<newvar>/...

The _lb.npz file is copied from the base variant unchanged, since the LongBench probe does
not depend on the training pool.

  python iclr/merge_docqa.py --model llama3-8b --gen docqa_gen.jsonl \
      --base_pool poolx --new_pool pooly --base_var v1t64 --doc_var docqa64 --new_var v1t64d
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

ROOT = os.environ.get("ICLR_TRACES", "/data2/smp9898/iclr_traces")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--base_pool", default="poolx")
    ap.add_argument("--new_pool", default="pooly")
    ap.add_argument("--base_var", default="v1t64")
    ap.add_argument("--doc_var", default="docqa64")
    ap.add_argument("--new_var", default="v1t64d")
    a = ap.parse_args()

    src = f"datasets/training/raw/recipe_{a.base_pool}_{a.model}.jsonl"
    dst = f"datasets/training/raw/recipe_{a.new_pool}_{a.model}.jsonl"
    base = [json.loads(l) for l in open(src)]
    new = [json.loads(l) for l in open(a.gen)]
    ok = [o for o in new if (o.get("full_cache_pred") or "").strip() and len(o.get("action_outputs", [])) == 13]
    with open(dst, "w") as f:
        for o in base + ok:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"[merge] recipe {len(base)} + {len(ok)}/{len(new)} -> {len(base)+len(ok)}  {dst}")

    od = f"{ROOT}/probe_{a.new_var}"
    os.makedirs(od, exist_ok=True)
    zb = np.load(f"{ROOT}/probe_{a.base_var}/{a.model}_recipe.npz")
    zd = np.load(f"{ROOT}/probe_{a.doc_var}/{a.model}_recipe.npz")
    keep = {str(o["sample_id"]) for o in ok}
    m = np.array([str(k) in keep for k in zd["key"]])
    out = {k: np.concatenate([zb[k].astype("<U32") if k == "key" else zb[k],
                              zd[k][m].astype("<U32") if k == "key" else zd[k][m]])
           for k in zb.files}
    np.savez(f"{od}/{a.model}_recipe.npz", **out)
    shutil.copy(f"{ROOT}/probe_{a.base_var}/{a.model}_lb.npz", f"{od}/{a.model}_lb.npz")
    print(f"[merge] probe {len(zb['key'])} + {int(m.sum())} -> {len(out['key'])}  {od}")


if __name__ == "__main__":
    main()
