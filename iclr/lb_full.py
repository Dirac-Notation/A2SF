"""Load LongBench full-cache predictions aligned to fast_store sample indices.

Row order in result_txt/backup/<m>/<m>_full/<ds>.jsonl does NOT match the fast_store sample
index: the backup files are shard-shuffled. Joining on index therefore compares outputs from
different documents and turns every similarity analysis into noise (found 2026-09-05; it
invalidated the proxy analyses that preceded it). Join on (answers, length) instead.
"""
import json
import os


def _key(r):
    return (json.dumps(r.get("answers"), ensure_ascii=False), r.get("length"))


def load_full(model, dataset, root="result_txt/backup"):
    """Return {store_idx: full_cache_pred}; samples that cannot be aligned are skipped."""
    p = os.path.join(root, model, f"{model}_full", f"{dataset}.jsonl")
    if not os.path.exists(p):
        return {}
    try:
        rows = [json.loads(l) for l in open(p) if l.strip()]
    except Exception:
        return {}
    return {_key(r): (r.get("pred") or "") for r in rows}


def align(store, model, dataset, root="result_txt/backup"):
    """Return [(idx, store_rec, full_pred)], joined on (answers, length)."""
    fp = load_full(model, dataset, root)
    out = []
    for (ds, i), r in store.items():
        if ds != dataset:
            continue
        k = _key(r)
        if k in fp and fp[k].strip():
            out.append((i, r, fp[k]))
    return out
