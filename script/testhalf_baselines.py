"""Score SnapKV/TOVA/H2O baselines on the SAME test-half used by the meta-only CV, for a
fair comparison with the unified-recipe RL. Replicates cv_unified's split (RandomState(0),
meta lb_states order = identity => test rows = te_i = LB line indices).
"""
import json, os, numpy as np, torch, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from longbench_eval import scorer
t2d = json.load(open(f"{REPO}/config/task2dataset.json"))
METHODS = {"SnapKV-16": "SnapKV-16", "SnapKV-32": "SnapKV-32", "TOVA": "TOVA", "H2O": "H2O"}

def test_rows(model):
    """Replicate cv_unified meta split -> {ds: array of test LB-line indices}."""
    ix = torch.load(f"{REPO}/runs/fast_lb_eval/index_{model}_128.pt", map_location="cpu")
    lbs = torch.load(f"{REPO}/runs/fast_lb_eval/lb_states_{model}_meta.pt", map_location="cpu")
    dss = [d for dl in t2d.values() for d in dl if d + "/scores" in ix and d + "/states" in lbs]
    rng = np.random.RandomState(0)
    out = {}
    for ds in dss:
        order = lbs[ds + "/order"].numpy(); N = len(order)
        perm = rng.permutation(N); half = N // 2
        te = perm[half:]
        out[ds] = np.array([int(order[i]) for i in te])  # meta order=arange -> LB line idx
    return out

def score_method(model, method, te):
    d = f"{REPO}/result_txt/backup/{model}/128/{model}_{method}_128"
    if not os.path.isdir(d): return None
    dsv = []
    for ds, rows in te.items():
        fp = os.path.join(d, f"{ds}.jsonl")
        if not os.path.exists(fp): continue
        all_rows = [json.loads(l) for l in open(fp)]
        sc = []
        for r in rows:
            if r >= len(all_rows): continue
            x = all_rows[r]
            try: s = scorer(ds, [x["pred"]], [x["answers"]], x.get("all_classes"))
            except: s = 0.0
            sc.append(s if s > 1 else s * 100)
        if sc: dsv.append(np.mean(sc))
    return np.mean(dsv) if dsv else None

print(f"{'model':12s} " + " ".join(f"{m:>9s}" for m in METHODS))
for model in ["llama3-1b", "llama3-8b", "qwen2", "mistral-7b"]:
    te = test_rows(model)
    row = [score_method(model, m, te) for m in METHODS]
    print(f"{model:12s} " + " ".join(f"{v:9.2f}" if v is not None else f"{'--':>9s}" for v in row))
