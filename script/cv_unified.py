"""Cross-validation unified-recipe test: train RL on an LB-distributed split (per-task
optima MATCH eval), evaluate on the held-out split via the existing fast_lb_eval.

Produces: train jsonl + train states (for RL/train.py), and a TEST-half index + lb_states
in the SAME format as the originals (so fast_lb_eval works unchanged).

  python script/cv_unified.py --model llama3-8b --lb_states runs/fast_lb_eval/lb_states_8b_myv_none.pt
"""
import argparse, json, os, sys
import numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--lb_states", required=True)
ap.add_argument("--out_prefix", default=None)
ap.add_argument("--fold", type=int, default=0, help="fold index 0..nfolds-1 (test fold)")
ap.add_argument("--nfolds", type=int, default=2, help="k for k-fold CV")
a = ap.parse_args()
m = a.model
pref = a.out_prefix or f"cv_{m}"
t2d = json.load(open(f"{REPO}/config/task2dataset.json")); ds2task = {d: t for t, dl in t2d.items() for d in dl}
ix = torch.load(f"{REPO}/runs/fast_lb_eval/index_{m}_128.pt", map_location="cpu")
lbs = torch.load(f"{REPO}/{a.lb_states}", map_location="cpu")
dss = [d for dl in t2d.values() for d in dl if d + "/scores" in ix and d + "/states" in lbs]
rng = np.random.RandomState(0)
sd = int(lbs[dss[0] + "/states"].shape[-1])

train_rows = []; train_state_list = []
test_ix = {}; test_lbs = {"state_dim": sd}
FIELDS = ["preds", "scores", "answers", "all_classes", "lengths"]
for ds in dss:
    order = lbs[ds + "/order"].numpy(); st = lbs[ds + "/states"].numpy()
    N = len(order); perm = rng.permutation(N)
    k = max(2, a.nfolds); fs = N // k
    lo = a.fold * fs; hi = N if a.fold == k - 1 else (a.fold + 1) * fs
    te_i = perm[lo:hi]
    tr_i = np.concatenate([perm[:lo], perm[hi:]])
    sc = ix[ds + "/scores"].numpy()
    # train rows + states
    for i in tr_i:
        ri = int(order[i]); sid = f"{ds}__{ri}"
        s = [float(x) for x in sc[ri]]
        train_rows.append({"sample_id": sid, "input_prompt": "", "task_type": ds2task.get(ds, "?"),
                           "dataset": ds, "length": 0,
                           "action_scores_gt_by_budget": {"128": s},
                           "action_scores_maxo_by_budget": {"128": s}})
        train_state_list.append(torch.tensor(st[i]))
    # test index (subset all fields to test rows, re-indexed to arange)
    te_rows = [int(order[i]) for i in te_i]
    for f in FIELDS:
        key = ds + "/" + f
        if key not in ix: continue
        val = ix[key]
        if isinstance(val, torch.Tensor):
            test_ix[key] = val[te_rows]
        else:
            test_ix[key] = [val[r] for r in te_rows]
    # test lb_states (states for te_i, order = arange)
    test_lbs[ds + "/states"] = torch.tensor(st[te_i])
    test_lbs[ds + "/order"] = torch.arange(len(te_i))

os.makedirs(f"{REPO}/datasets/cv", exist_ok=True)
tj = f"{REPO}/datasets/cv/{pref}_train.jsonl"; vj = f"{REPO}/datasets/cv/{pref}_val.jsonl"
with open(tj, "w") as f:
    for r in train_rows: f.write(json.dumps(r) + "\n")
N = len(train_rows)
nval = max(20, N // 10)
with open(vj, "w") as f:
    for r in train_rows[-nval:]: f.write(json.dumps(r) + "\n")
# states keyed by train.py's _prompt_id: train line i -> i; val line j -> N + j
states_flat = {i: train_state_list[i] for i in range(N)}
for j in range(nval):
    states_flat[N + j] = train_state_list[N - nval + j]
states_flat["state_dim"] = sd
# CRITICAL: copy encoder meta (num_task_types/num_metric_types/...) so train.py builds
# the per-task head. Without these train.py defaults num_task_types=0 -> no per-task head.
# Prefer meta carried by the input lb_states itself; else fall back to a mini-attn ref.
META_KEYS = ["num_metric_types", "num_task_types", "side_dim", "num_heads", "num_hidden_pool", "config"]
if all(k in lbs for k in ["num_metric_types", "num_task_types"]):
    ref = lbs
else:
    rp = f"{REPO}/runs/states/old_8b_2v_none.pt" if sd > 60 else f"{REPO}/runs/states/old_8b_none.pt"
    ref = torch.load(rp, map_location="cpu") if os.path.exists(rp) else {}
for k in META_KEYS:
    if k in ref:
        states_flat[k] = ref[k]
torch.save(states_flat, f"{REPO}/runs/states/{pref}_train.pt")
print(f"copied meta: num_task_types={ref.get('num_task_types')}", flush=True)
torch.save(test_ix, f"{REPO}/runs/fast_lb_eval/{pref}_test_index.pt")
torch.save(test_lbs, f"{REPO}/runs/fast_lb_eval/{pref}_test_states.pt")
# test-half task-fixed + oracle + per-action (reference), computed on test scores
tf_ref = {}
for task, dl in t2d.items():
    dd = [d for d in dl if d + "/scores" in test_ix]
    if not dd: continue
    a_best = int(np.argmax([np.mean([test_ix[d + "/scores"].numpy().mean(0)[k] for d in dd]) for k in range(13)]))
    for d in dd: tf_ref[d] = a_best
allds = [d for d in dss if d + "/scores" in test_ix]
tf = np.mean([test_ix[d + "/scores"].numpy().mean(0)[tf_ref[d]] for d in allds])
orc = np.mean([test_ix[d + "/scores"].numpy().mean(0).max() for d in allds])
print(f"built {len(train_rows)} train rows (state_dim={sd}); TEST-half: task-fixed={tf:.2f} per-ds-oracle={orc:.2f}", flush=True)
print(f"FILES tj={tj} states=runs/states/{pref}_train.pt test_index=runs/fast_lb_eval/{pref}_test_index.pt test_states=runs/fast_lb_eval/{pref}_test_states.pt", flush=True)
