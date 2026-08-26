"""General: learn a neural contextual-bandit Q over an arbitrary observable-metadata
state, and compare to the tabular (argmax) policy. --fields picks the state axes from
{task, subtype, metric}. Reward = action_scores_gt_by_budget; eval on real-LB index.

  python script/train_meta_neural.py --model llama3-1b \
      --recipe datasets/training/raw/recipe_v3_1b/train.jsonl \
      --index runs/fast_lb_eval/index_llama3-1b_128.pt --fields task,metric
"""
import argparse, json, os
import numpy as np, torch, torch.nn as nn
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEWSHOT_SUBTYPE = {"trec": "classification", "triviaqa": "qa", "samsum": "dialogue"}
# recipe-consistent metric label per LB dataset (non-Few-Shot determined by task)
TASK_METRIC = {"Single-doc QA": "qa_f1_score", "Multi-doc QA": "qa_f1_score",
               "Passage Retrieval": "qa_f1_score", "Code Complete": "code_sim_score",
               "Summarization": "rouge_score"}
FEWSHOT_METRIC = {"trec": "classification_score", "triviaqa": "qa_f1_score", "samsum": "rouge_score"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--recipe", required=True)
    ap.add_argument("--index", required=True); ap.add_argument("--budget", default="128")
    ap.add_argument("--fields", default="task,subtype")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=400)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    fields = a.fields.split(",")

    rows = [json.loads(l) for l in open(a.recipe)]
    def rec_val(r, fld):
        if fld == "task": return r["task_type"]
        if fld == "subtype": return r.get("subtype", "_")
        if fld == "metric": return r.get("metric_type", "?")
    vocab = {f: sorted({rec_val(r, f) for r in rows}) for f in fields}
    vi = {f: {v: i for i, v in enumerate(vocab[f])} for f in fields}
    offs = {}; D = 0
    for f in fields: offs[f] = D; D += len(vocab[f])

    def enc(vals):
        v = np.zeros(D, np.float32)
        for f in fields:
            j = vi[f].get(vals[f], vi[f].get("_", 0))
            v[offs[f] + j] = 1.0
        return v

    def reward(r):
        sc = r["action_scores_gt_by_budget"]; return sc[a.budget] if isinstance(sc, dict) else sc

    X = torch.tensor(np.array([enc({f: rec_val(r, f) for f in fields}) for r in rows]))
    R = torch.tensor(np.array([reward(r) for r in rows], np.float32))

    net = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 13))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(a.epochs):
        opt.zero_grad(); loss = ((net(X) - R) ** 2).mean(); loss.backward(); opt.step()

    # tabular policy over the same state
    by_s = defaultdict(list)
    for r in rows: by_s[tuple(rec_val(r, f) for f in fields)].append(reward(r))
    s_arg = {k: int(np.mean(v, 0).argmax()) for k, v in by_s.items()}
    by_t = defaultdict(list)
    for r in rows: by_t[r["task_type"]].append(reward(r))
    t_arg = {k: int(np.mean(v, 0).argmax()) for k, v in by_t.items()}

    d2t = json.load(open(f"{REPO}/config/task2dataset.json")); ds2task = {d: t for t, ds in d2t.items() for d in ds}
    def lb_val(ds, fld):
        t = ds2task[ds]
        if fld == "task": return t
        if fld == "subtype": return FEWSHOT_SUBTYPE.get(ds, "_")
        if fld == "metric": return FEWSHOT_METRIC.get(ds, TASK_METRIC.get(t, "qa_f1_score"))

    idx = torch.load(a.index, map_location="cpu", weights_only=False)
    net.eval(); v_nn, v_tab, v_tf = [], [], []
    with torch.no_grad():
        for k in idx:
            if not k.endswith("/scores"): continue
            ds = k[:-7]; sc = np.asarray(idx[k]); t = ds2task.get(ds)
            if sc.ndim != 2 or sc.shape[1] != 13 or t is None: continue
            vals = {f: lb_val(ds, f) for f in fields}
            a_nn = int(net(torch.tensor(enc(vals))[None]).argmax())
            a_tab = s_arg.get(tuple(vals[f] for f in fields), t_arg.get(t, 0))
            v_nn.append(sc[:, a_nn].mean()); v_tab.append(sc[:, a_tab].mean()); v_tf.append(sc[:, t_arg.get(t, 0)].mean())
    print(f"{a.model:11} [{a.fields:18}] task-fixed={np.mean(v_tf):.2f}  table={np.mean(v_tab):.2f}  "
          f"NEURAL-Q={np.mean(v_nn):.2f}  ({len(by_s)} cells)")


if __name__ == "__main__":
    main()
