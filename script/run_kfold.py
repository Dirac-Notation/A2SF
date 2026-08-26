"""k-fold CV for robust full-LB numbers. Each fold trains on (k-1)/k of LB, evals on 1/k;
average over k folds = full-LB cross-val score. More train data per fold (k=5 -> 80%) gives
better per-task estimates than 2-fold, recovering task-fixed-level accuracy.

  python script/run_kfold.py --gpus 0,1,2,3,4,5,6,7 --nfolds 5
"""
import argparse, json, os, subprocess, time
REPO = "/home/smp9898/A2SF"
PY = os.path.expanduser("~/miniconda3/envs/A2SF/bin/python")
LOG = f"{REPO}/logs/champion_repro_campaign.md"
DUMMY_MA = "runs/mini_attn_v5_8b/mini_attn_best.pt"
JOBS = [
    ("llama3-1b", "mini",   "runs/fast_lb_eval/lb_states_myv_none.pt", []),
    ("llama3-1b", "meta",   "runs/fast_lb_eval/lb_states_llama3-1b_meta.pt", []),
    ("llama3-8b", "meta",   "runs/fast_lb_eval/lb_states_llama3-8b_meta.pt", []),
    ("qwen2",     "meta",   "runs/fast_lb_eval/lb_states_qwen2_meta.pt", []),
    ("mistral-7b","meta",   "runs/fast_lb_eval/lb_states_mistral-7b_meta.pt", []),
]

def log(m):
    with open(LOG, "a") as f: f.write(m + "\n")
    print(m, flush=True)

def build(model, view, lbs, fold, k):
    pref = f"k{k}_{model}_{view}_f{fold}"
    subprocess.run([PY, f"{REPO}/script/cv_unified.py", "--model", model, "--lb_states", lbs,
        "--out_prefix", pref, "--fold", str(fold), "--nfolds", str(k)], capture_output=True, text=True)
    import torch
    tip = f"{REPO}/runs/fast_lb_eval/{pref}_test_index.pt"
    if not os.path.exists(tip): return None
    ix = torch.load(tip, map_location="cpu"); ix["datasets"] = sorted(set(x.split("/")[0] for x in ix if "/" in x))
    torch.save(ix, tip)
    return pref

def launch(model, pref, sv, gpu, EP=200):
    run = f"runs/{pref}_run"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = (f"{PY} {REPO}/RL/train.py --model {model} --budget 128 --states_file {REPO}/runs/states/{pref}_train.pt "
           f"--data_file {REPO}/datasets/cv/{pref}_train.jsonl --val_data_file {REPO}/datasets/cv/{pref}_val.jsonl "
           f"--score_field action_scores_gt_by_budget --val_score_field action_scores_gt_by_budget "
           f"--mini_attn_ckpt {REPO}/{DUMMY_MA} --extra_view none --loss listwise --loss_temp 0.1 "
           f"--epochs {EP} --ucb_topk 4 --ucb_beta 1.0 --seed 42 --save_dir {REPO}/{run} {' '.join(sv)} >/dev/null 2>&1 && "
           f"{PY} {REPO}/script/fast_lb_eval.py --rl_checkpoint {REPO}/{run}/policy_best.pt --run_name {pref} --budget 128 "
           f"--states_path {REPO}/runs/fast_lb_eval/{pref}_test_states.pt "
           f"--index_path {REPO}/runs/fast_lb_eval/{pref}_test_index.pt 2>/dev/null | grep 'Overall Average'")
    return subprocess.Popen(["bash", "-c", cmd], env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpus", default="0"); ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--models", default="")
    a = ap.parse_args(); gpus = [g for g in a.gpus.split(",") if g]; k = a.nfolds
    log(f"\n## {time.strftime('%H:%M')} {k}-FOLD CV (robust full-LB) gpus={gpus}")
    JOBS2 = [j for j in JOBS if (not a.models or j[0] in a.models.split(","))]
    tasks = []
    for model, view, lbs, sv in JOBS2:
        for fold in range(k):
            pref = build(model, view, lbs, fold, k)
            if pref: tasks.append((model, view, pref, sv, fold))
    res = {}
    for i in range(0, len(tasks), len(gpus)):
        wave = tasks[i:i + len(gpus)]
        procs = [(launch(m, p, sv, gpus[j], a.epochs), f"{m}/{v}", f) for j, (m, v, p, sv, f) in enumerate(wave)]
        for proc, key, f in procs:
            out, _ = proc.communicate(); lb = None
            for ln in (out or "").splitlines():
                if "Overall Average" in ln:
                    try: lb = float(ln.split(":")[-1].strip())
                    except: pass
            if lb is not None: res.setdefault(key, []).append(lb)
            log(f"    {key} fold{f} -> LB={lb}")
    log(f"## {k}-FOLD full-LB averages:")
    for key, vals in sorted(res.items()):
        if vals: log(f"    {key:18s} -> full-LB={sum(vals)/len(vals):.2f}  (n={len(vals)}, {[round(v,1) for v in vals]})")

if __name__ == "__main__":
    main()
