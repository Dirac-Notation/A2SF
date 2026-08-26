"""2-fold CV for FULL-LB numbers (paper denominator). For each (model,view): fold0 trains
on half-A evals half-B, fold1 trains on half-B evals half-A; the two test-half overalls
average to the full-LB cross-val score. Winning config only: gt + listwise + per-task head.

  python script/run_2fold.py --gpus 0,1,2,3,4,5,6,7
"""
import argparse, json, os, subprocess, time
REPO = "/home/smp9898/A2SF"
PY = os.path.expanduser("~/miniconda3/envs/A2SF/bin/python")
LOG = f"{REPO}/logs/champion_repro_campaign.md"
DUMMY_MA = "runs/mini_attn_v5_8b/mini_attn_best.pt"
JOBS = [
    ("llama3-1b", "mini",   "runs/fast_lb_eval/lb_states_myv_none.pt", []),
    ("llama3-1b", "meta",   "runs/fast_lb_eval/lb_states_llama3-1b_meta.pt", []),
    ("llama3-8b", "mini",   "runs/fast_lb_eval/lb_states_8b_myv_none.pt", []),
    ("llama3-8b", "meta",   "runs/fast_lb_eval/lb_states_llama3-8b_meta.pt", []),
    ("qwen2",     "meta",   "runs/fast_lb_eval/lb_states_qwen2_meta.pt", []),
    ("mistral-7b","meta",   "runs/fast_lb_eval/lb_states_mistral-7b_meta.pt", []),
]

def log(m):
    with open(LOG, "a") as f: f.write(m + "\n")
    print(m, flush=True)

def build(model, view, lbs, fold):
    pref = f"f2_{model}_{view}_f{fold}"
    r = subprocess.run([PY, f"{REPO}/script/cv_unified.py", "--model", model, "--lb_states", lbs,
        "--out_prefix", pref, "--fold", str(fold)], capture_output=True, text=True)
    import torch
    tip = f"{REPO}/runs/fast_lb_eval/{pref}_test_index.pt"
    if not os.path.exists(tip): return None
    ix = torch.load(tip, map_location="cpu"); ix["datasets"] = sorted(set(k.split("/")[0] for k in ix if "/" in k))
    torch.save(ix, tip)
    return pref

def launch(model, view, pref, sv, gpu):
    run = f"runs/{pref}_run"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = (f"{PY} {REPO}/RL/train.py --model {model} --budget 128 "
           f"--states_file {REPO}/runs/states/{pref}_train.pt "
           f"--data_file {REPO}/datasets/cv/{pref}_train.jsonl --val_data_file {REPO}/datasets/cv/{pref}_val.jsonl "
           f"--score_field action_scores_gt_by_budget --val_score_field action_scores_gt_by_budget "
           f"--mini_attn_ckpt {REPO}/{DUMMY_MA} --extra_view none --loss listwise --loss_temp 0.1 "
           f"--epochs 200 --ucb_topk 4 --ucb_beta 1.0 --seed 42 --save_dir {REPO}/{run} {' '.join(sv)} >/dev/null 2>&1 && "
           f"{PY} {REPO}/script/fast_lb_eval.py --rl_checkpoint {REPO}/{run}/policy_best.pt --run_name {pref} --budget 128 "
           f"--states_path {REPO}/runs/fast_lb_eval/{pref}_test_states.pt "
           f"--index_path {REPO}/runs/fast_lb_eval/{pref}_test_index.pt 2>/dev/null | grep 'Overall Average'")
    return subprocess.Popen(["bash", "-c", cmd], env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpus", default="0"); a = ap.parse_args()
    gpus = [g for g in a.gpus.split(",") if g]
    log(f"\n## {time.strftime('%H:%M')} 2-FOLD CV (full-LB) gpus={gpus}")
    tasks = []
    for model, view, lbs, sv in JOBS:
        for fold in (0, 1):
            pref = build(model, view, lbs, fold)
            if pref: tasks.append((model, view, pref, sv, fold))
            else: log(f"  build fail {model} {view} f{fold}")
    res = {}
    for i in range(0, len(tasks), len(gpus)):
        wave = tasks[i:i + len(gpus)]
        procs = [(launch(m, v, p, sv, gpus[j]), m, v, f) for j, (m, v, p, sv, f) in enumerate(wave)]
        for proc, m, v, f in procs:
            out, _ = proc.communicate(); lb = None
            for ln in (out or "").splitlines():
                if "Overall Average" in ln:
                    try: lb = float(ln.split(":")[-1].strip())
                    except: pass
            res.setdefault(f"{m}/{v}", {})[f] = lb
            log(f"    {m} {v} fold{f} -> LB={lb}")
    log("## 2-FOLD full-LB (avg of folds):")
    for k, folds in sorted(res.items()):
        if 0 in folds and 1 in folds and folds[0] and folds[1]:
            log(f"    {k:18s} -> full-LB={ (folds[0]+folds[1])/2 :.2f}  (f0={folds[0]:.2f} f1={folds[1]:.2f})")

if __name__ == "__main__":
    main()
