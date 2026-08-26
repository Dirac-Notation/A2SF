"""Parallel unified-recipe CV campaign across a GPU pool. Builds each (model,view) LB
train/test split once, then runs the Reward x Loss grid as PARALLEL subprocesses bound
to GPUs. Logs test-half LB. Reproduces champion (mini-attn 1B/8B) + extends to qwen2/
mistral (meta-only) under ONE recipe.

  python script/parallel_cv.py --gpus 0,1,2,3,4,5,6,7
"""
import argparse, json, os, subprocess, time, itertools
REPO = "/home/smp9898/A2SF"
PY = os.path.expanduser("~/miniconda3/envs/A2SF/bin/python")
LOG = f"{REPO}/logs/champion_repro_campaign.md"
DUMMY_MA = "runs/mini_attn_v5_8b/mini_attn_best.pt"

# (model, view, lb_states)  — view 'mini'/'mini2v' use encoder states; 'meta' = model-agnostic
JOBS = [
    ("llama3-8b", "mini",   "runs/fast_lb_eval/lb_states_8b_myv_none.pt"),
    ("llama3-8b", "mini2v", "runs/fast_lb_eval/lb_states_8b_2v_none.pt"),
    ("llama3-1b", "mini",   "runs/fast_lb_eval/lb_states_myv_none.pt"),
    ("llama3-8b", "meta",   "runs/fast_lb_eval/lb_states_llama3-8b_meta.pt"),
    ("llama3-1b", "meta",   "runs/fast_lb_eval/lb_states_llama3-1b_meta.pt"),
    ("qwen2",     "meta",   "runs/fast_lb_eval/lb_states_qwen2_meta.pt"),
    ("mistral-7b","meta",   "runs/fast_lb_eval/lb_states_mistral-7b_meta.pt"),
]
REWARDS = ["gt", "maxo"]; LOSSES = ["listwise", "mse"]

def log(m):
    with open(LOG, "a") as f: f.write(m + "\n")
    print(m, flush=True)

def build(model, view, lb_states):
    pref = f"pcv_{model}_{view}"
    r = subprocess.run([PY, f"{REPO}/script/cv_unified.py", "--model", model,
        "--lb_states", lb_states, "--out_prefix", pref], capture_output=True, text=True)
    import torch
    tip = f"{REPO}/runs/fast_lb_eval/{pref}_test_index.pt"
    if not os.path.exists(tip):
        return None, "build-fail: " + (r.stderr[-200:] if r.stderr else "")
    ix = torch.load(tip, map_location="cpu"); ix["datasets"] = sorted(set(k.split("/")[0] for k in ix if "/" in k))
    torch.save(ix, tip)
    tf = "?"
    for ln in (r.stdout + r.stderr).splitlines():
        if "task-fixed=" in ln: tf = ln.split("task-fixed=")[1].split()[0]
    return pref, tf

def launch(model, view, pref, reward, loss, gpu):
    field = f"action_scores_{reward}_by_budget"
    run = f"runs/pcv_{model}_{view}_{reward}_{loss}"
    sv = ["--no_single_view"] if view == "mini2v" else []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = (f"{PY} {REPO}/RL/train.py --model {model} --budget 128 "
           f"--states_file {REPO}/runs/states/{pref}_train.pt "
           f"--data_file {REPO}/datasets/cv/{pref}_train.jsonl --val_data_file {REPO}/datasets/cv/{pref}_val.jsonl "
           f"--score_field {field} --val_score_field {field} --mini_attn_ckpt {REPO}/{DUMMY_MA} "
           f"--extra_view none --loss {loss} --loss_temp 0.1 --epochs 200 --ucb_topk 4 --ucb_beta 1.0 "
           f"--seed 42 --save_dir {REPO}/{run} {' '.join(sv)} >/dev/null 2>&1 && "
           f"{PY} {REPO}/script/fast_lb_eval.py --rl_checkpoint {REPO}/{run}/policy_best.pt "
           f"--run_name {model}_{view}_{reward}_{loss} --budget 128 "
           f"--states_path {REPO}/runs/fast_lb_eval/{pref}_test_states.pt "
           f"--index_path {REPO}/runs/fast_lb_eval/{pref}_test_index.pt 2>/dev/null | grep 'Overall Average'")
    return subprocess.Popen(["bash", "-c", cmd], env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpus", default="0")
    a = ap.parse_args()
    gpus = [g for g in a.gpus.split(",") if g]
    log(f"\n## {time.strftime('%H:%M')} PARALLEL CV campaign gpus={gpus}")
    # build all splits
    built = []
    for model, view, lbs in JOBS:
        pref, tf = build(model, view, lbs)
        if pref is None: log(f"  [{model} {view}] BUILD FAIL {tf}"); continue
        log(f"  [{model} {view}] split tf={tf}")
        for reward, loss in itertools.product(REWARDS, LOSSES):
            built.append((model, view, pref, reward, loss))
    # run in parallel waves of len(gpus)
    best = {}
    for i in range(0, len(built), len(gpus)):
        wave = built[i:i + len(gpus)]
        procs = []
        for j, (model, view, pref, reward, loss) in enumerate(wave):
            p = launch(model, view, pref, reward, loss, gpus[j])
            procs.append((p, model, view, reward, loss))
        for p, model, view, reward, loss in procs:
            out, _ = p.communicate()
            lb = None
            for ln in (out or "").splitlines():
                if "Overall Average" in ln:
                    try: lb = float(ln.split(":")[-1].strip())
                    except: pass
            tag = f"{model} {view} {reward} {loss}"
            if lb is None: log(f"    {tag:34s} -> ERR")
            else:
                log(f"    {tag:34s} -> LB={lb:.2f}")
                key = f"{model}/{view}"
                if lb > best.get(key, (0, ""))[0]: best[key] = (lb, tag)
            subprocess.run(["rm", "-rf", f"{REPO}/runs/pcv_{model}_{view}_{reward}_{loss}"])
    log(f"## {time.strftime('%H:%M')} PARALLEL CV best: " + " | ".join(f"{k}:{v[0]:.2f}" for k, v in sorted(best.items())))

if __name__ == "__main__":
    main()
