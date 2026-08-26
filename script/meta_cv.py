"""Meta-only cross-val: model-agnostic unified recipe for ALL models (incl qwen2/mistral
where the LLaMA-only encoders fail). Builds LB train/test split from meta-only states,
trains RL (per-task head) on train-half, evals on test-half. Should reach ~task-fixed
(beats SnapKV) on every model.

  python script/meta_cv.py --gpus 0,1,2,3 --models qwen2,mistral-7b,llama3-8b,llama3-1b
"""
import argparse, json, os, subprocess, time, itertools
REPO = "/home/smp9898/A2SF"
PY = os.path.expanduser("~/miniconda3/envs/A2SF/bin/python")
LOG = f"{REPO}/logs/champion_repro_campaign.md"
DUMMY_MA = "runs/mini_attn_v5_8b/mini_attn_best.pt"  # unused when --states_file given
REWARDS = ["gt", "maxo"]; LOSSES = ["listwise", "mse"]

def log(m):
    with open(LOG, "a") as f: f.write(m + "\n")
    print(m, flush=True)

def build(model):
    pref = f"mcv_{model}"
    r = subprocess.run([PY, f"{REPO}/script/cv_unified.py", "--model", model,
        "--lb_states", f"runs/fast_lb_eval/lb_states_{model}_meta.pt", "--out_prefix", pref],
        capture_output=True, text=True)
    import torch
    tip = f"{REPO}/runs/fast_lb_eval/{pref}_test_index.pt"
    ix = torch.load(tip, map_location="cpu"); ix["datasets"] = sorted(set(k.split("/")[0] for k in ix if "/" in k))
    torch.save(ix, tip)
    tf = "?"
    for ln in (r.stdout + r.stderr).splitlines():
        if "task-fixed=" in ln: tf = ln.split("task-fixed=")[1].split()[0]
    return pref, tf

def run_one(model, pref, reward, loss, gpu):
    field = f"action_scores_{reward}_by_budget"
    run = f"runs/mcv_{model}_{reward}_{loss}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    subprocess.run([PY, f"{REPO}/RL/train.py", "--model", model, "--budget", "128",
        "--states_file", f"{REPO}/runs/states/{pref}_train.pt",
        "--data_file", f"{REPO}/datasets/cv/{pref}_train.jsonl",
        "--val_data_file", f"{REPO}/datasets/cv/{pref}_val.jsonl",
        "--score_field", field, "--val_score_field", field,
        "--mini_attn_ckpt", DUMMY_MA, "--extra_view", "none",
        "--loss", loss, "--loss_temp", "0.1", "--epochs", "200",
        "--ucb_topk", "4", "--ucb_beta", "1.0", "--seed", "42",
        "--save_dir", f"{REPO}/{run}"], capture_output=True, text=True, env=env)
    ck = f"{REPO}/{run}/policy_best.pt"
    if not os.path.exists(ck): return None
    e = subprocess.run([PY, f"{REPO}/script/fast_lb_eval.py", "--rl_checkpoint", ck,
        "--run_name", f"mcv_{model}_{reward}_{loss}", "--budget", "128",
        "--states_path", f"{REPO}/runs/fast_lb_eval/{pref}_test_states.pt",
        "--index_path", f"{REPO}/runs/fast_lb_eval/{pref}_test_index.pt"],
        capture_output=True, text=True, env=env)
    lb = None
    for ln in (e.stdout + e.stderr).splitlines():
        if "Overall Average" in ln:
            try: lb = float(ln.split(":")[-1].strip())
            except: pass
    subprocess.run(["rm", "-rf", f"{REPO}/{run}"])
    return lb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--models", default="qwen2,mistral-7b,llama3-8b,llama3-1b")
    a = ap.parse_args()
    gpus = [g for g in a.gpus.split(",") if g]
    log(f"\n## {time.strftime('%H:%M')} META-only CV (model-agnostic unified) gpus={gpus}")
    best = {}
    for model in a.models.split(","):
        pref, tf = build(model)
        log(f"  [{model}] meta split, test task-fixed={tf}")
        gi = 0
        for reward, loss in itertools.product(REWARDS, LOSSES):
            gpu = gpus[gi % len(gpus)]; gi += 1
            lb = run_one(model, pref, reward, loss, gpu)
            tag = f"{model} meta {reward} {loss}"
            if lb is None: log(f"    {tag:34s} -> ERR")
            else:
                log(f"    {tag:34s} -> LB={lb:.2f}")
                if lb > best.get(model, (0, ""))[0]: best[model] = (lb, tag)
    log(f"## {time.strftime('%H:%M')} META-CV best: " + " | ".join(f"{m}:{v[0]:.2f}" for m, v in best.items()))

if __name__ == "__main__":
    main()
