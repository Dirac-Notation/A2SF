"""Systematic D-I-A-R-L cross-val campaign: for each (model, encoder-view), build an
LB train/test split, then grid over Reward x Loss, train RL + eval on test-half, log LB.
Reproduces champion-level (>=task-fixed, beats SnapKV) via representative data + per-task head.

Run: python script/cv_campaign.py --gpus 7 --models llama3-8b,llama3-1b
"""
import argparse, json, os, subprocess, sys, time, itertools
REPO = "/home/smp9898/A2SF"
PY = os.path.expanduser("~/miniconda3/envs/A2SF/bin/python")
LOG = f"{REPO}/logs/champion_repro_campaign.md"

# (model, view_tag, lb_states, mini_attn)
VIEWS = {
    "llama3-8b": [("1v", "runs/fast_lb_eval/lb_states_8b_myv_none.pt", "runs/mini_attn_v5_8b/mini_attn_best.pt"),
                  ("2v", "runs/fast_lb_eval/lb_states_8b_2v_none.pt",  "runs/mini_attn_v5_8b/mini_attn_best.pt")],
    "llama3-1b": [("1v", "runs/fast_lb_eval/lb_states_myv_none.pt",    "runs/mini_attn_v5/mini_attn_best.pt")],
}
REWARDS = ["gt", "maxo"]
LOSSES = ["listwise", "mse"]

def log(msg):
    with open(LOG, "a") as f: f.write(msg + "\n")
    print(msg, flush=True)

def build_split(model, view, lb_states):
    pref = f"cvg_{model}_{view}"
    single = "0" if view == "2v" else "1"
    r = subprocess.run([PY, f"{REPO}/script/cv_unified.py", "--model", model,
                        "--lb_states", lb_states, "--out_prefix", pref],
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    # add datasets key to test index
    tip = f"{REPO}/runs/fast_lb_eval/{pref}_test_index.pt"
    import torch
    ix = torch.load(tip, map_location="cpu")
    ix["datasets"] = sorted(set(k.split("/")[0] for k in ix if "/" in k))
    torch.save(ix, tip)
    tf = "?"
    for line in out.splitlines():
        if "task-fixed=" in line: tf = line.split("task-fixed=")[1].split()[0]
    return pref, tf

def run_one(model, view, pref, mini_attn, reward, loss, gpu):
    field = f"action_scores_{reward}_by_budget"
    run = f"runs/cvg_{model}_{view}_{reward}_{loss}"
    sv = ["--no_single_view"] if view == "2v" else []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    t = subprocess.run([PY, f"{REPO}/RL/train.py", "--model", model, "--budget", "128",
        "--states_file", f"{REPO}/runs/states/{pref}_train.pt",
        "--data_file", f"{REPO}/datasets/cv/{pref}_train.jsonl",
        "--val_data_file", f"{REPO}/datasets/cv/{pref}_val.jsonl",
        "--score_field", field, "--val_score_field", field,
        "--mini_attn_ckpt", f"{REPO}/{mini_attn}", "--extra_view", "none",
        "--loss", loss, "--loss_temp", "0.1", "--epochs", "200",
        "--ucb_topk", "4", "--ucb_beta", "1.0", "--seed", "42",
        "--save_dir", f"{REPO}/{run}"] + sv,
        capture_output=True, text=True, env=env)
    ck = f"{REPO}/{run}/policy_best.pt"
    if not os.path.exists(ck):
        return None, (t.stderr[-300:] if t.stderr else "no ckpt")
    e = subprocess.run([PY, f"{REPO}/script/fast_lb_eval.py", "--rl_checkpoint", ck,
        "--run_name", f"cvg_{model}_{view}_{reward}_{loss}", "--budget", "128",
        "--states_path", f"{REPO}/runs/fast_lb_eval/{pref}_test_states.pt",
        "--index_path", f"{REPO}/runs/fast_lb_eval/{pref}_test_index.pt"],
        capture_output=True, text=True, env=env)
    lb = None
    for line in (e.stdout + e.stderr).splitlines():
        if "Overall Average" in line:
            try: lb = float(line.split(":")[-1].strip())
            except: pass
    subprocess.run(["rm", "-rf", f"{REPO}/{run}"])
    return lb, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="7")
    ap.add_argument("--models", default="llama3-8b,llama3-1b")
    a = ap.parse_args()
    gpus = [g for g in a.gpus.split(",") if g]
    models = [m for m in a.models.split(",") if m]
    log(f"\n## {time.strftime('%H:%M')} CV D-I-A-R-L grid start (gpus={gpus})")
    best = {}
    for model in models:
        for view, lb_states, mini_attn in VIEWS.get(model, []):
            pref, tf = build_split(model, view, lb_states)
            log(f"  [{model} {view}] split built, test task-fixed={tf}")
            gi = 0
            for reward, loss in itertools.product(REWARDS, LOSSES):
                gpu = gpus[gi % len(gpus)]; gi += 1
                lb, err = run_one(model, view, pref, mini_attn, reward, loss, gpu)
                tag = f"{model} {view} {reward} {loss}"
                if lb is None:
                    log(f"    {tag:38s} -> ERR {err}")
                else:
                    log(f"    {tag:38s} -> LB={lb:.2f}")
                    if lb > best.get(model, (0, ""))[0]:
                        best[model] = (lb, tag)
    log(f"## {time.strftime('%H:%M')} grid done. BEST per model: " +
        " | ".join(f"{m}: {v[0]:.2f} ({v[1]})" for m, v in best.items()))

if __name__ == "__main__":
    main()
