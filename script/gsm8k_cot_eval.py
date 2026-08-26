"""Rebuttal P0-c-2 (CdWY): short-prompt, long-generation CoT under KV compression.
GSM8K 8-shot CoT (prompt ~1k tokens > budget 128, generation up to 320 tokens).
Methods: full / TOVA / SnapKV / H2O / WAITS (submitted per-prompt agent:
MiniAttnEncoder + NeuralUCBAgent, per-prompt (a,b)).

  python script/gsm8k_cot_eval.py --model llama3-1b --gpus 4,5,6,7 \
      --rl_ckpt runs/repro_2694_seed42/policy_best.pt --n_samples 300

Output: result_txt/analysis/long_decoding/gsm8k_<model>.jsonl (row per sample:
{sid, gt, preds: {method: {ans, correct}}}). Accuracy = exact match of final number.
"""
import argparse
import json
import os
import re
import sys
import multiprocessing as mp

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from utils import CompressionConfig, load_model  # noqa: E402

N_SHOT = 8
MAX_NEW = 320


def build_cfg(method, budget, a=None, b=None):
    if method == "full":
        return None
    cfg = CompressionConfig()
    cfg["total_budget"] = int(budget)
    cfg["recent_budget"] = 16
    if method == "keydiff":
        cfg["compression_method"] = "keydiff"
        cfg["recent_budget"] = 0            # 원 논문 vanilla (sink/recent 없음)
        cfg["n_sink"] = 0
        return cfg
    if method == "snap":
        cfg["compression_method"] = "snap"
        cfg["observation_window"] = int(b)
        cfg["a"] = 10
        cfg["b"] = int(b)
    else:  # waits
        cfg["compression_method"] = "waits"
        cfg["a"] = float(a)
        cfg["b"] = int(b)
        cfg["observation_window"] = int(b)
    return cfg


def extract_number(text):
    text = text.split("Question:")[0]
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    if not m:
        return None
    return m[-1].replace(",", "").rstrip(".")


def load_rl(rl_ckpt, model, tokenizer):
    """Submitted per-prompt agent: MiniAttnEncoder + NeuralUCBAgent (as in benchmark_ttft)."""
    ckpt = torch.load(rl_ckpt, map_location="cpu", weights_only=False)
    ac = ckpt.get("arch_config", {}) or {}
    sd = ckpt.get("agent_state_dict", ckpt)
    dev = next(model.parameters()).device

    from RL.env.mini_attn_encoder import MiniAttnEncoder
    encoder = MiniAttnEncoder(
        target_model=model, target_tokenizer=tokenizer,
        mini_attn_ckpt_path=str(ac.get("mini_attn_ckpt")),
        device=str(dev),
        encoder_topk=int(ac.get("encoder_topk", 16)),
        max_input_length=int(ac.get("encoder_max_input_length", 32768)),
        include_hidden_pool=bool(int(ac.get("num_hidden_pool", 0)) > 0),
        hidden_pool_window=int(ac.get("encoder_hidden_pool_window", 0)),
        single_view=bool(int(ac.get("num_views", 2)) == 1),
        feature_mode=str(ac.get("encoder_feature_mode", "stats")),
        extra_view=str(ac.get("extra_view", "none")),
    )
    encoder.eval()

    from RL.agent.neural_ucb_agent import NeuralUCBAgent
    agent = NeuralUCBAgent(
        state_dim=int(ac["state_dim"]),
        a_values=ac["a_values"].to(torch.float32),
        b_values=ac["b_values"].to(torch.float32),
        num_metric_types=int(ac["num_metric_types"]),
        num_task_types=int(ac["num_task_types"]),
        side_dim=int(ac["side_dim"]),
        num_heads=int(ac["num_heads"]),
        num_hidden_pool=int(ac.get("num_hidden_pool", 0)),
        backbone_depth=int(ac.get("backbone_depth", 2)),
        dropout=0.0,
        paired_actions=bool(ac.get("paired_actions", True)),
        task_cond_head=bool(ac.get("task_cond_head", True)),
        task_head_mlp=bool(ac.get("task_head_mlp", False)),
        output_activation=str(ac.get("output_activation", "sigmoid")),
        num_views=int(ac.get("num_views", 1)),
        include_seq_len=bool(ac.get("include_seq_len", True)),
    ).to(dev)
    model_sd = agent.state_dict()
    fixed = {}
    for k, v in sd.items():
        if k in model_sd and v.shape != model_sd[k].shape and v.ndim == model_sd[k].ndim - 1 \
                and model_sd[k].shape[0] == 1:
            v = v.unsqueeze(0)
        fixed[k] = v
    agent.load_state_dict(fixed, strict=False)
    agent.eval()
    return encoder, agent


def _worker(wid, gpu, model_name, rl_ckpt, budget, shard, rq, fixed_action=None, all_methods=False, methods_sel=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.set_grad_enabled(False)
    model, tok = load_model(model_name)
    encoder = agent = None
    need_agent = fixed_action is None and (methods_sel is None or "waits" in methods_sel)
    if need_agent:
        encoder, agent = load_rl(rl_ckpt, model, tok)
    dev = next(model.parameters()).device
    print(f"[w{wid}] gpu={gpu} start, {len(shard)} samples", flush=True)

    for row in shard:
        sid, prompt, gt = row["sid"], row["prompt"], row["gt"]
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        preds = {}
        if methods_sel is not None:
            methods = []
            for m in methods_sel:
                if m == "waits":
                    wa, wb = float(fixed_action.split(":")[0]), int(fixed_action.split(":")[1])
                    methods.append(("waits", build_cfg("waits", budget, a=wa, b=wb)))
                elif m == "kvzip":
                    methods.append(("kvzip", "KVZIP"))
                elif m == "full":
                    methods.append(("full", None))
                elif m == "tova":
                    methods.append(("tova", build_cfg("snap", budget, b=1)))
                elif m == "snapkv":
                    methods.append(("snapkv", build_cfg("snap", budget, b=16)))
                elif m == "h2o":
                    methods.append(("h2o", build_cfg("snap", budget, b=32768)))
                else:
                    methods.append((m, build_cfg(m, budget)))
        elif fixed_action is not None:
            wa, wb = float(fixed_action.split(":")[0]), int(fixed_action.split(":")[1])
            methods = [("waits", build_cfg("waits", budget, a=wa, b=wb))]
            if all_methods:
                methods = [("full", None), ("tova", build_cfg("snap", budget, b=1)),
                           ("snapkv", build_cfg("snap", budget, b=16)),
                           ("h2o", build_cfg("snap", budget, b=32768))] + methods
        else:
            # per-prompt WAITS action (submitted agent)
            state = encoder.encode_context(prompt, generation_length=MAX_NEW,
                                           token_budget=budget,
                                           metric_type="qa_f1_score", task_type="Few Shot")
            (a_val, b_val), _ = agent.act(state.unsqueeze(0).to(dev, dtype=torch.float32))
            wa, wb = float(a_val.item()), int(round(float(b_val.item())))
            methods = [("full", None), ("tova", build_cfg("snap", budget, b=1)),
                       ("snapkv", build_cfg("snap", budget, b=16)),
                       ("h2o", build_cfg("snap", budget, b=32768)),
                       ("waits", build_cfg("waits", budget, a=wa, b=wb))]
        for name, cfg in methods:
            if cfg == "KVZIP":
                from utils_real_drop.kvzip import kvzip_generate
                gen_ids = kvzip_generate(model, tok, ids, max_new_tokens=MAX_NEW,
                                         budget=budget, recent_budget=16, n_sink=4,
                                         stop_token_ids=[tok.eos_token_id])
                text = tok.decode(gen_ids, skip_special_tokens=True)
            else:
                model.init_cache(cfg)
                with torch.inference_mode():
                    out = model.generate(input_ids=ids, max_new_tokens=MAX_NEW,
                                         do_sample=False, num_beams=1,
                                         pad_token_id=tok.eos_token_id,
                                         tokenizer=tok, stop_strings=["Question:"])
                text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            ans = extract_number(text)
            preds[name] = {"ans": ans, "correct": bool(ans == gt)}
        if "wa" in dir() or "wa" in locals():
            try:
                preds["waits_action"] = f"{wa:g}:{wb}"
            except NameError:
                pass
        rq.put({"sid": sid, "gt": gt, "preds": preds, "prompt_tokens": int(ids.shape[1])})
    rq.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--rl_ckpt", default="")
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--n_samples", type=int, default=300)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--fixed_action", default=None, help="a:b — skip agent, fixed WAITS curve, waits-only run")
    ap.add_argument("--all_methods", action="store_true", help="with --fixed_action: run full/tova/snapkv/h2o + fixed-action waits")
    ap.add_argument("--methods", default=None, help="comma list: full,tova,snapkv,h2o,waits,keydiff,kvzip — overrides method set")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from datasets import load_dataset
    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")

    shots = []
    for i in range(N_SHOT):
        q, a = train[i]["question"], train[i]["answer"]
        shots.append(f"Question: {q}\nAnswer: {a}\n")
    prefix = "\n".join(shots) + "\n"

    rows = []
    for i in range(args.start_idx, min(args.start_idx + args.n_samples, len(test))):
        gt = test[i]["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({"sid": i, "prompt": prefix + f"Question: {test[i]['question']}\nAnswer:",
                     "gt": gt})

    outdir = os.path.join(REPO, "result_txt/analysis/long_decoding")
    os.makedirs(outdir, exist_ok=True)
    out_p = os.path.join(outdir, f"gsm8k_{args.model}{args.tag}.jsonl")
    done = set()
    if os.path.exists(out_p):
        for l in open(out_p):
            done.add(json.loads(l)["sid"])
    todo = [r for r in rows if r["sid"] not in done]
    print(f"{len(todo)}/{len(rows)} to run")

    gpus = [int(x) for x in args.gpus.split(",")]
    shards = [todo[i::len(gpus)] for i in range(len(gpus))]
    ctx = mp.get_context("spawn")
    rq = ctx.Queue()
    procs = [ctx.Process(target=_worker,
                         args=(i, g, args.model, args.rl_ckpt, args.budget, shards[i], rq, args.fixed_action, args.all_methods,
                               args.methods.split(",") if args.methods else None))
             for i, g in enumerate(gpus)]
    for p in procs:
        p.start()
    n_alive, n_done = len(procs), 0
    with open(out_p, "a") as f:
        while n_alive:
            item = rq.get()
            if item is None:
                n_alive -= 1
                continue
            f.write(json.dumps(item) + "\n")
            f.flush()
            n_done += 1
            if n_done % 20 == 0:
                print(f"[gsm8k] {n_done}/{len(todo)}", flush=True)
    for p in procs:
        p.join()

    # summary
    accs = {}
    for l in open(out_p):
        r = json.loads(l)
        for m, v in r["preds"].items():
            if isinstance(v, dict):
                accs.setdefault(m, []).append(v["correct"])
    print("=== GSM8K 8-shot CoT accuracy ===")
    for m, v in sorted(accs.items()):
        print(f"{m:8s} {100.0 * sum(v) / len(v):.1f}%  (n={len(v)})")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
