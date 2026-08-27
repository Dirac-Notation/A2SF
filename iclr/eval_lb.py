"""Step 4: two-pass LongBench feasibility eval (accuracy ladder).

Rows (each = one pred dir under result_txt/pred/128/):
  fixed    : best single (a,b), uniform across heads      [1 pass]
  agent    : pass1 captures per-head K,V -> trained agent -> per-head (a,b)
             pass2 compressed generation with that table   [2 passes]
  oracle   : pass1 full-cache generation w/ capture -> per-head argmin AUC action
  shuffled : oracle table with head assignment permuted (control)

Pass-1 artifacts (per-sample tables) are cached under /data2 so rows share them.
LongBench is EVAL-ONLY: the agent never sees it during training (wikitext-only).

Usage (one dataset, one GPU):
  python iclr/eval_lb.py --model llama3-1b --dataset hotpotqa --row agent \
      --n_samples 50 --agent_ckpt iclr/runs/agent_1b/agent_best.pt --gpu 0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BUDGET = 128
RECENT = 16
DATASETS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
            "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
            "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]


# ── capture plumbing (trace_dump과 동일한 등록 방식) ──────────────────────────
from iclr.trace_dump import (CANDIDATES, Capture, w0_capture_attention,
                             curve_weights, _CAP)
from iclr.build_rewards import auc_cost


def per_head_reward_table(u, cand, noise_seed):
    """u [L,H,N], cand [C,L,H,N] -> (argmin idx [L,H], full ratio matrix [C,L,H]).
    ratio = AUC cost / oracle-ranking cost (>=1, lower better); saved per sample so
    LB-distribution profile/flatness analyses need no re-run."""
    C, L, H, N = cand.shape
    rng = np.random.RandomState(noise_seed)
    noise = rng.uniform(0, 1e-6, size=N).astype(np.float32)
    choice = np.zeros((L, H), dtype=np.int64)
    ratio = np.ones((C, L, H), dtype=np.float32)
    for l in range(L):
        for h in range(H):
            uv = u[l, h]
            c_star = auc_cost(np.argsort(-(uv + noise), kind="stable"), uv)
            if c_star <= 0:
                continue
            costs = [auc_cost(np.argsort(-(cand[c, l, h] + noise), kind="stable"), uv)
                     for c in range(C)]
            ratio[:, l, h] = np.array(costs, dtype=np.float32) / c_star
            choice[l, h] = int(np.argmin(costs))
    return choice, ratio


def choice_to_table(choice):
    a = [[CANDIDATES[c][0] for c in row] for row in choice.tolist()]
    b = [[CANDIDATES[c][1] for c in row] for row in choice.tolist()]
    return {"a_heads_by_layer": a, "b_heads_by_layer": b,
            "choice": choice.tolist()}


def build_prompt(model_name, ds, obj, tokenizer, max_len, prompt_fmt=None):
    """Mirrors script/worker_lb.py's per-sample path exactly: the local LB jsonls
    are PRE-FORMATTED (obj['input_prompt']); mid-truncate then chat-wrap."""
    import utils
    prompt = obj["input_prompt"]
    ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(ids) > max_len:
        half = max_len // 2
        prompt = tokenizer.decode(ids[:half], skip_special_tokens=True) + \
            tokenizer.decode(ids[-half:], skip_special_tokens=True)
    return utils.build_chat_prompt(prompt, model_name, tokenizer, ds)


@torch.no_grad()
def pass1_capture(model, tokenizer, prompt, max_gen, L, Hq, Hkv, device):
    """Full-cache greedy generation with capture -> (u, cand, K, V, text)."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    N = ids.shape[1]
    w = curve_weights(N, device)
    cap = Capture(L, Hkv, N, w, device)
    _CAP["obj"] = cap
    cap.phase = "prefill"
    out = model(ids, use_cache=True)
    past = out.past_key_values
    next_id = out.logits[:, -1:].argmax(-1)
    cap.phase = "decode"
    gen = []
    eos = tokenizer.eos_token_id
    for _ in range(max_gen):
        gen.append(int(next_id))
        if int(next_id) == eos:
            break
        step = model(next_id, past_key_values=past, use_cache=True)
        past = step.past_key_values
        next_id = step.logits[:, -1:].argmax(-1)
    _CAP["obj"] = None
    text = tokenizer.decode(gen, skip_special_tokens=True)
    K = torch.stack(cap.k_store)  # [L,Hkv,N,D] fp16 cpu
    V = torch.stack(cap.v_store)
    return (cap.u.cpu().numpy(), cap.cand.cpu().numpy(), K, V, text, N)


@torch.no_grad()
def pass1_forced(model, tokenizer, prompt, cont_text, L, Hkv, device):
    """Teacher-forced pass1: ONE forward of [prompt | stored full-cache pred].
    Rows below N feed cand, continuation rows feed u (combined capture phase)."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    N = ids.shape[1]
    cont = tokenizer(cont_text, return_tensors="pt", add_special_tokens=False).input_ids
    full = torch.cat([ids, cont], 1).to(device)
    w = curve_weights(N, device)
    cap = Capture(L, Hkv, N, w, device)
    cap.phase = "combined"
    _CAP["obj"] = cap
    model(full, use_cache=False)
    _CAP["obj"] = None
    return cap.u.numpy(), cap.cand.numpy(), N


def load_full_preds(model_name, dataset):
    """dataset -> {idx: full-cache pred text} from the canonical store."""
    import gzip
    out = {}
    path = f"result_txt/backup/fast_store/store_{model_name}_128.jsonl.gz"
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["dataset"] == dataset and "full" in r["actions"]:
                out[r["idx"]] = r["actions"]["full"]["pred"]
    return out


@torch.no_grad()
def agent_choice(agent, K, V, device):
    """K,V [L,H,N,D] fp16 -> per-head action idx [L,H]."""
    from iclr.train_agent import HeadAgents, doc_features
    L, H, N, D = K.shape
    feats = doc_features({"K": K.numpy(), "V": V.numpy()}, device)
    logits = agent(feats)                      # [L*H, C]
    return logits.argmax(-1).view(L, H).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3-1b")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--row", required=True,
                    choices=["fixed", "agent", "oracle", "shuffled"])
    ap.add_argument("--fixed_a", type=float, default=None)
    ap.add_argument("--fixed_b", type=float, default=None)
    ap.add_argument("--agent_ckpt", default="iclr/runs/agent_1b/agent_best.pt")
    ap.add_argument("--n_samples", type=int, default=50)
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--run_tag", default="feas")
    ap.add_argument("--also_agent", action="store_true",
                    help="oracle row: also emit the agent table from the same pass-1")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    os.chdir(ROOT)
    device = "cuda"
    import utils

    model2maxlen = json.load(open("config/model2maxlen.json"))
    max_gen = json.load(open("config/dataset2maxlen.json"))[args.dataset]
    max_len = model2maxlen[args.model]

    samples = []
    with open(f"datasets/longbench/{args.dataset}.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= args.n_samples:
                break

    cache_dir = f"/data2/smp9898/iclr_traces/lb_pass1/{args.model}/{args.dataset}"
    os.makedirs(cache_dir, exist_ok=True)
    out_dir = f"result_txt/pred/128/{args.run_tag}_{args.row}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.dataset}.jsonl")

    # ── pass 1 (agent/oracle/shuffled 공용) ──
    need_pass1 = args.row in ("agent", "oracle", "shuffled")
    if need_pass1:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS.register("w0_capture", w0_capture_attention)
        mp = json.load(open("config/model2path.json"))[args.model]
        tok1 = AutoTokenizer.from_pretrained(mp)
        m1 = AutoModelForCausalLM.from_pretrained(
            mp, torch_dtype=torch.bfloat16,
            attn_implementation="w0_capture").to(device).eval()
        L, Hq, Hkv = (m1.config.num_hidden_layers, m1.config.num_attention_heads,
                      m1.config.num_key_value_heads)
        full_preds = load_full_preds(args.model, args.dataset) \
            if args.row in ("oracle", "shuffled") else {}
        agent = None
        if args.row == "agent" or (args.row == "oracle" and args.also_agent):
            from iclr.train_agent import HeadAgents
            ck = torch.load(args.agent_ckpt, map_location=device, weights_only=False)
            agent = HeadAgents(ck["G"], ck["d_feat"], ck["C"]).to(device)
            agent.load_state_dict(ck["state_dict"])
            agent.eval()

        for i, obj in enumerate(samples):
            tpath = os.path.join(cache_dir, f"s{i:03d}_{args.row}.json")
            if os.path.exists(tpath):
                continue
            oracle_path = os.path.join(cache_dir, f"s{i:03d}_oracle.json")
            if args.row == "shuffled" and os.path.exists(oracle_path):
                ch = np.array(json.load(open(oracle_path))["choice"])
                rng = np.random.RandomState(1000 + i)
                flat = ch.flatten(); rng.shuffle(flat)
                json.dump(choice_to_table(flat.reshape(ch.shape)), open(tpath, "w"))
                continue
            prompt = build_prompt(args.model, args.dataset, obj, tok1, max_len)
            cont = full_preds.get(obj.get("idx"), "") if args.row != "agent" else ""
            if cont.strip():
                u, cand, N = pass1_forced(m1, tok1, prompt, cont, L, Hkv, device)
                K = V = None
            else:
                u, cand, K, V, _, N = pass1_capture(m1, tok1, prompt, max_gen,
                                                    L, Hq, Hkv, device)
            if args.row == "agent":
                ch = agent_choice(agent, K, V, device)
            else:
                ch, ratio = per_head_reward_table(u.astype(np.float32),
                                                  cand.astype(np.float32), i)
                np.savez_compressed(os.path.join(cache_dir, f"s{i:03d}_rewards.npz"),
                                    ratio=ratio, n_prefill=N)
                if args.row == "shuffled":
                    rng = np.random.RandomState(1000 + i)
                    flat = ch.flatten(); rng.shuffle(flat)
                    ch = flat.reshape(ch.shape)
                elif args.row == "oracle" and args.also_agent and K is not None:
                    ch_a = agent_choice(agent, K, V, device)
                    json.dump(choice_to_table(ch_a),
                              open(os.path.join(cache_dir, f"s{i:03d}_agent.json"), "w"))
            json.dump(choice_to_table(ch), open(tpath, "w"))
            print(f"[pass1] {args.dataset} s{i} done (N={N})", flush=True)
        del m1
        torch.cuda.empty_cache()

    # ── pass 2: 압축 생성 ──
    model, tokenizer = utils.load_model(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from utils_real_drop import CompressionConfig
    import utils as U

    done_idx = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done_idx = {json.loads(l)["idx"] for l in f}

    for i, obj in enumerate(samples):
        if i in done_idx:
            continue
        cfg = CompressionConfig()
        cfg["compression_method"] = "waits"
        cfg["total_budget"] = BUDGET
        cfg["recent_budget"] = RECENT
        cfg["observation_window"] = RECENT
        if args.row == "fixed":
            cfg["a"] = args.fixed_a; cfg["b"] = args.fixed_b
        else:
            t = json.load(open(os.path.join(cache_dir, f"s{i:03d}_{args.row}.json")))
            cfg["a"] = t["a_heads_by_layer"][0][0]
            cfg["b"] = t["b_heads_by_layer"][0][0]
            cfg["a_heads_by_layer"] = t["a_heads_by_layer"]
            cfg["b_heads_by_layer"] = t["b_heads_by_layer"]
        prompt = build_prompt(args.model, args.dataset, obj, tokenizer, max_len)
        enc = tokenizer(prompt, truncation=False, return_tensors="pt")
        input_ids = enc.input_ids.to(model.device)
        attention_mask = enc.attention_mask.to(torch.bfloat16).to(model.device)
        ctx_len = int(input_ids.shape[-1])
        model.init_cache(cfg)
        gen_kwargs = dict(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_gen, num_beams=1, do_sample=False,
            pad_token_id=tokenizer.eos_token_id, tokenizer=tokenizer,
            stop_strings=U.chat_stop_strings(args.model), num_logits_to_keep=1,
        )
        if args.dataset == "samsum":
            gen_kwargs["min_length"] = ctx_len + 1
            gen_kwargs["eos_token_id"] = [
                tokenizer.eos_token_id,
                tokenizer.encode("\n", add_special_tokens=False)[-1],
            ]
        with torch.inference_mode():
            out = model.generate(**gen_kwargs)[0]
        pred = tokenizer.decode(out[ctx_len:], skip_special_tokens=True)
        model.init_cache(None)
        with open(out_path, "a") as f:
            f.write(json.dumps({"idx": i, "pred": pred,
                                "answers": obj.get("answers"),
                                "all_classes": obj.get("all_classes"),
                                "length": obj.get("length")},
                               ensure_ascii=False) + "\n")
        print(f"[pass2:{args.row}] {args.dataset} s{i} done", flush=True)
    print(f"[eval_lb] {args.dataset} {args.row} COMPLETE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
