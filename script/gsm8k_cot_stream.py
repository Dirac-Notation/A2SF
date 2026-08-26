"""True short-prompt / long-generation CoT: zero-shot 질문 프롬프트(~100토큰) + 긴 CoT
생성(<=768토큰), 생성 중 CHUNK마다 캐시를 budget으로 재압축 (crop-refeed: 마지막 CHUNK
토큰의 KV를 잘라내고 동일 위치로 재전방→ q_len>1 스코어링 경로가 자동 압축).

  python script/gsm8k_cot_stream.py --model llama3-8b --gpus 0,1 \
      --methods full,tova,snapkv,h2o,waits --waits_action 0.1:1 --n_samples 200
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
from utils_real_drop.compress import CompressedCache  # noqa: E402

CHUNK = 256
MAX_NEW = 3072
INSTR = ("Solve the following math problem. Reason step by step, "
         "and finish your answer with '#### <number>'.\n\n")


def build_cfg(method, budget, a=None, b=None):
    if method == "full":
        return None
    cfg = CompressionConfig()
    cfg["total_budget"] = int(budget)
    cfg["recent_budget"] = 16
    if method == "keydiff":
        cfg["compression_method"] = "keydiff"
        cfg["recent_budget"] = 0
        cfg["n_sink"] = 0
        return cfg
    if method == "waits":
        cfg["compression_method"] = "waits"
        cfg["a"] = float(a); cfg["b"] = int(b); cfg["observation_window"] = int(b)
    else:
        w = {"tova": 1, "snapkv": 16, "h2o": 32768}[method]
        cfg["compression_method"] = "snap"
        cfg["observation_window"] = w; cfg["a"] = 10; cfg["b"] = w
    return cfg


def crop_last(cache, w):
    for layer in cache.layers:
        layer.keys = layer.keys[:, :, :-w, :].contiguous()
        layer.values = layer.values[:, :, :-w, :].contiguous()
    cache._seen -= w


@torch.inference_mode()
def stream_generate(model, tok, ids, cfg, refresh=False):
    eos = tok.eos_token_id
    if cfg is None:
        past = None
        out = model(ids, use_cache=True)
    else:
        model.init_cache(cfg)
        past = CompressedCache(model.config, cfg)
        out = model(ids, past_key_values=past, use_cache=True)
    past = out.past_key_values
    gen = []
    cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    for step in range(MAX_NEW):
        gen.append(int(cur))
        if int(cur) == eos:
            break
        out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        cur = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        # 생성 중 재압축
        if cfg is not None and len(gen) % CHUNK == 0:
            if refresh:
                # RefreshKV: 프롬프트+생성분 전체를 다시 prefill하여 현재 시점 기준으로
                # 재선택 (이전 라운드에 버려진 토큰도 후보로 복원됨)
                full_ids = torch.cat([ids, torch.tensor([gen], device=ids.device)], dim=1)
                model.init_cache(cfg)
                fresh = CompressedCache(model.config, cfg)
                with torch.inference_mode():
                    o = model(full_ids, past_key_values=fresh, use_cache=True)
                past = o.past_key_values
            elif past.layers[0].keys.shape[2] > cfg["total_budget"]:
                chunk_ids = torch.tensor([gen[-CHUNK:]], device=ids.device)
                crop_last(past, CHUNK)
                past.reset_scorers()      # 재압축마다 스코어러 상태 초기화 (필수)
                model(chunk_ids, past_key_values=past, use_cache=True)
        if len(gen) % 16 == 0:
            text = tok.decode(gen, skip_special_tokens=True)
            body = text.split("</think>")[-1] if "</think>" in text else ("" if "<think>" in text else text)
            if body and (re.search(r"####\s*-?[\d,]+", body) or re.search(r"boxed\{-?[\d,]+", body)):
                break
    return tok.decode(gen, skip_special_tokens=True)


def extract_number(text):
    if "</think>" in text:
        text = text.split("</think>")[-1]
    mb = re.findall(r"boxed\{(-?[\d,]+\.?\d*)\}", text)
    if mb:
        return mb[-1].replace(",", "").rstrip(".")
    m = re.findall(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return m[-1].replace(",", "").rstrip(".")
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return m[-1].replace(",", "").rstrip(".") if m else None


def _worker(wid, gpu, model_name, budget, shard, methods, waits_ab, rq, refresh=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.set_grad_enabled(False)
    model, tok = load_model(model_name)
    dev = next(model.parameters()).device
    for row in shard:
        if "qwen3" in model_name:
            prompt = tok.apply_chat_template([{"role": "user", "content": row["q"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
        else:
            prompt = tok.apply_chat_template([{"role": "user", "content": INSTR + row["q"]}],
                tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        preds = {}
        for m in methods:
            cfg = build_cfg(m, budget, a=waits_ab[0], b=waits_ab[1]) if m != "full" else None
            try:
                text = stream_generate(model, tok, ids, cfg, refresh=refresh)
                ans = extract_number(text)
            except Exception as e:
                ans, text = None, f"ERR {e}"
            preds[m] = {"ans": ans, "correct": bool(ans == row["gt"]), "len": len(text.split())}
        rq.put({"sid": row["sid"], "gt": row["gt"], "prompt_tokens": int(ids.shape[1]), "preds": preds})
    rq.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--methods", default="full,tova,snapkv,h2o,waits")
    ap.add_argument("--waits_action", default="10:16")
    ap.add_argument("--tag", default="")
    ap.add_argument("--refresh", action="store_true", help="RefreshKV 방식: 주기마다 전체 재-prefill")
    args = ap.parse_args()
    from datasets import load_dataset
    test = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for i in range(args.start_idx, min(args.start_idx + args.n_samples, len(test))):
        rows.append({"sid": i, "q": test[i]["question"],
                     "gt": test[i]["answer"].split("####")[-1].strip().replace(",", "")})
    outdir = os.path.join(REPO, "result_txt/analysis/long_decoding")
    out_p = os.path.join(outdir, f"gsm8k_stream_{args.model}{args.tag}.jsonl")
    done = set()
    if os.path.exists(out_p):
        done = {json.loads(l)["sid"] for l in open(out_p)}
    todo = [r for r in rows if r["sid"] not in done]
    print(f"{len(todo)}/{len(rows)} to run", flush=True)
    wab = (float(args.waits_action.split(":")[0]), int(args.waits_action.split(":")[1]))
    gpus = [int(x) for x in args.gpus.split(",")]
    shards = [todo[i::len(gpus)] for i in range(len(gpus))]
    ctx = mp.get_context("spawn")
    rq = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(i, g, args.model, args.budget, shards[i],
                                               args.methods.split(","), wab, rq, args.refresh))
             for i, g in enumerate(gpus)]
    for p in procs:
        p.start()
    alive, n = len(procs), 0
    with open(out_p, "a") as f:
        while alive:
            item = rq.get()
            if item is None:
                alive -= 1
                continue
            f.write(json.dumps(item) + "\n"); f.flush(); n += 1
            if n % 20 == 0:
                print(f"[stream] {n}/{len(todo)}", flush=True)
    for p in procs:
        p.join()
    accs = {}
    for l in open(out_p):
        r = json.loads(l)
        for m, v in r["preds"].items():
            accs.setdefault(m, []).append(v["correct"])
    print("=== stream CoT accuracy ===")
    for m, v in sorted(accs.items()):
        print(f"{m:8s} {100.0*sum(v)/len(v):.1f}% (n={len(v)})")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
