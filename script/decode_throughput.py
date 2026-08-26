"""§2 보강: decode throughput(tokens/s) + peak memory, full vs 압축 budget별.
단일 GPU. TTFT(1토큰)와 512토큰 총시간의 차로 decode 구간 속도 산출.

  python script/decode_throughput.py --model llama3-1b --lengths 8192 32768
출력: stdout 표 (logs/rebuttal/throughput_<model>.log 로 리다이렉트해 사용)
"""
import argparse
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from utils import CompressionConfig, load_model  # noqa: E402

GEN = 512


def cfg_for(budget):
    if budget is None:
        return None
    c = CompressionConfig()
    c["compression_method"] = "waits"
    c["a"] = 10.0
    c["b"] = 16
    c["observation_window"] = 16
    c["total_budget"] = int(budget)
    c["recent_budget"] = 16
    return c


def run(model, tok, ids, cfg, n_new):
    model.init_cache(cfg)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        model.generate(input_ids=ids, max_new_tokens=n_new, min_new_tokens=n_new,
                       do_sample=False, num_beams=1, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", nargs="+", type=int, default=[8192, 32768])
    ap.add_argument("--budgets", nargs="+", type=int, default=[128, 256, 512])
    args = ap.parse_args()
    model, tok = load_model(args.model)
    unit = "The quick brown fox jumps over the lazy dog. "
    print(f"model={args.model} gen={GEN}")
    print(f"{'len':>7} {'method':>10} {'ttft_s':>8} {'decode_tok/s':>13} {'peak_GiB':>9}")
    for L in args.lengths:
        text = unit * (L // 10)
        ids = tok(text, return_tensors="pt", truncation=True, max_length=L).input_ids
        ids = ids.to(next(model.parameters()).device)
        for name, budget in [("full", None)] + [(f"waits_b{b}", b) for b in args.budgets]:
            cfg = cfg_for(budget)
            run(model, tok, ids, cfg, 4)  # warmup
            torch.cuda.reset_peak_memory_stats()
            t1 = run(model, tok, ids, cfg, 1)
            tN = run(model, tok, ids, cfg, GEN)
            peak = torch.cuda.max_memory_allocated() / 1024**3
            tps = (GEN - 1) / max(tN - t1, 1e-9)
            print(f"{ids.shape[1]:>7} {name:>10} {t1:8.2f} {tps:13.1f} {peak:9.2f}", flush=True)


if __name__ == "__main__":
    main()
