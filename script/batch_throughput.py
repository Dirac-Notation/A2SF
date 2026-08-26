"""R1-Q6/R3-Q2 배치별 throughput: batch {1,4,8}, 8k 프롬프트, 256토큰 생성.
full vs WAITS b128. OOM은 표기 (압축 캐시가 큰 batch를 가능하게 함이 논지).

  python script/batch_throughput.py --model llama3-8b
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

GEN = 256


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
    ap.add_argument("--length", type=int, default=8192)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8])
    args = ap.parse_args()
    model, tok = load_model(args.model)
    torch.cuda.synchronize()
    base_mem = torch.cuda.max_memory_allocated()   # 모델 가중치 등 고정 메모리
    print(f"baseline(weights) = {base_mem/1024**3:.2f} GiB")
    unit = "The quick brown fox jumps over the lazy dog. "
    text = unit * (args.length // 10)
    base = tok(text, return_tensors="pt", truncation=True, max_length=args.length).input_ids
    base = base.to(next(model.parameters()).device)
    print(f"model={args.model} len={base.shape[1]} gen={GEN}")
    print(f"{'batch':>6} {'method':>10} {'agg_tok/s':>10} {'peak_GiB':>9} {'KV_GiB':>8}")
    for B in args.batches:
        ids = base.repeat(B, 1)
        for name, budget in [("full", None), ("waits_b128", 128)]:
            cfg = cfg_for(budget)
            try:
                run(model, tok, ids, cfg, 4)  # warmup
                torch.cuda.reset_peak_memory_stats()
                t1 = run(model, tok, ids, cfg, 1)
                tN = run(model, tok, ids, cfg, GEN)
                peak_b = torch.cuda.max_memory_allocated()
                peak = peak_b / 1024**3
                kv = max(0.0, (peak_b - base_mem) / 1024**3)
                tps = B * (GEN - 1) / max(tN - t1, 1e-9)
                print(f"{B:>6} {name:>10} {tps:10.1f} {peak:9.2f} {kv:8.2f}", flush=True)
            except torch.OutOfMemoryError:
                print(f"{B:>6} {name:>10} {'OOM':>10} {'-':>9}", flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
