"""
TTFT (Time-to-First-Token) benchmark for A2SF compression methods.

Measures prefill+first-decode latency for:
  - Full KV cache (no compression)
  - Heuristic: TOVA / SnapKV-16 / H2O
  - RL (NeuralUCB + MiniAttn): t_encode + t_agent + t_llm

RL overhead = t_encode + t_agent on top of the heuristic LLM time.

Usage (heuristics only):
  python benchmark_ttft.py --model llama3-1b --gpu 0

Usage (with RL model):
  python benchmark_ttft.py --model llama3-1b --gpu 0 \
      --rl_ckpt runs/RL_minattn_v5_maxo/best.pt

Optional:
  --budgets 128 256 512
  --lengths 512 2048 4096 8192 16384
  --n_runs 10
"""
import argparse
import gc
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")   # suppress Python warning noise

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_real_drop.kv_llama import KVLlamaForCausalLM
from utils_real_drop.cache import CompressedKVCache


# ─── helpers ─────────────────────────────────────────────────────────────────

class CompressionConfig(dict):
    """Minimal dict-backed config; attribute reads fall through to dict.get."""
    def __getattr__(self, key):
        return self.get(key)
    def __setattr__(self, key, value):
        self[key] = value


def load_model(model_name: str, device: str) -> Tuple[KVLlamaForCausalLM, AutoTokenizer]:
    with open("config/model2path.json") as f:
        model_path = json.load(f)[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = KVLlamaForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    model.init_cache(None)
    return model, tokenizer


def make_prompt(target_tokens: int, tokenizer) -> Tuple[str, torch.Tensor]:
    """Build a synthetic prompt of ≈target_tokens tokens via binary-search on repetition."""
    unit = "The quick brown fox jumps over the lazy dog. "
    lo, hi = 1, target_tokens * 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        text = unit * mid
        n = tokenizer(text, return_tensors="pt", truncation=True,
                      max_length=target_tokens + 16).input_ids.shape[-1]
        if n <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    text = unit * lo
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=target_tokens).input_ids
    return text, ids


def heuristic_config(method: str, window: int, budget: int) -> CompressionConfig:
    cfg = CompressionConfig()
    cfg["compression_method"] = method
    cfg["observation_window"] = window
    cfg["total_budget"] = budget
    cfg["a"] = 10
    cfg["b"] = window
    cfg["recent_budget"] = 16
    return cfg


# ─── timing ──────────────────────────────────────────────────────────────────

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_llm_generate(
    model: KVLlamaForCausalLM,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    cfg: Optional[CompressionConfig],
    n_runs: int,
    warmup: int = 2,
) -> List[float]:
    """Return list of TTFT times (ms) for LLM.generate(max_new_tokens=1)."""
    times = []
    for i in range(warmup + n_runs):
        model.init_cache(cfg)
        _sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=1,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                num_logits_to_keep=1,
            )
        _sync()
        elapsed = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            times.append(elapsed)
        gc.collect()
    return times


def time_rl_components(
    model: KVLlamaForCausalLM,
    encoder,
    agent,
    text: str,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    budget: int,
    agent_device,
    n_runs: int,
    warmup: int = 2,
) -> Dict[str, List[float]]:
    """
    Returns dict with lists of per-run timings (ms):
      t_encode  — MiniAttnEncoder.encode_context
      t_agent   — NeuralUCBAgent.act
      t_llm     — LLM.generate with sigmoid action config
    """
    enc_times, agent_times, llm_times = [], [], []

    for i in range(warmup + n_runs):
        # 1. MiniAttn encode
        _sync()
        t0 = time.perf_counter()
        state = encoder.encode_context(
            text,
            generation_length=64,
            token_budget=budget,
            metric_type="qa_f1_score",
            dataset="2wikimqa",
        )
        _sync()
        t_enc = (time.perf_counter() - t0) * 1000

        # 2. Agent inference
        state_dev = state.unsqueeze(0).to(agent_device, dtype=torch.float32)
        _sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            (a_val, b_val), _ = agent.act(state_dev)
        _sync()
        t_ag = (time.perf_counter() - t0) * 1000

        # 3. LLM generate with selected sigmoid action
        a = float(a_val.item())
        b = int(round(b_val.item()))
        cfg = CompressionConfig()
        cfg["compression_method"] = "sigmoid"
        cfg["a"] = a
        cfg["b"] = b
        cfg["total_budget"] = budget
        cfg["recent_budget"] = 16

        model.init_cache(cfg)
        _sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=1,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                num_logits_to_keep=1,
            )
        _sync()
        t_llm = (time.perf_counter() - t0) * 1000

        if i >= warmup:
            enc_times.append(t_enc)
            agent_times.append(t_ag)
            llm_times.append(t_llm)

        gc.collect()

    return {"t_encode": enc_times, "t_agent": agent_times, "t_llm": llm_times}


# ─── reporting ────────────────────────────────────────────────────────────────

def _fmt(vals: List[float]) -> str:
    if not vals:
        return "   N/A   "
    return f"{np.mean(vals):7.1f}±{np.std(vals):.1f}"


def print_table(results: Dict, budget: int, lengths: List[int], has_rl: bool):
    """Print TTFT comparison table for one budget."""
    print(f"\n{'='*80}")
    print(f"  Budget = {budget}   (all times in ms, mean±std)")
    print(f"{'='*80}")

    hdr = f"{'Length':>7}  {'Full':>12}  {'TOVA':>12}  {'SnapKV':>12}  {'H2O':>12}"
    if has_rl:
        hdr += f"  {'RL-total':>12}  {'t_encode':>12}  {'t_agent':>10}  {'t_llm(RL)':>12}"
    print(hdr)
    print("-" * (len(hdr)))

    for L in lengths:
        key = (budget, L)
        if key not in results:
            continue
        row = f"{L:>7}"
        row += f"  {_fmt(results[key].get('full', []))}"
        row += f"  {_fmt(results[key].get('tova', []))}"
        row += f"  {_fmt(results[key].get('snapkv', []))}"
        row += f"  {_fmt(results[key].get('h2o', []))}"
        if has_rl:
            rl = results[key].get("rl", {})
            enc = rl.get("t_encode", [])
            ag  = rl.get("t_agent",  [])
            llm = rl.get("t_llm",    [])
            total = [e + a + l for e, a, l in zip(enc, ag, llm)] if enc else []
            row += f"  {_fmt(total)}"
            row += f"  {_fmt(enc)}"
            row += f"  {_fmt(ag)}"
            row += f"  {_fmt(llm)}"
        print(row)

    if has_rl:
        print()
        print("  RL overhead = t_encode + t_agent  (extra latency vs heuristic)")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3-1b")
    ap.add_argument("--gpu", default="0", help="CUDA device index (single GPU)")
    ap.add_argument("--rl_ckpt", default="", help="Path to RL agent checkpoint (.pt)")
    ap.add_argument("--budgets", nargs="+", type=int, default=[128],
                    help="KV cache budgets to benchmark (default: 128)")
    ap.add_argument("--lengths", nargs="+", type=int,
                    default=[512, 2048, 4096, 8192, 16384],
                    help="Prompt token lengths")
    ap.add_argument("--n_runs", type=int, default=10, help="Measurement runs per config")
    ap.add_argument("--warmup", type=int, default=3, help="Warmup runs (discarded)")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(f"[benchmark_ttft] model={args.model}  device={device}")
    print(f"[benchmark_ttft] budgets={args.budgets}  lengths={args.lengths}")
    print(f"[benchmark_ttft] n_runs={args.n_runs}  warmup={args.warmup}")

    # ── load LLM ──
    print("[benchmark_ttft] loading LLM...")
    model, tokenizer = load_model(args.model, device)
    model_device = next(model.parameters()).device

    # ── optionally load RL model ──
    encoder = None
    agent = None
    has_rl = bool(args.rl_ckpt)

    if has_rl:
        print(f"[benchmark_ttft] loading RL model from {args.rl_ckpt}...")
        # Load via A2SFModel to reuse the same LLM (access through model_runner.model)
        # But A2SFModel loads its own copy of the LLM, so we reuse encoder/agent only.
        ckpt = torch.load(args.rl_ckpt, map_location="cpu", weights_only=False)
        arch_config = ckpt.get("arch_config", {}) or {}
        state_dict = ckpt.get("agent_state_dict", ckpt)

        mini_attn_ckpt = str(arch_config.get("mini_attn_ckpt", "") or "")
        if not mini_attn_ckpt or not os.path.exists(mini_attn_ckpt):
            # Try default path
            mini_attn_ckpt = "runs/mini_attn_v5/mini_attn_best.pt"
        print(f"[benchmark_ttft] mini_attn_ckpt={mini_attn_ckpt}")

        from RL.env.mini_attn_encoder import MiniAttnEncoder
        encoder = MiniAttnEncoder(
            target_model=model,
            target_tokenizer=tokenizer,
            mini_attn_ckpt_path=mini_attn_ckpt,
            device=str(model_device),
            encoder_topk=int(arch_config.get("encoder_topk", 16)),
            max_input_length=int(arch_config.get("encoder_max_input_length", 32768)),
            include_hidden_pool=bool(arch_config.get("encoder_include_hidden_pool", True)),
            hidden_pool_window=int(arch_config.get("encoder_hidden_pool_window", 0)),
            feature_mode=str(arch_config.get("encoder_feature_mode", "stats")),
        )
        encoder.eval()

        from RL.agent.neural_ucb_agent import NeuralUCBAgent
        a_values = arch_config["a_values"].to(torch.float32)
        b_values = arch_config["b_values"].to(torch.float32)
        agent = NeuralUCBAgent(
            state_dim=int(arch_config["state_dim"]),
            a_values=a_values,
            b_values=b_values,
            num_metric_types=int(arch_config["num_metric_types"]),
            num_task_types=int(arch_config["num_task_types"]),
            side_dim=int(arch_config["side_dim"]),
            num_heads=int(arch_config["num_heads"]),
            num_hidden_pool=int(arch_config.get("num_hidden_pool", 0)),
            backbone_depth=int(arch_config.get("backbone_depth", 2)),
            dropout=0.0,
            paired_actions=bool(arch_config.get("paired_actions", True)),
            task_cond_head=bool(arch_config.get("task_cond_head", True)),
            task_head_mlp=bool(arch_config.get("task_head_mlp", False)),
            output_activation=str(arch_config.get("output_activation", "sigmoid")),
            num_views=int(arch_config.get("num_views", 2)),
            include_seq_len=bool(arch_config.get("include_seq_len", True)),
        ).to(model_device)
        # Fix shape mismatch for UCB buffers that gained a leading dim in newer code
        model_sd = agent.state_dict()
        fixed_sd = {}
        for k, v in state_dict.items():
            if k in model_sd and v.shape != model_sd[k].shape:
                target_shape = model_sd[k].shape
                if v.ndim == len(target_shape) - 1 and target_shape[0] == 1:
                    v = v.unsqueeze(0)
            fixed_sd[k] = v
        agent.load_state_dict(fixed_sd, strict=False)
        agent.eval()
        print("[benchmark_ttft] RL components loaded.")

    # ── prepare prompts ──
    print("[benchmark_ttft] preparing prompts...")
    prompts = {}  # length -> (text, input_ids, attn_mask)
    for L in args.lengths:
        text, ids = make_prompt(L, tokenizer)
        ids = ids.to(model_device)
        mask = torch.ones_like(ids)
        actual = ids.shape[-1]
        prompts[L] = (text, ids, mask)
        print(f"  target={L:>6}  actual={actual:>6} tokens")

    # ── heuristic configs ──
    heuristic_methods = [
        ("full",   None),
        ("tova",   heuristic_config("snap", 1,     0)),   # budget filled per-run
        ("snapkv", heuristic_config("snap", 16,    0)),
        ("h2o",    heuristic_config("snap", 32768, 0)),
    ]

    # ── run benchmark ──
    results = {}

    for budget in args.budgets:
        # Update heuristic configs with this budget
        for name, cfg in heuristic_methods:
            if cfg is not None:
                cfg["total_budget"] = budget

        for L in args.lengths:
            text, input_ids, attn_mask = prompts[L]
            key = (budget, L)
            results[key] = {}

            print(f"\n[B={budget}  L={L}]")

            for name, cfg in heuristic_methods:
                print(f"  {name:8s} ...", end="", flush=True)
                times = time_llm_generate(model, input_ids, attn_mask, cfg, args.n_runs, args.warmup)
                results[key][name] = times
                print(f" {np.mean(times):.1f}ms")

            if has_rl:
                print(f"  {'RL':8s} ...", end="", flush=True)
                rl_data = time_rl_components(
                    model=model,
                    encoder=encoder,
                    agent=agent,
                    text=text,
                    input_ids=input_ids,
                    attn_mask=attn_mask,
                    budget=budget,
                    agent_device=model_device,
                    n_runs=args.n_runs,
                    warmup=args.warmup,
                )
                results[key]["rl"] = rl_data
                total = np.mean(rl_data["t_encode"]) + np.mean(rl_data["t_agent"]) + np.mean(rl_data["t_llm"])
                print(f" {total:.1f}ms  (enc={np.mean(rl_data['t_encode']):.1f}  ag={np.mean(rl_data['t_agent']):.1f}  llm={np.mean(rl_data['t_llm']):.1f})")

    # ── print tables ──
    for budget in args.budgets:
        print_table(results, budget, args.lengths, has_rl)

    print(f"\n[benchmark_ttft] done.")


if __name__ == "__main__":
    main()
