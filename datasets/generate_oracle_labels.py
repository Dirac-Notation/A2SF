#!/usr/bin/env python3
"""Generate future-aware oracle per-token importance labels for encoder training.

The oracle (per reward-design discussion): the KV-cache selection made *knowing
the future*. Concretely, for each prompt:
  1. reuse the full-cache greedy generation `full_cache_pred` (already in
     common.jsonl — NO re-decode), as the "future" text;
  2. prefill the prompt (sdpa, efficient on long context) -> past_kv;
  3. teacher-force the generated tokens with output_attentions=True (only the few
     generated queries materialise attention, so this is cheap even at S~32k) and
     accumulate, per layer/head, the attention mass each context token receives;
  4. that accumulated attention IS the oracle importance: which context tokens the
     generation actually needed.

This is the verbatim `teacher_forcing_answer_score` numerics from
experiments/paper_figures/observations/common.py (the paper's §3 oracle), reused
here as a dense per-token supervision target for the MiniCrossAttn encoder
(which predicts mini_attn(embeds)[0] : (L,) per-token importance).

Output (one file per sample, resumable):
  <outdir>/<sid>.pt = {
      "sample_id": int, "seq_len": int, "n_pred": int,
      "imp_per_layer": fp16 (n_layers, seq_len)  # head-averaged, L1-normalised per layer
  }
Aggregate to (seq_len,) at train time via imp_per_layer.mean(0), or pick layers.

  python datasets/generate_oracle_labels.py \
      --input datasets/training/scored/faithful_v1/common.jsonl \
      --outdir runs/oracle_labels/llama3-1b_faithful \
      --model llama3-1b --gpus 0,1,2,3,4,5,6,7 --max_input_length 32768
"""
import argparse, json, os, sys, time
import multiprocessing as mp
from typing import Dict, List

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_TF_BATCH = 16  # tokens per teacher-forcing step (matches observations/common.py)

# verbatim from datasets/generate_sigmoid_dataset.py (copied to avoid the local
# `datasets/` dir colliding with the HuggingFace `datasets` package on import).
_NO_CHAT = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]
def _format_prompt(prompt: str, dataset_name: str, model_name: str) -> str:
    if str(dataset_name or "").strip().lower() not in _NO_CHAT:
        if "llama" in str(model_name).lower():
            return f"[INST]{prompt}[/INST]"
    return prompt


def _model_path(model_name: str) -> str:
    with open(os.path.join(REPO_ROOT, "config", "model2path.json")) as f:
        return json.load(f)[model_name]


def _set_attn(model, impl: str):
    """Flip attention implementation at runtime.

    v5's sdpa silently drops output_attentions (returns None), unlike 4.x which
    fell back to eager. We need real attention weights only for the short
    teacher-forcing queries, so we prefill under sdpa (no S*S materialisation on
    long context) and switch to eager just for the teacher-forcing forwards
    (queries <= _TF_BATCH, so the attention matrix stays tiny).
    """
    model.config._attn_implementation = impl
    for mod in model.modules():
        if hasattr(mod, "config"):
            mod.config._attn_implementation = impl
        if hasattr(mod, "_attn_implementation"):
            mod._attn_implementation = impl


def teacher_forcing_answer_score(model, tokenizer, pred_text, past_kv, seq_len, device):
    """(n_layers, n_heads, seq_len) decode-time attention each ctx token receives.

    Verbatim numerics from experiments/paper_figures/observations/common.py.
    """
    pred_ids = tokenizer(pred_text, add_special_tokens=False,
                         return_tensors="pt").input_ids.to(device)
    n_pred = pred_ids.size(1)
    if n_pred == 0:
        return None, 0
    cfg = model.config
    answer = torch.zeros(cfg.num_hidden_layers, cfg.num_attention_heads, seq_len,
                         dtype=torch.float32)
    import warnings
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for start in range(0, n_pred, _TF_BATCH):
            chunk = pred_ids[:, start:start + _TF_BATCH]
            out = model(chunk, past_key_values=past_kv,
                        use_cache=True, output_attentions=True)
            past_kv = out.past_key_values
            if out.attentions is not None:
                for li, attn_l in enumerate(out.attentions):
                    if attn_l is None:
                        continue
                    a = attn_l[0, :, :, :seq_len].float()  # (n_heads, K, seq_len)
                    answer[li] += a.sum(dim=1).cpu()
            del out
    return answer, n_pred


def _reduce(answer: torch.Tensor) -> torch.Tensor:
    """(L, H, S) accumulated attention -> (L, S) head-averaged, L1-normalised/layer."""
    a = answer.clamp(min=0.0)
    a = a / (a.sum(-1, keepdim=True) + 1e-12)  # per (L,H) distribution over S
    return a.mean(dim=1)                        # head-average -> (L, S)


def _worker(worker_id, gpu_id, model_name, outdir, max_len, tq, rq):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_grad_enabled(False)
    mp_path = _model_path(model_name)
    print(f"[worker {worker_id}] gpu={gpu_id} loading {mp_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(mp_path)
    model = AutoModelForCausalLM.from_pretrained(
        mp_path, torch_dtype=torch.bfloat16,
        device_map={"": "cuda"}, attn_implementation="sdpa",
    ).eval()
    device = model.device
    try:
        while True:
            sample = tq.get()
            if sample is None:
                break
            sid = int(sample["sample_id"])
            pred_text = str(sample.get("full_cache_pred") or "")
            if not pred_text.strip():
                rq.put((sid, 0, 0, 0.0, "empty_pred")); continue
            prompt = _format_prompt(str(sample["input_prompt"]),
                                    str(sample.get("dataset") or ""), model_name)
            enc = tok(prompt, truncation=True, max_length=int(max_len),
                      return_tensors="pt")
            input_ids = enc.input_ids.to(device)
            seq_len = int(input_ids.shape[-1])
            t0 = time.time()
            # prefill via sdpa (no attentions -> no S*S blow-up on long context)
            _set_attn(model, "sdpa")
            out = model(input_ids, use_cache=True, output_attentions=False)
            past_kv = out.past_key_values
            del out
            # teacher-forcing under eager so output_attentions is honoured
            _set_attn(model, "eager")
            answer, n_pred = teacher_forcing_answer_score(
                model, tok, pred_text, past_kv, seq_len, device)
            if answer is None:
                rq.put((sid, seq_len, 0, time.time() - t0, "no_pred_tok")); continue
            imp = _reduce(answer).to(torch.float16).contiguous()  # (L, S)
            torch.save({"sample_id": sid, "seq_len": seq_len, "n_pred": int(n_pred),
                        "imp_per_layer": imp},
                       os.path.join(outdir, f"{sid}.pt"))
            del answer, past_kv
            rq.put((sid, seq_len, int(n_pred), time.time() - t0, "ok"))
    except Exception as exc:
        import traceback
        rq.put(("__error__", f"worker {worker_id} gpu={gpu_id}: {exc}\n{traceback.format_exc()}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="common.jsonl with full_cache_pred")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default="llama3-1b")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--max_input_length", type=int, default=32768)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    gpu_ids = [int(g) for g in a.gpus.split(",") if g != ""]

    rows: List[Dict] = []
    with open(a.input) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    done = {int(fn[:-3]) for fn in os.listdir(a.outdir) if fn.endswith(".pt")}
    todo = [r for r in rows if int(r["sample_id"]) not in done]
    todo.sort(key=lambda s: -int(s.get("length", 0)))  # longest first for balance
    print(f"total={len(rows)}  done={len(done)}  todo={len(todo)}  gpus={gpu_ids}")
    if not todo:
        print("nothing to do"); return

    ctx = mp.get_context("spawn")
    tq, rq = ctx.Queue(), ctx.Queue()
    for s in todo:
        tq.put(s)
    for _ in gpu_ids:
        tq.put(None)
    procs = []
    for wid, g in enumerate(gpu_ids):
        p = ctx.Process(target=_worker,
                        args=(wid, g, a.model, a.outdir, a.max_input_length, tq, rq))
        p.start(); procs.append(p)

    t_start = time.time()
    for i in range(len(todo)):
        item = rq.get()
        if item[0] == "__error__":
            for p in procs: p.terminate()
            raise RuntimeError(item[1])
        sid, S, npred, dur, status = item
        done_n = i + 1
        rate = done_n / max(1e-6, time.time() - t_start)
        eta = (len(todo) - done_n) / max(rate, 1e-6)
        print(f"[oracle] {done_n}/{len(todo)} sid={sid} S={S} npred={npred} "
              f"{status} dur={dur:.1f}s rate={rate:.2f}/s eta={eta/60:.1f}m", flush=True)
    for p in procs:
        p.join(timeout=30)
        if p.is_alive(): p.terminate()
    print(f"done -> {a.outdir}  ({len(done) + len(todo)} samples)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
