"""UPPER-BOUND input: TRUE decode-attention feature per LB prompt.
Prefill (sdpa) -> teacher-force the FULL-CACHE response (eager, output_attentions) ->
decode-time attention each context token receives -> SAME 20-d feature as the prefill
posfeat (16-bin rel-pos histogram + sink/recency/com/spread). This is the "future"
the generation actually attended to. Compare against prefill posfeat to test whether a
future-attention proxy makes best-(a,b) predictable.

  python experiments/oracle_decode_feature.py --gpus 0,1,2,3,4,5,6,7 --per_ds 45 \
      --out runs/states/oracle_decode_feat.pt
"""
import argparse, json, os, sys, time
import multiprocessing as mp
import numpy as np, torch
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))
from transfer_input_extract import feature, fmt   # reuse identical 20-d feature + prompt fmt

NBINS = 16; MAXLEN = 6000; MAX_PRED_TOK = 128; _TF_BATCH = 16

def _set_attn(model, impl):
    model.config._attn_implementation = impl
    for mod in model.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
            mod.config._attn_implementation = impl
        if hasattr(mod, "_attn_implementation"):
            mod._attn_implementation = impl

def decode_imp(model, tok, ids, pred_text, dev):
    S = ids.shape[1]; cfg = model.config
    _set_attn(model, "sdpa"); model.init_cache(None)
    with torch.no_grad(): pkv = model(ids, use_cache=True).past_key_values
    pred_ids = tok(pred_text, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)[:, :MAX_PRED_TOK]
    if pred_ids.size(1) == 0: return None
    answer = torch.zeros(cfg.num_hidden_layers, cfg.num_attention_heads, S, dtype=torch.float32)
    _set_attn(model, "eager")
    import warnings
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for st in range(0, pred_ids.size(1), _TF_BATCH):
            out = model(pred_ids[:, st:st+_TF_BATCH], past_key_values=pkv,
                        use_cache=True, output_attentions=True)
            pkv = out.past_key_values
            if out.attentions is not None:
                for li, al in enumerate(out.attentions):
                    if al is None: continue
                    answer[li] += al[0, :, :, :S].float().sum(dim=1).cpu()
    a = answer.clamp(min=0); a = a / (a.sum(-1, keepdim=True) + 1e-12)
    return a.mean(dim=1).mean(dim=0)   # head-avg then layer-avg -> (S,) decode importance

def _worker(wid, gpu, model_name, tq, rq):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu); torch.set_grad_enabled(False)
    from utils import load_model
    model, tok = load_model(model_name); dev = model.device
    while True:
        s = tq.get()
        if s is None: break
        key, prompt, ds, pred = s
        try:
            ids = tok(fmt(prompt, ds, model_name), truncation=True, max_length=MAXLEN, return_tensors="pt").input_ids.to(dev)
            imp = decode_imp(model, tok, ids, pred, dev)
            f = feature(imp) if imp is not None else np.zeros(NBINS+4, "float32")
        except Exception as e:
            f = np.zeros(NBINS+4, "float32")
        rq.put((key, ds, f)); torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3-1b"); ap.add_argument("--gpus", default="0")
    ap.add_argument("--per_ds", type=int, default=45)
    ap.add_argument("--full_pred_dir", default="result_txt/pred/128/llama3-1b_full")
    ap.add_argument("--out", default="runs/states/oracle_decode_feat.pt"); a = ap.parse_args()
    gpu_ids = [int(g) for g in a.gpus.split(",") if g != ""]
    ix = torch.load(os.path.join(REPO, "runs/fast_lb_eval/index_llama3-1b_128.pt"), map_location="cpu")
    datasets = sorted(k[:-7] for k in ix if k.endswith("/scores"))
    tasks = []
    for ds in datasets:
        lines = open(os.path.join(REPO, f"datasets/longbench/{ds}.jsonl")).readlines()
        preds = [json.loads(l).get("pred", "") for l in open(os.path.join(REPO, a.full_pred_dir, f"{ds}.jsonl"))]
        step = max(1, len(lines) // a.per_ds); c = 0
        for j in range(0, len(lines), step):
            if c >= a.per_ds or j >= len(preds): break
            tasks.append((f"{ds}:{j}", json.loads(lines[j])["input_prompt"], ds, preds[j])); c += 1
    print(f"{len(tasks)} prompts, gpus={gpu_ids}", flush=True)
    ctx = mp.get_context("spawn"); tq, rq = ctx.Queue(), ctx.Queue()
    for t in tasks: tq.put(t)
    for _ in gpu_ids: tq.put(None)
    procs = [ctx.Process(target=_worker, args=(i, g, a.model, tq, rq)) for i, g in enumerate(gpu_ids)]
    [p.start() for p in procs]
    res = {}; t0 = time.time()
    for n in range(len(tasks)):
        key, ds, f = rq.get(); res[key] = (ds, f)
        if (n+1) % 50 == 0: print(f"  {n+1}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    [p.join(timeout=20) for p in procs]
    out = {"keys": list(res.keys()), "feat": torch.tensor(np.stack([res[k][1] for k in res])), "feat_dim": NBINS+4}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); torch.save(out, a.out)
    print(f"saved {a.out}  feat {out['feat'].shape}", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True); main()
