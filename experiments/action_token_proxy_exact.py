"""EXACT per-(prompt, action) kept-token proxy: FULL queries + PER-KV-HEAD selection,
using the production `_accumulate_scores` + `TokenSelector` + `WaitsScorer` verbatim.

For each prompt: prefill (capture full per-layer post-rotary Q, and K from cache); then
for each action build WaitsScorer(a,b), run the production scoring (all queries, GQA
per-kv-head) and TokenSelector (top-(budget-recent) by score + recent tail) PER (layer,
kv-head). proxy[action] = mean over all (layer, kv_head) of the 16-bin relative-position
histogram of the kept `budget` keys.

  python experiments/action_token_proxy_exact.py --gpus 0,1,2,3,4,5,6,7 --per_ds 40 \
      --out runs/states/action_token_proxy_exact.pt
"""
import argparse, json, os, sys, math, time
import multiprocessing as mp
import numpy as np, torch
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from utils_real_drop.compress import _accumulate_scores
from utils_real_drop.scorers.waits import WaitsScorer
from utils_real_drop.selectors.token import TokenSelector
from utils_real_drop.selectors.base import uniform_budgets

NBINS = 16; BUDGET = 128; RECENT = 16; MAXLEN = 6000
A_VALS = [0,.01,.01,.01,.01,.1,.1,.1,.1,10,10,10,10]
B_VALS = [1,1,16,32,128,1,16,32,128,1,16,32,128]
_NO_CHAT = ["trec","triviaqa","samsum","lsht","lcc","repobench-p"]
def fmt(p, ds, m="llama3-1b"):
    return f"[INST]{p}[/INST]" if (str(ds).lower() not in _NO_CHAT and "llama" in m) else p

def proxies_exact(model, ids, dev):
    S = ids.shape[1]; cfg = model.config
    nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // nh
    store = {}; hooks = []
    def mk(i):
        def hk(mod, args, kw):
            h = kw.get("hidden_states"); h = h if h is not None else (args[0] if args else None)
            if h is not None and h.size(1) > 1: store[i] = h.detach()
        return hk
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_pre_hook(mk(i), with_kwargs=True))
    model.init_cache(None)
    with torch.no_grad(): pkv = model(ids, use_cache=True).past_key_values
    for h in hooks: h.remove()
    posids = torch.arange(S, device=dev).unsqueeze(0)
    relbin = (torch.arange(S, device=dev) * NBINS // max(1, S)).clamp(0, NBINS-1)
    selector = TokenSelector(uniform_budgets(cfg.num_hidden_layers, BUDGET), recent_budget=RECENT)
    # precompute per-layer full post-rotary Q and K
    QK = []
    for i in range(cfg.num_hidden_layers):
        at = model.model.layers[i].self_attn
        q = at.q_proj(store[i]).view(1, S, nh, hd).transpose(1, 2)   # (1,nh,S,hd)
        cos, sin = model.model.rotary_emb(q, posids)
        qr, _ = apply_rotary_pos_emb(q, q, cos, sin)
        QK.append((qr, pkv.layers[i].keys)); store[i] = None
    out = np.zeros((13, NBINS), "float32")
    for ai, (a, b) in enumerate(zip(A_VALS, B_VALS)):
        hist = torch.zeros(NBINS, device=dev); cnt = 0
        for i in range(cfg.num_hidden_layers):
            qr, k = QK[i]
            scorer = WaitsScorer(nkv, a=a, b=b)
            scorer.prepare_prefill(S, dev, qr.dtype)
            scores = _accumulate_scores(qr, k, scorer, nkv, hd, None)   # (1,nkv,S) ALL queries
            idx = selector.select(i, scores, S)                        # (1,nkv,budget) per kv-head
            for h in range(nkv):
                hh = relbin[idx[0, h]]
                hist += torch.bincount(hh, minlength=NBINS).float()
                cnt += 1
        out[ai] = (hist / max(1, hist.sum())).cpu().numpy()
    return out

def _worker(wid, gpu, model_name, tq, rq):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu); torch.set_grad_enabled(False)
    from utils import load_model
    model, tok = load_model(model_name); dev = model.device
    while True:
        s = tq.get()
        if s is None: break
        key, prompt, ds = s
        try:
            ids = tok(fmt(prompt, ds, model_name), truncation=True, max_length=MAXLEN, return_tensors="pt").input_ids.to(dev)
            px = proxies_exact(model, ids, dev)
        except Exception as e:
            px = np.zeros((13, NBINS), "float32")
        rq.put((key, ds, px)); torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3-1b"); ap.add_argument("--gpus", default="0")
    ap.add_argument("--per_ds", type=int, default=40)
    ap.add_argument("--out", default="runs/states/action_token_proxy_exact.pt"); a = ap.parse_args()
    gpu_ids = [int(g) for g in a.gpus.split(",") if g != ""]
    ix = torch.load(os.path.join(REPO, "runs/fast_lb_eval/index_llama3-1b_128.pt"), map_location="cpu")
    datasets = sorted(k[:-7] for k in ix if k.endswith("/scores"))
    tasks = []
    for ds in datasets:
        lines = open(os.path.join(REPO, f"datasets/longbench/{ds}.jsonl")).readlines()
        step = max(1, len(lines) // a.per_ds); c = 0
        for j in range(0, len(lines), step):
            if c >= a.per_ds: break
            tasks.append((f"{ds}:{j}", json.loads(lines[j])["input_prompt"], ds)); c += 1
    print(f"{len(tasks)} prompts, gpus={gpu_ids}", flush=True)
    ctx = mp.get_context("spawn"); tq, rq = ctx.Queue(), ctx.Queue()
    for t in tasks: tq.put(t)
    for _ in gpu_ids: tq.put(None)
    procs = [ctx.Process(target=_worker, args=(i, g, a.model, tq, rq)) for i, g in enumerate(gpu_ids)]
    [p.start() for p in procs]
    res = {}; t0 = time.time()
    for n in range(len(tasks)):
        key, ds, px = rq.get(); res[key] = (ds, px)
        if (n+1) % 50 == 0: print(f"  {n+1}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    [p.join(timeout=20) for p in procs]
    out = {"keys": list(res.keys()), "ds": [res[k][0] for k in res],
           "proxy": torch.tensor(np.stack([res[k][1] for k in res])),
           "A_VALS": A_VALS, "B_VALS": B_VALS, "NBINS": NBINS}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); torch.save(out, a.out)
    print(f"saved {a.out}  proxy shape {out['proxy'].shape}", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True); main()
