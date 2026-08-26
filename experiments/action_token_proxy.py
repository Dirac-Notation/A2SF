"""Per-(prompt, action) KEPT-TOKEN DISTRIBUTION proxy.

For each prompt, reconstruct prefill attention from a spread of query positions
(evenly-spaced across the sequence + dense recent window), aggregate over
layers/heads, then for each of the 13 (a,b) actions apply its sigmoid query
weighting -> accumulated per-key score -> top-`budget` selection -> relative
position histogram (NBINS). proxy[action] = NBINS-vector = WHAT TOKENS the action
keeps (geometric effect), index-free.

  python experiments/action_token_proxy.py --gpus 0,1,2,3,4,5,6,7 --per_ds 40 \
      --out runs/states/action_token_proxy.pt
"""
import argparse, json, os, sys, math, time
import multiprocessing as mp
import numpy as np, torch
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

NBINS = 16; BUDGET = 128; N_SPREAD = 80; N_RECENT = 80; MAXLEN = 8000
A_VALS = [0,.01,.01,.01,.01,.1,.1,.1,.1,10,10,10,10]
B_VALS = [1,1,16,32,128,1,16,32,128,1,16,32,128]
_NO_CHAT = ["trec","triviaqa","samsum","lsht","lcc","repobench-p"]
def fmt(p, ds, m="llama3-1b"):
    return f"[INST]{p}[/INST]" if (str(ds).lower() not in _NO_CHAT and "llama" in m) else p

def agg_attn(model, ids, dev):
    """Return (Qpos[Q], A_agg[Q,S]) aggregated attention summed over layers & heads
    for a spread+recent set of query positions."""
    S = ids.shape[1]; cfg = model.config
    nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // nh; g = nh // nkv
    # query positions: spread + recent (dedup, sorted) — known from S
    spread = np.linspace(0, S-1, N_SPREAD).astype(int)
    recent = np.arange(max(0, S-N_RECENT), S)
    Q = np.unique(np.concatenate([spread, recent])); Qt = torch.tensor(Q, device=dev)
    Qi = torch.tensor(Q, device=dev)
    # capture ONLY sampled-query hidden states per layer (memory-light)
    store = {}
    hooks = []
    def mk(i):
        def hk(mod, args, kw):
            h = kw.get("hidden_states");  h = h if h is not None else (args[0] if args else None)
            if h is not None and h.size(1) > 1: store[i] = h[:, Qi, :].detach()
        return hk
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_pre_hook(mk(i), with_kwargs=True))
    model.init_cache(None)
    with torch.no_grad(): pkv = model(ids, use_cache=True).past_key_values
    for h in hooks: h.remove()
    A_agg = torch.zeros(len(Q), S)
    for i in range(cfg.num_hidden_layers):
        h = store[i]                                # (1,|Q|,H)
        q = model.model.layers[i].self_attn.q_proj(h).view(1, len(Q), nh, hd).transpose(1, 2)
        cos, sin = model.model.rotary_emb(q, Qt.unsqueeze(0))
        qr, _ = apply_rotary_pos_emb(q, q, cos, sin)
        k = pkv.layers[i].keys                       # (1,nkv,S,hd)
        sc = torch.matmul(qr.view(1, nkv, g, len(Q), hd), k.unsqueeze(2).transpose(-1, -2)) / math.sqrt(hd)
        sc = sc.view(1, nh, len(Q), S)
        mask = (torch.arange(S, device=dev).view(1, 1, 1, S) <= Qt.view(1, 1, len(Q), 1))
        sc = sc.masked_fill(~mask, float("-inf"))
        a = torch.softmax(sc.float(), -1)[0]         # (nh,|Q|,S)
        A_agg += a.sum(0).cpu()
        store[i] = None
    return Q, A_agg, S                                # A_agg (|Q|,S)

def proxies(Q, A_agg, S):
    """For each action -> NBINS relative-position histogram of kept BUDGET keys."""
    Qf = Q.astype(np.float64); out = np.zeros((13, NBINS), "float32")
    relbin = (np.arange(S) * NBINS // max(1, S)).clip(0, NBINS-1)
    A = A_agg.numpy().astype(np.float64)             # (|Q|,S)
    for ai, (a, b) in enumerate(zip(A_VALS, B_VALS)):
        w = 1.0 / (1.0 + np.exp(-a * (Qf - (S - b - 0.5))))   # sigmoid weight per query
        score = (w[:, None] * A).sum(0)              # (S,)
        keep = np.argsort(-score)[:min(BUDGET, S)]
        h = np.zeros(NBINS)
        for k in keep: h[relbin[k]] += 1
        out[ai] = h / max(1, len(keep))
    return out                                       # (13,NBINS)

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
            Q, A_agg, S = agg_attn(model, ids, dev)
            px = proxies(Q, A_agg, S)
        except Exception as e:
            px = np.zeros((13, NBINS), "float32")
        rq.put((key, ds, px)); torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3-1b"); ap.add_argument("--gpus", default="0")
    ap.add_argument("--per_ds", type=int, default=40)
    ap.add_argument("--out", default="runs/states/action_token_proxy.pt"); a = ap.parse_args()
    gpu_ids = [int(g) for g in a.gpus.split(",") if g != ""]
    ix = torch.load(os.path.join(REPO, "runs/fast_lb_eval/index_llama3-1b_128.pt"), map_location="cpu")
    datasets = sorted(k[:-7] for k in ix if k.endswith("/scores"))
    tasks = []
    for ds in datasets:
        lines = open(os.path.join(REPO, f"datasets/longbench/{ds}.jsonl")).readlines()
        step = max(1, len(lines) // a.per_ds)
        for j in range(0, len(lines), step):
            if sum(1 for t in tasks if t[2] == ds) >= a.per_ds: break
            tasks.append((f"{ds}:{j}", json.loads(lines[j])["input_prompt"], ds))
    print(f"{len(tasks)} prompts, gpus={gpu_ids}", flush=True)
    ctx = mp.get_context("spawn"); tq, rq = ctx.Queue(), ctx.Queue()
    for t in tasks: tq.put(t)
    for _ in gpu_ids: tq.put(None)
    procs = [ctx.Process(target=_worker, args=(i, g, a.model, tq, rq)) for i, g in enumerate(gpu_ids)]
    [p.start() for p in procs]
    res = {}; t0 = time.time()
    for n in range(len(tasks)):
        key, ds, px = rq.get(); res[key] = (ds, px)
        if (n+1) % 100 == 0: print(f"  {n+1}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    [p.join(timeout=20) for p in procs]
    out = {"keys": list(res.keys()), "ds": [res[k][0] for k in res],
           "proxy": torch.tensor(np.stack([res[k][1] for k in res])),   # (N,13,NBINS)
           "A_VALS": A_VALS, "B_VALS": B_VALS, "NBINS": NBINS}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); torch.save(out, a.out)
    print(f"saved {a.out}  proxy shape {out['proxy'].shape}", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True); main()
