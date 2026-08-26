"""Extract a TRANSFERABLE (distribution-invariant) positional-profile input from
prefill attention: relative-position histogram + sink/recency mass + center-of-mass
+ spread. Length-normalised => same meaning across faithful/LB.

  faithful:  python experiments/transfer_input_extract.py --mode faithful --out runs/states/posfeat_faithful.pt --gpus 0,1,2,3,4,5,6,7
  lb:        python experiments/transfer_input_extract.py --mode lb       --out runs/fast_lb_eval/posfeat_lb.pt --gpus 0,1,2,3,4,5,6,7
"""
import argparse, json, os, sys, math, time
import multiprocessing as mp
import numpy as np, torch
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments/paper_figures/observations"))
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

NBINS = 16; MAXW = 256
DS_METRIC = {"hotpotqa":"qa_f1_score","2wikimqa":"qa_f1_score","multifieldqa_en":"qa_f1_score",
    "qasper":"qa_f1_score","narrativeqa":"qa_f1_score","triviaqa":"qa_f1_score","gov_report":"rouge_score",
    "qmsum":"rouge_score","multi_news":"rouge_score","samsum":"rouge_score","trec":"classification_score",
    "passage_retrieval_en":"retrieval_score","passage_count":"count_score","lcc":"code_sim_score","repobench-p":"code_sim_score"}
_NO_CHAT = ["trec","triviaqa","samsum","lsht","lcc","repobench-p"]
def fmt(p, ds, m="llama3-1b"):
    return f"[INST]{p}[/INST]" if (str(ds).lower() not in _NO_CHAT and "llama" in m) else p

def windowed_imp(model, ids, dev):
    """per-key importance (S,) = sum over layers/heads/last-W-queries of prefill attention."""
    import common as C  # AttentionCollector
    col = C.AttentionCollector(model, MAXW); S = ids.shape[1]; col.reset(S)
    cfg = model.config; nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // nh; g = nh // nkv
    model.init_cache(None)
    with torch.no_grad(): pkv = model(ids, use_cache=True).past_key_values
    imp = torch.zeros(S)
    for i in range(cfg.num_hidden_layers):
        at = model.model.layers[i].self_attn; h = col._window_inputs[i]; W = h.size(1)
        q = at.q_proj(h).view(1, W, nh, hd).transpose(1, 2)
        cos, sin = model.model.rotary_emb(q, torch.arange(S-W, S, device=dev).unsqueeze(0))
        qr, _ = apply_rotary_pos_emb(q, q, cos, sin); k = pkv.layers[i].keys
        sc = torch.matmul(qr.view(1, nkv, g, W, hd), k.unsqueeze(2).transpose(-1, -2)) / math.sqrt(hd)
        sc = sc.view(1, nh, W, S)
        kp = torch.arange(S, device=dev); qp = torch.arange(S-W, S, device=dev)
        sc.masked_fill_(~(kp.unsqueeze(0) <= qp.unsqueeze(1)).unsqueeze(0).unsqueeze(0), float("-inf"))
        a = torch.softmax(sc.float(), -1)[0]          # (nh,W,S)
        imp += a.sum(dim=(0, 1)).cpu()
    col.remove_hooks()
    return imp  # (S,)

def feature(imp):
    S = imp.shape[0]; p = imp.clamp(min=0); p = p / (p.sum() + 1e-9)
    pos = np.arange(S)
    # relative-position histogram (NBINS bins over [0,S))
    bins = (pos * NBINS // max(1, S)).clip(0, NBINS-1)
    hist = np.zeros(NBINS)
    pv = p.numpy()
    for b in range(NBINS): hist[b] = pv[bins == b].sum()
    sink = pv[:4].sum()
    recency = pv[int(S*0.95):].sum()
    rel = pos / max(1, S-1)
    com = float((pv * rel).sum())
    spread = float(np.sqrt(((rel - com)**2 * pv).sum()))
    return np.concatenate([hist, [sink, recency, com, spread]]).astype("float32")

def _worker(wid, gpu, model_name, mode, tq, rq):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu); torch.set_grad_enabled(False)
    from utils import load_model
    model, tok = load_model(model_name); dev = model.device
    try:
        while True:
            s = tq.get()
            if s is None: break
            key, prompt, ds = s
            ids = tok(fmt(prompt, ds, model_name), truncation=True, max_length=32768, return_tensors="pt").input_ids.to(dev)
            try:
                f = feature(windowed_imp(model, ids, dev))
            except Exception:
                f = np.zeros(NBINS+4, "float32")
            rq.put((key, ds, f)); torch.cuda.empty_cache()
    except Exception as e:
        import traceback; rq.put(("__err__", f"w{wid}: {e}\n{traceback.format_exc()}"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["faithful","lb"], required=True)
    ap.add_argument("--model", default="llama3-1b"); ap.add_argument("--gpus", default="0")
    ap.add_argument("--out", required=True); a = ap.parse_args()
    gpu_ids = [int(g) for g in a.gpus.split(",") if g!=""]
    tasks = []
    if a.mode == "faithful":
        ntr = sum(1 for _ in open(os.path.join(REPO,"datasets/training/scored/faithful_v1/train.jsonl")))
        for off, fn in [(0,"train"),(ntr,"validation")]:
            for j,l in enumerate(open(os.path.join(REPO,f"datasets/training/scored/faithful_v1/{fn}.jsonl"))):
                d=json.loads(l); tasks.append((off+j, d["input_prompt"], d.get("dataset","")))
    else:
        ix = torch.load(os.path.join(REPO,"runs/fast_lb_eval/index_llama3-1b_128.pt"), map_location="cpu")
        for ds in sorted(k[:-7] for k in ix if k.endswith("/scores")):
            for j,l in enumerate(open(os.path.join(REPO,f"datasets/longbench/{ds}.jsonl"))):
                tasks.append((f"{ds}:{j}", json.loads(l)["input_prompt"], ds))
    print(f"mode={a.mode}: {len(tasks)} prompts, gpus={gpu_ids}", flush=True)
    ctx = mp.get_context("spawn"); tq, rq = ctx.Queue(), ctx.Queue()
    for t in tasks: tq.put(t)
    for _ in gpu_ids: tq.put(None)
    procs = [ctx.Process(target=_worker, args=(i,g,a.model,a.mode,tq,rq)) for i,g in enumerate(gpu_ids)]
    [p.start() for p in procs]
    feats = {}; t0=time.time()
    for n in range(len(tasks)):
        item = rq.get()
        if item[0]=="__err__": [p.terminate() for p in procs]; raise RuntimeError(item[1])
        key, ds, f = item; feats[key]=(ds,f)
        if (n+1)%200==0: print(f"  {n+1}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    [p.join(timeout=20) for p in procs]
    # save
    if a.mode == "faithful":
        out = {k: torch.tensor(v[1]) for k,v in feats.items()}; out["feat_dim"]=NBINS+4
    else:
        out = {"feat_dim": NBINS+4}; from collections import defaultdict
        byds = defaultdict(dict)
        for k,(ds,f) in feats.items(): byds[ds][int(k.split(":")[1])] = f
        out["datasets"] = sorted(byds)
        for ds in byds:
            n = len(byds[ds]); out[f"{ds}/states"] = torch.tensor(np.stack([byds[ds][i] for i in range(n)]))
            out[f"{ds}/order"] = torch.arange(n)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); torch.save(out, a.out)
    print(f"saved {a.out}  feat_dim={NBINS+4}", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True); main()
