"""Test the per-layer (a,b) SCHEDULE hypothesis: does differentiating (a,b) per
layer beat the best UNIFORM (a,b)? Decodes an LB subset with several schedules,
scores GT. Multi-GPU. (No fast index — per-layer needs decoding.)

  python experiments/perlayer_eval.py --n_per 40 --gpus 0,1,2,3,4,5,6,7
"""
import argparse, json, os, sys, time
import multiprocessing as mp
import numpy as np, torch
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from utils import CompressionConfig, load_model
from datasets.generate_sigmoid_dataset import _format_prompt, _build_gen_kwargs, score_vs_ref

NL = 16  # llama3-1b layers
DS_METRIC = {"hotpotqa":"qa_f1_score","2wikimqa":"qa_f1_score","multifieldqa_en":"qa_f1_score",
             "qasper":"qa_f1_score","narrativeqa":"qa_f1_score","triviaqa":"qa_f1_score",
             "gov_report":"rouge_score","qmsum":"rouge_score","multi_news":"rouge_score",
             "samsum":"rouge_score","lcc":"code_sim_score","repobench-p":"code_sim_score"}
SUBSET = ["hotpotqa","2wikimqa","multifieldqa_en","qasper","gov_report","samsum","lcc","triviaqa"]

def ramp(lo, hi):  # length-NL linear ramp
    return [float(lo + (hi-lo)*i/(NL-1)) for i in range(NL)]

def schedules():
    # each: name -> (a_list[NL], b_list[NL])
    S = {}
    S["uni_a.1_b16"]  = ([0.1]*NL, [16.0]*NL)
    S["uni_a10_b16"]  = ([10.0]*NL, [16.0]*NL)     # SnapKV-ish steep recent
    S["uni_a0_b1"]    = ([0.0]*NL, [1.0]*NL)        # H2O
    S["ramp_a0to10"]  = (ramp(0.0,10.0), [16.0]*NL) # early flat -> late steep
    S["ramp_a10to0"]  = (ramp(10.0,0.0), [16.0]*NL) # early steep -> late flat
    S["ramp_b1to128"] = ([0.1]*NL, ramp(1.0,128.0)) # window grows with depth
    S["ramp_b128to1"] = ([0.1]*NL, ramp(128.0,1.0))
    return S

def _cfg(a_list, b_list, budget):
    c = CompressionConfig()
    c.compression_method = "waits"; c.total_budget = int(budget); c.local_ratios = 0.125
    c.a = torch.tensor([a_list[0]]); c.b = torch.tensor([b_list[0]])  # dummy fallback
    c.a_schedule = [torch.tensor([x]) for x in a_list]
    c.b_schedule = [torch.tensor([x]) for x in b_list]
    return c

def _worker(wid, gpu, model_name, tq, rq, budget, scheds):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.set_grad_enabled(False)
    model, tok = load_model(model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    try:
        while True:
            s = tq.get()
            if s is None: break
            ds = s["dataset"]; prompt = _format_prompt(s["input_prompt"], ds, model_name)
            enc = tok(prompt, truncation=False, return_tensors="pt")
            ids = enc.input_ids.to(model.device); am = enc.attention_mask.to(model.device)
            ctx = ids.shape[-1]; gk = _build_gen_kwargs(tok, ds, int(s.get("generation_length",64)), ctx)
            metric = DS_METRIC[ds]; ans = s.get("answers", []); allc = s.get("all_classes", [])
            out = {}
            for name,(a,b) in scheds.items():
                model.init_cache(_cfg(a,b,budget))
                with torch.inference_mode():
                    o = model.generate(input_ids=ids, attention_mask=am, **gk)
                pred = tok.decode(o[0, ctx:], skip_special_tokens=True)
                out[name] = score_vs_ref(pred, ans, metric, allc)
            rq.put((ds, out))
    except Exception as e:
        import traceback; rq.put(("__err__", f"w{wid}: {e}\n{traceback.format_exc()}"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per", type=int, default=40); ap.add_argument("--gpus", default="0")
    ap.add_argument("--model", default="llama3-1b"); ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--out", default="result_txt/analysis/perlayer/result.json")
    a = ap.parse_args()
    import random as rnd; rnd.seed(42)
    gpu_ids = [int(g) for g in a.gpus.split(",") if g!=""]
    scheds = schedules()
    samples = []
    for ds in SUBSET:
        rows = [json.loads(l) for l in open(os.path.join(REPO, f"datasets/longbench/{ds}.jsonl"))]
        pool = [r for r in rows if 2000 <= r.get("length",0) <= 8000]
        for r in rnd.sample(pool, min(a.n_per, len(pool))): r["dataset"]=ds; samples.append(r)
    print(f"perlayer eval: {len(samples)} prompts x {len(scheds)} schedules, gpus={gpu_ids}", flush=True)
    ctx = mp.get_context("spawn"); tq, rq = ctx.Queue(), ctx.Queue()
    for s in sorted(samples, key=lambda x:-x.get("length",0)): tq.put(s)
    for _ in gpu_ids: tq.put(None)
    procs = [ctx.Process(target=_worker, args=(i,g,a.model,tq,rq,a.budget,scheds)) for i,g in enumerate(gpu_ids)]
    [p.start() for p in procs]
    from collections import defaultdict
    acc = defaultdict(lambda: defaultdict(list)); n=0; t0=time.time()
    for _ in range(len(samples)):
        item = rq.get()
        if item[0]=="__err__": [p.terminate() for p in procs]; raise RuntimeError(item[1])
        ds, out = item; n+=1
        for name,v in out.items(): acc[name][ds].append(v)
        if n%20==0: print(f"  {n}/{len(samples)} ({time.time()-t0:.0f}s)", flush=True)
    [p.join(timeout=20) for p in procs]
    # overall = mean over datasets of mean per dataset
    res = {}
    for name in scheds:
        per_ds = {ds: float(np.mean(acc[name][ds])) for ds in SUBSET if acc[name][ds]}
        res[name] = {"overall": float(np.mean(list(per_ds.values()))*100), "per_ds": {k:round(v*100,2) for k,v in per_ds.items()}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out,"w"), indent=2)
    print("\n=== per-layer schedule eval (overall, GT x100) ===")
    for name in sorted(res, key=lambda k:-res[k]["overall"]):
        print(f"  {name:16s} {res[name]['overall']:.2f}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True); main()
