"""The deployed label-free router: train on the recipe corpus, test on LongBench untouched.

Input (24 dims), all of it available from the single probe forward pass of iclr/probe_v3.py,
so routing costs one short prefill regardless of prompt length:

  generation-length bucket one-hot (3) | probe softmax (5) | probe hidden-state PCA (16)

The PCA basis is fitted on the training corpus only, so no LongBench information enters the
model. Target is the raw per-action reward; the loss is listwise score-weighted cross entropy,
which keeps the size of the gaps between actions instead of flattening them to ranks. Three
seeds are ensembled.

--feats selects the feature subset for ablations, --force_k fixes the action-menu size (13
means the full grid, which is what is deployed).

  python iclr/mlp_v3.py --model llama3-8b --pool poolz --variant v1t64e --feats prob \
      --reward p0 --force_k 13 --hid_dim 16 --tgt raw --loss ce --hidden 128 --depth 3
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn as nn
from scipy.stats import rankdata
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0,'.')
from iclr.proxy_gate import tanimoto, load_store
from iclr.reward import action_rewards, has_gold
from iclr.recipe_table import KEYS13, bucket
from iclr.clean_cell_table import true_cls
D2M=json.load(open("config/dataset2maxlen.json")); BUCK=["S","M","L"]
PROBE_ROOT=os.environ.get("ICLR_TRACES", "/data2/smp9898/iclr_traces")

def onehot(b):
    v=[0.]*3; v[BUCK.index(b)]=1.; return v

SURF_DIMS=None   # None means all of them; set to pick a subset of surface dims for ablations.
LEN_ENC="bucket"  # generation-length encoding: bucket / scalar / both / fine

def len_feat(g):
    """Encode the generation length.

    The value sets differ between training and evaluation: the recipe uses
    {16, 32, 64, 128, 256} while LongBench uses {32, 64, 128, 512}, and 512
    (gov_report, qmsum, multi_news) never appears in training. Bucketing (L means >128) is what
    connects the recipe's 256 to LongBench's 512, so a plain per-value one-hot is unusable: it
    would be all zeros on those three datasets. "scalar" extrapolates instead, and "both" keeps
    the bucket bridge while adding resolution inside a bucket.
    """
    import math
    b = "S" if g <= 32 else ("M" if g <= 128 else "L")
    oh = [0.]*3; oh[BUCK.index(b)] = 1.
    sc = [math.log(max(g, 1)) / 6.5]                      # log(512)/6.5 ~ 0.96, so the scalar stays in [0, 1]
    if LEN_ENC == "bucket": return oh
    if LEN_ENC == "scalar": return sc
    if LEN_ENC == "both":   return oh + sc
    if LEN_ENC == "fine":                                  # <=32 / 64 / 128 / >128
        f = [0.]*4
        f[0 if g <= 32 else (1 if g <= 64 else (2 if g <= 128 else 3))] = 1.
        return f
    raise ValueError(LEN_ENC)


def build_feat(b, prob, surf, mode):
    f=onehot(b) if not isinstance(b,(int,float)) else len_feat(b)
    if "prob" in mode: f=f+list(prob)
    if "surf" in mode:
        f=f+([float(surf[i]) for i in SURF_DIMS] if SURF_DIMS else list(surf))
    return f

def fit(X,Y,seeds=(0,1,2),epochs=400,loss="mse",hidden=64,depth=2,w=None):
    nets=[]
    for sd in seeds:
        torch.manual_seed(sd)
        layers,d=[],X.shape[1]
        for _ in range(depth): layers+=[nn.Linear(d,hidden),nn.ReLU()]; d=hidden
        layers+=[nn.Linear(d,Y.shape[1])]
        net=nn.Sequential(*layers)
        opt=torch.optim.Adam(net.parameters(),lr=1e-3)
        n=len(X); idx=np.random.RandomState(sd).permutation(n); nv=max(64,n//10)
        va,tr=idx[:nv],idx[nv:]
        g=torch.Generator().manual_seed(sd)          # seed the batch shuffle too, so runs do not depend on the global RNG
        best,bs,pat=1e9,None,0
        for _ in range(epochs):
            net.train(); p=tr[torch.randperm(len(tr),generator=g).numpy()]
            for i in range(0,len(p),256):
                b=p[i:i+256]; o=net(X[b])
                per=((o-Y[b])**2).mean(-1) if loss=="mse" else -(Y[b]*torch.log_softmax(o,-1)).sum(-1)
                l=(per*w[b]).sum()/w[b].sum() if w is not None else per.mean()
                opt.zero_grad(); l.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                ov=net(X[va])
                vl=float(((ov-Y[va])**2).mean() if loss=="mse" else -(Y[va]*torch.log_softmax(ov,-1)).sum(-1).mean())
            if vl<best-1e-6: best,bs,pat=vl,{k:v.clone() for k,v in net.state_dict().items()},0
            else:
                pat+=1
                if pat>=30: break
        net.load_state_dict(bs); net.eval(); nets.append(net)
    return nets

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True)
    ap.add_argument("--pool",default="poolx")
    ap.add_argument("--feats",default="prob+surf")
    ap.add_argument("--variant",default="v3")
    ap.add_argument("--force_k",type=int,default=0,help="0 picks k on a recipe held-out split; >0 fixes it")
    ap.add_argument("--surf_dims",default="",help="comma-separated surface dim indices; empty means all")
    ap.add_argument("--dump_picks",default="",help="path to dump the per-sample chosen action, for paired noise tests")
    ap.add_argument("--reward",default="p0",choices=["p0","gold","mix"],
        help="p0 = proxy only, gold = rows with references only, mix = gold when available else proxy")
    ap.add_argument("--min_full",type=float,default=0.0,help="minimum full-cache score for a gold row; filters unreliable references")
    ap.add_argument("--gold_weight",type=int,default=1,help="how many times to repeat a gold-bearing row under the mix reward")
    ap.add_argument("--min_disc",type=float,default=0.0,
        help="drop rows whose reward std across the 13 actions is below this")
    ap.add_argument("--disc_weight",action="store_true",
        help="weight each row by its across-action reward std instead of repeating it")
    ap.add_argument("--hid_var",default="v1t64h",help="variant directory holding the hidden-state npz")
    ap.add_argument("--hid_dim",type=int,default=0,help=">0 adds PCA-K of the probe hidden state; costs no extra forward pass")
    ap.add_argument("--tgt",default="rank",choices=["rank","raw"],help="training target")
    ap.add_argument("--loss",default="mse",choices=["mse","ce"],help="mse / listwise score-weighted")
    ap.add_argument("--hidden",type=int,default=64)
    ap.add_argument("--depth",type=int,default=2)
    ap.add_argument("--len_enc",default="bucket",choices=["bucket","scalar","both","fine"],
        help="generation-length encoding; per-value one-hot is unusable because the training and eval value sets differ")
    a=ap.parse_args()
    global SURF_DIMS, LEN_ENC
    LEN_ENC=a.len_enc
    SURF_DIMS=[int(x) for x in a.surf_dims.split(",")] if a.surf_dims else None
    PROBE=f"{PROBE_ROOT}/probe_{a.variant}"
    PROBE_H=f"{PROBE_ROOT}/probe_{a.hid_var}"
    HR=HL=None
    if a.hid_dim>0:
        zhr=np.load(f"{PROBE_H}/{a.model}_recipe.npz"); zhl=np.load(f"{PROBE_H}/{a.model}_lb.npz")
        HR={str(zhr["key"][i]):zhr["hid"][i].astype(np.float32) for i in range(len(zhr["key"]))}
        HL={str(zhl["key"][i]):zhl["hid"][i].astype(np.float32) for i in range(len(zhl["key"]))}
    zr=np.load(f"{PROBE}/{a.model}_recipe.npz")
    P={str(zr["key"][i]):(zr["prob"][i],zr["surf"][i]) for i in range(len(zr["key"]))}
    from iclr.reward import full_reward
    rows=[]; wts=[]; skipped=0
    for obj in (json.loads(l) for l in open(f"datasets/training/raw/recipe_{a.pool}_{a.model}.jsonl")):
        sid=str(obj["sample_id"]); fc=(obj.get("full_cache_pred") or "")
        if not fc.strip() or len(obj.get("action_outputs",[]))!=13 or sid not in P: continue
        if HR is not None and sid not in HR: continue
        if a.reward in ("gold","mix"):
            g=has_gold(obj)
            if a.reward=="gold" and not g: skipped+=1; continue
            if g and a.min_full>0 and full_reward(obj,"gold")<a.min_full: skipped+=1; continue
        pr,sf=P[sid]
        g_ok = a.reward in ("gold","mix") and has_gold(obj)
        rec=(int(obj["generation_length"]),
             true_cls(obj["dataset"],obj["task_type"],obj.get("metric")), pr, sf,
             action_rewards(obj,"gold" if g_ok else "p0"),
             (HR[sid] if HR is not None else None))
        # gold rows are more reliable than proxy rows, so mix can upweight them by repetition
        disc=float(rec[4].std())
        if a.min_disc>0 and disc<a.min_disc: skipped+=1; continue
        rows.extend([rec]*(a.gold_weight if (g_ok and a.reward=="mix") else 1))
        wts.extend([disc]*(a.gold_weight if (g_ok and a.reward=="mix") else 1))
    import collections as _c
    _src=_c.Counter("rv6" if str(r[0]) and False else "" for r in rows)
    print(f"  {len(rows)} training rows ({skipped} dropped)")
    # menu order: greedy oracle-P0 over the discrete recipe cells
    cl={}
    # menu cells always use buckets; --len_enc only changes the input features
    for b,c,_,_,p,_h in rows: cl.setdefault((bucket(b),c),[]).append(p)
    Cm={k:np.stack(v).mean(0) for k,v in cl.items()}; W={k:len(v) for k,v in cl.items()}
    val=lambda m: sum(W[k]*max(Cm[k][x] for x in m) for k in Cm)/sum(W.values())
    menu=[]
    for _ in range(13): menu.append(max((x for x in range(13) if x not in menu),key=lambda x: val(menu+[x])))
    P0M=np.stack([r[4] for r in rows])
    R = np.stack([rankdata(p)/13. for p in P0M]) if a.tgt=="rank" else (P0M/100.0 if P0M.max()>1.5 else P0M.copy())
    # the PCA basis is fitted on the recipe alone, so LongBench is never seen
    PCA=None
    if a.hid_dim>0:
        Hm=np.stack([r[5] for r in rows]); mu_h=Hm.mean(0)
        _,_,Vt=np.linalg.svd(Hm-mu_h,full_matrices=False); W=Vt[:a.hid_dim].T
        Z=(Hm-mu_h)@W; sd_h=Z.std(0)+1e-6
        PCA=(mu_h,W,sd_h)
    Xl_=[build_feat(b,pr,sf,a.feats) for b,_,pr,sf,_,_ in rows]
    if PCA is not None:
        Zr=((np.stack([r[5] for r in rows])-PCA[0])@PCA[1])/PCA[2]
        Xl_=[x+list(z) for x,z in zip(Xl_,Zr)]
    X=torch.tensor(np.array(Xl_,dtype=np.float32),dtype=torch.float32)
    # choose k on a 3-fold recipe held-out split
    rng=np.random.RandomState(0); idx=rng.permutation(len(rows)); best=None
    ks_try=[a.force_k] if a.force_k else range(2,14)
    for k in ks_try:
        mu=sorted(menu[:k]); Y=torch.tensor(R[:,mu],dtype=torch.float32); sc=[]
        for f in range(3):
            te=idx[f::3]; tr=np.setdiff1d(idx,te)
            n=fit(X[tr],Y[tr],seeds=(0,),loss=a.loss,hidden=a.hidden,depth=a.depth)
            with torch.no_grad(): pr_=n[0](X[te]).argmax(1).numpy()
            sc.append(float(np.mean([R[te[i],mu[pr_[i]]] for i in range(len(te))])))
        v=float(np.mean(sc))
        if best is None or v>best[0]: best=(v,k)
    k=best[1]; mu=sorted(menu[:k])
    Wt=torch.tensor(np.array(wts,dtype=np.float32)) if a.disc_weight else None
    if Wt is not None: Wt=Wt/Wt.mean()
    nets=fit(X, torch.tensor(R[:,mu],dtype=torch.float32),loss=a.loss,hidden=a.hidden,depth=a.depth,w=Wt)
    # LongBench, used purely as a test set
    zl=np.load(f"{PROBE}/{a.model}_lb.npz")
    PL={str(zl["key"][i]):(zl["prob"][i],zl["surf"][i]) for i in range(len(zl["key"]))}
    store=load_store(a.model); gt={}
    for (ds,i),r in store.items():
        if all(x in r["actions"] for x in KEYS13): gt[(ds,i)]=np.array([r["actions"][x]["score"] for x in KEYS13])
    ks=[kk for kk in sorted(gt) if f"{kk[0]}|{kk[1]}" in PL]
    if HL is not None: ks=[kk for kk in ks if f"{kk[0]}|{kk[1]}" in HL]
    XlL=[build_feat(int(D2M[kk[0]]),*PL[f"{kk[0]}|{kk[1]}"],a.feats) for kk in ks]
    if PCA is not None:
        Zl=((np.stack([HL[f"{kk[0]}|{kk[1]}"] for kk in ks])-PCA[0])@PCA[1])/PCA[2]
        XlL=[x+list(z) for x,z in zip(XlL,Zl)]
    Xl=torch.tensor(np.array(XlL,dtype=np.float32),dtype=torch.float32)
    with torch.no_grad(): sel=(sum(n(Xl) for n in nets)/len(nets)).argmax(1).numpy()
    g=mu[int(np.argmax(R[:,mu].mean(0)))]
    per,pg={},{}
    for j,kk in enumerate(ks):
        per.setdefault(kk[0],[]).append(gt[kk][mu[int(sel[j])]]); pg.setdefault(kk[0],[]).append(gt[kk][g])
    mac=lambda d: float(np.mean([np.mean(v) for v in d.values()]))
    t2d=json.load(open("config/task2dataset.json")); f2=  {d:t for t,ds in t2d.items() for d in ds}
    famr,famg={},{}
    for ds,v in per.items(): famr.setdefault(f2.get(ds,"?"),[]).append(float(np.mean(v)))
    for ds,v in pg.items():  famg.setdefault(f2.get(ds,"?"),[]).append(float(np.mean(v)))
    fam={t:{"ours":float(np.mean(famr[t])),"glob":float(np.mean(famg[t]))} for t in famr}
    print("  per task: " + "  ".join(f"{t[:12]}={fam[t]['ours']:.2f}" for t in sorted(fam)))
    out={"model":a.model,"feats":a.feats,"surf_dims":a.surf_dims,"reward":a.reward,"len_enc":a.len_enc,"hid_dim":a.hid_dim,"min_disc":a.min_disc,"disc_weight":a.disc_weight,"tgt":a.tgt,"loss":a.loss,"hidden":a.hidden,"depth":a.depth,"gold_weight":a.gold_weight,"k":k,"menu":[KEYS13[x] for x in mu],"fam":fam,
         "cv":float(best[0]),"lb":mac(per),"glob":mac(pg),"n":len(ks),"variant":a.variant}
    print(f"{a.model:12s} {a.variant:4s} [{a.feats+'/'+a.len_enc:20s}] k={k:2d} cv={out['cv']:.4f} LB={out['lb']:6.2f} (global {out['glob']:.2f}, Δ{out['lb']-out['glob']:+.2f})")
    if a.dump_picks:
        os.makedirs(os.path.dirname(a.dump_picks) or ".",exist_ok=True)
        json.dump({f"{kk[0]}|{kk[1]}":KEYS13[mu[int(sel[j])]] for j,kk in enumerate(ks)},
                  open(a.dump_picks,"w"))
    json.dump(out,open(f"result_txt/analysis/proxy_gate/mlpv3_{a.variant}_{a.model}_{a.feats.replace("+","_")}_{a.reward}{("_d"+a.surf_dims.replace(",","")) if a.surf_dims else ""}.json","w"),indent=1,ensure_ascii=False)
if __name__=="__main__": main()
