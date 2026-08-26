"""Head meaningfulness = contribution to the OUTPUT (gradient attribution to the
answer), NOT just sink. Test: do low-importance heads inflate the Tanimoto?
Per (layer, kv-head): importance = |grad . activation| of o_proj input w.r.t.
the answer NLL.  Then corr(importance, per-head Tanimoto)."""
import os, sys, math, json, warnings
import numpy as np, torch, torch.nn.functional as F
REPO = "/home/smp9898/A2SF"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments/paper_figures/observations"))
import common as C
from utils import load_model
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

def cw(model, col, pkv, dev):
    S=col._prefill_len; cfg=model.config; nh,nkv=cfg.num_attention_heads,cfg.num_key_value_heads
    hd=cfg.hidden_size//nh; g=nh//nkv; out=[]
    for i in range(cfg.num_hidden_layers):
        at=model.model.layers[i].self_attn; h=col._window_inputs[i]; W=h.size(1)
        q=at.q_proj(h).view(1,W,nh,hd).transpose(1,2)
        cos,sin=model.model.rotary_emb(q,torch.arange(S-W,S,device=dev).unsqueeze(0))
        qr,_=apply_rotary_pos_emb(q,q,cos,sin); k=pkv.layers[i].keys
        sc=torch.matmul(qr.view(1,nkv,g,W,hd),k.unsqueeze(2).transpose(-1,-2))/math.sqrt(hd)
        sc=sc.view(1,nh,W,S)
        kp=torch.arange(S,device=dev);qp=torch.arange(S-W,S,device=dev)
        sc.masked_fill_(~(kp.unsqueeze(0)<=qp.unsqueeze(1)).unsqueeze(0).unsqueeze(0),float("-inf"))
        out.append(torch.softmax(sc.float(),-1)[0].cpu())
    return torch.stack(out)

def tfo(model,tok,pred,pkv,S,dev):
    pid=tok(pred,add_special_tokens=False,return_tensors="pt").input_ids.to(dev)
    if pid.size(1)==0:return None
    cfg=model.config;ans=torch.zeros(cfg.num_hidden_layers,cfg.num_attention_heads,S)
    def sa(im):
        model.config._attn_implementation=im
        for m in model.modules():
            if hasattr(m,"config"):m.config._attn_implementation=im
            if hasattr(m,"_attn_implementation"):m._attn_implementation=im
    sa("eager")
    with torch.no_grad(),warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in range(0,pid.size(1),16):
            o=model(pid[:,s:s+16],past_key_values=pkv,use_cache=True,output_attentions=True);pkv=o.past_key_values
            if o.attentions is not None:
                for li,a in enumerate(o.attentions):
                    if a is not None:ans[li]+=a[0,:,:,:S].float().sum(1).cpu()
    sa("sdpa");return ans

def tan_perhead(acc,onorm):
    m=acc.clamp(min=0); mn=m/(m.sum(-1,keepdim=True)+1e-12)
    return (torch.minimum(onorm,mn).sum(-1)/torch.maximum(onorm,mn).sum(-1).clamp(1e-12))

def head_importance(model,tok,ids,ans_ids,dev):
    """|grad . act| of each o_proj input head, summed over positions/dims -> (L,nkv)."""
    cfg=model.config; nh,nkv=cfg.num_attention_heads,cfg.num_key_value_heads; hd=cfg.hidden_size//nh; g=nh//nkv
    cap={}
    def mk(li):
        def h(mod,inp):
            x=inp[0]
            if x.requires_grad: x.retain_grad(); cap[li]=x
        return h
    hks=[model.model.layers[i].self_attn.o_proj.register_forward_pre_hook(mk(i)) for i in range(cfg.num_hidden_layers)]
    full=torch.cat([ids,ans_ids],dim=1)
    Sp=ids.shape[1]; tgt=full[0,Sp:]; L=tgt.shape[0]   # answer tokens
    model.init_cache(None)
    out=model(full[:,:-1], use_cache=False, logits_to_keep=L)  # only last L logits
    lp=F.log_softmax(out.logits[0].float(),-1)         # (L, vocab)
    nll=-lp[torch.arange(L),tgt].sum()
    model.zero_grad(set_to_none=True); nll.backward()
    imp=[]
    for li in range(cfg.num_hidden_layers):
        x=cap[li]; gr=x.grad
        a=(x.detach()*gr.detach()).abs().float()[0]    # (S, hidden)
        a=a.view(a.shape[0],nh,hd).sum(dim=(0,2))       # (nh,)
        a=a.view(nkv,g).sum(1)                          # (nkv,)
        imp.append(a.cpu())
    for h in hks: h.remove()
    return torch.stack(imp)                             # (L, nkv)

def main():
    from scipy.stats import spearmanr
    model,tok=load_model("llama3-1b");dev=model.device
    rnd=__import__("random");rnd.seed(5); samples=[]
    for ds in ["hotpotqa","2wikimqa","multifieldqa_en","qasper"]:
        rows=[json.loads(l) for l in open(f"{REPO}/datasets/longbench/{ds}.jsonl")]
        pool=[r for r in rows if 2000<=r.get("length",0)<=5000]
        for r in rnd.sample(pool,2): r["dataset"]=ds; samples.append(r)
    col=C.AttentionCollector(model,C.MAX_WINDOW)
    AI=[];AT=[];AS=[]
    for r in samples:
      try:
        p=f"[INST]{r['input_prompt']}[/INST]"
        ids=tok(p,truncation=True,max_length=5200,return_tensors="pt").input_ids.to(dev);S=ids.shape[1]
        model.init_cache(None)
        with torch.no_grad():
            gen=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=32,
                do_sample=False,num_beams=1,pad_token_id=tok.eos_token_id,tokenizer=tok,stop_strings="[/INST]",num_logits_to_keep=1)
        ans_ids=gen[:,S:]
        if ans_ids.shape[1]==0: continue
        # head importance (grad attribution)
        imp=head_importance(model,tok,ids,ans_ids,dev)            # (L,nkv)
        # per-head Tanimoto (flat-256 vs oracle)
        col.reset(S); model.init_cache(None)
        with torch.no_grad(): pkv=model(ids,use_cache=True).past_key_values
        pf=C.prefill_to_pf_kv(cw(model,col,pkv,dev),C.CHUNK)
        model.init_cache(None)
        with torch.no_grad(): pkv2=model(ids,use_cache=True).past_key_values
        pred=tok.decode(ans_ids[0],skip_special_tokens=True)
        oa=tfo(model,tok,pred,pkv2,S,dev)
        if oa is None: continue
        onorm=C.oracle_to_norm(oa,S); th=tan_perhead(pf.sum(0),onorm)
        sink=onorm[:,:,:4].sum(-1)
        AI+=imp.flatten().tolist(); AT+=th.flatten().tolist(); AS+=sink.flatten().tolist()
        print(f"[{r['dataset']}] S={S} | imp(med)={imp.median():.3f} Tani(med)={th.median():.2f}")
        del pf, onorm, th, imp, oa, pkv, pkv2, gen, ans_ids
        model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
      except Exception as exc:
        print(f"[{r['dataset']}] skip ({type(exc).__name__})", flush=True)
        model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
    col.remove_hooks()
    I=np.array(AI);T=np.array(AT);Sk=np.array(AS)
    Ir=(I-I.min())/(I.max()-I.min()+1e-9)
    print(f"\n=== 헤드 출력기여도 vs Tanimoto (n={len(I)} 헤드) ===")
    print(f"  corr(importance, per-head Tanimoto) = {spearmanr(I,T).correlation:+.3f}  (음수면 의미없는 헤드가 Tanimoto 높임)")
    print(f"  corr(importance, sink_frac)         = {spearmanr(I,Sk).correlation:+.3f}")
    q=np.quantile(I,[0.25,0.75])
    print(f"  low-importance 헤드(하위25%) 평균Tani={T[I<=q[0]].mean():.3f} vs high-importance(상위25%) 평균Tani={T[I>=q[1]].mean():.3f}")
if __name__=="__main__": main()
