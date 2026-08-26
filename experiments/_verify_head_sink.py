"""Test: do sink-dominated (trivial) heads inflate the soft Tanimoto?
Per (layer, kv-head): sink concentration vs per-head Tanimoto. If high-sink heads
have high Tanimoto, the (L,kv)-averaged soft Tanimoto is inflated by trivial heads."""
import os, sys, math, json, warnings
import numpy as np, torch
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

def tan_perhead(acc, onorm):  # (L,kv) Ruzicka per head
    m=acc.clamp(min=0); mn=m/(m.sum(-1,keepdim=True)+1e-12)
    return (torch.minimum(onorm,mn).sum(-1)/torch.maximum(onorm,mn).sum(-1).clamp(1e-12))

def main():
    from scipy.stats import spearmanr
    model,tok=load_model("llama3-1b");dev=model.device
    rnd=__import__("random");rnd.seed(3)
    samples=[]
    for ds in ["hotpotqa","2wikimqa","multifieldqa_en","qasper"]:
        rows=[json.loads(l) for l in open(f"{REPO}/datasets/longbench/{ds}.jsonl")]
        pool=[r for r in rows if 2000<=r.get("length",0)<=6000]
        for r in rnd.sample(pool,2): r["dataset"]=ds; samples.append(r)
    col=C.AttentionCollector(model,C.MAX_WINDOW); G=None
    SINK=4
    allsink=[]; alltan=[]; allent=[]
    for r in samples:
        ds=r["dataset"]; p=f"[INST]{r['input_prompt']}[/INST]"
        ids=tok(p,truncation=True,max_length=32768,return_tensors="pt").input_ids.to(dev); S=ids.shape[1]
        model.init_cache(None);col.reset(S)
        with torch.no_grad(): pkv=model(ids,use_cache=True).past_key_values
        pf=C.prefill_to_pf_kv(cw(model,col,pkv,dev),C.CHUNK); G=pf.shape[0]
        model.init_cache(None)
        with torch.no_grad():
            gen=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=32,
                do_sample=False,num_beams=1,pad_token_id=tok.eos_token_id,tokenizer=tok,stop_strings="[/INST]",num_logits_to_keep=1)
        pred=tok.decode(gen[0,S:],skip_special_tokens=True)
        model.init_cache(None)
        with torch.no_grad(): pkv2=model(ids,use_cache=True).past_key_values
        ans=tfo(model,tok,pred,pkv2,S,dev)
        if ans is None: continue
        onorm=C.oracle_to_norm(ans,S)  # (L,kv,S)
        wflat=(pf.sum(0))              # flat-256 weighted score (L,kv,S)
        th=tan_perhead(wflat,onorm)    # (L,kv) per-head Tanimoto
        sink=onorm[:,:,:SINK].sum(-1)  # (L,kv) oracle sink fraction
        ent=-(onorm.clamp(min=1e-12)*onorm.clamp(min=1e-12).log()).sum(-1)  # (L,kv) entropy
        allsink+=sink.flatten().tolist(); alltan+=th.flatten().tolist(); allent+=ent.flatten().tolist()
        print(f"[{ds}] S={S} | head sink_frac: med={sink.median():.2f} >0.5={100*(sink>0.5).float().mean():.0f}% | per-head Tani med={th.median():.2f}")
    col.remove_hooks()
    s=np.array(allsink);t=np.array(alltan);e=np.array(allent)
    print(f"\n=== 전체 헤드 (n={len(s)} = L×kv×prompts) ===")
    print(f"  sink_frac>0.5 헤드: {100*np.mean(s>0.5):.0f}%  (oracle attention 절반 이상이 첫 {SINK}토큰)")
    print(f"  corr(sink_frac, per-head Tanimoto) = {spearmanr(s,t).correlation:+.3f}  (양수면 sink헤드가 Tanimoto 부풀림)")
    print(f"  corr(entropy,   per-head Tanimoto) = {spearmanr(e,t).correlation:+.3f}")
    print(f"  high-sink헤드(>0.5) 평균Tani={t[s>0.5].mean():.3f} vs low-sink(<0.2) 평균Tani={t[s<0.2].mean():.3f}")
if __name__=="__main__": main()
