"""Verify: coord-descent's SOFT Tanimoto (its objective) is monotonic in k,
while the HARD top-112 set Tanimoto (what the pilot plotted) need not be."""
import os, sys, json, math, warnings
import numpy as np, torch
REPO = "/home/smp9898/A2SF"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments/paper_figures/observations"))
import common as C
from utils import load_model
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

def compute_window_v5(model, collector, past_kv, device):
    S = collector._prefill_len; cfg = model.config
    n_h, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // n_h; g = n_h // n_kv; out = []
    for i in range(cfg.num_hidden_layers):
        attn = model.model.layers[i].self_attn; hidden = collector._window_inputs[i]; W = hidden.size(1)
        q = attn.q_proj(hidden).view(1, W, n_h, hd).transpose(1, 2)
        pos = torch.arange(S - W, S, device=device).unsqueeze(0)
        cos, sin = model.model.rotary_emb(q, pos); q_rot, _ = apply_rotary_pos_emb(q, q, cos, sin)
        k = past_kv.layers[i].keys; q_g = q_rot.view(1, n_kv, g, W, hd)
        sc = torch.matmul(q_g, k.unsqueeze(2).transpose(-1, -2)) / math.sqrt(hd); sc = sc.view(1, n_h, W, S)
        kp = torch.arange(S, device=device); qp = torch.arange(S - W, S, device=device)
        sc.masked_fill_(~(kp.unsqueeze(0) <= qp.unsqueeze(1)).unsqueeze(0).unsqueeze(0), float("-inf"))
        out.append(torch.softmax(sc.float(), dim=-1)[0].cpu())
    return torch.stack(out)

def tf_oracle(model, tok, pred, past_kv, S, device):
    pid = tok(pred, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    if pid.size(1) == 0: return None
    cfg = model.config; ans = torch.zeros(cfg.num_hidden_layers, cfg.num_attention_heads, S)
    def sa(impl):
        model.config._attn_implementation = impl
        for m in model.modules():
            if hasattr(m, "config"): m.config._attn_implementation = impl
            if hasattr(m, "_attn_implementation"): m._attn_implementation = impl
    sa("eager")
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in range(0, pid.size(1), 16):
            o = model(pid[:, s:s+16], past_key_values=past_kv, use_cache=True, output_attentions=True)
            past_kv = o.past_key_values
            if o.attentions is not None:
                for li, a in enumerate(o.attentions):
                    if a is not None: ans[li] += a[0, :, :, :S].float().sum(1).cpu()
    sa("sdpa"); return ans

def hard_tani(score_LkvS, oracle_LkvK, sel, head_len):
    keep = score_LkvS[:, :, :head_len].topk(sel, dim=-1).indices
    L, kv, _ = keep.shape; tot = 0.0
    for li in range(L):
        for h in range(kv):
            a = set(keep[li, h].tolist()); b = set(oracle_LkvK[li, h].tolist())
            tot += len(a & b) / max(1, len(a | b))
    return tot / (L * kv)

def main():
    model, tok = load_model("llama3-1b"); device = model.device
    local = int(C.LOCAL_RATIO * C.BUDGET); sel = C.BUDGET - local
    samples = []
    import random as rnd; rnd.seed(1)
    for ds in ["hotpotqa", "qasper", "gov_report"]:
        rows = [json.loads(l) for l in open(os.path.join(REPO, f"datasets/longbench/{ds}.jsonl"))]
        pool = [r for r in rows if 2000 <= r.get("length", 0) <= 6000]
        for r in rnd.sample(pool, 2): r["dataset"] = ds; samples.append(r)
    collector = C.AttentionCollector(model, C.MAX_WINDOW)
    KS = [16, 32, 64, 128, 256]
    for r in samples:
        prompt = f"[INST]{r['input_prompt']}[/INST]" if r["dataset"] != "samsum" else r["input_prompt"]
        ids = tok(prompt, truncation=True, max_length=32768, return_tensors="pt").input_ids.to(device)
        S = ids.shape[1]; head_len = S - local
        model.init_cache(None); collector.reset(S)
        with torch.no_grad(): pkv = model(ids, use_cache=True).past_key_values
        pf_kv = C.prefill_to_pf_kv(compute_window_v5(model, collector, pkv, device), C.CHUNK)
        G = pf_kv.shape[0]
        model.init_cache(None)
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=32,
                                 do_sample=False, num_beams=1, pad_token_id=tok.eos_token_id,
                                 tokenizer=tok, stop_strings="[/INST]", num_logits_to_keep=1)
        pred = tok.decode(gen[0, S:], skip_special_tokens=True)
        model.init_cache(None)
        with torch.no_grad(): pkv2 = model(ids, use_cache=True).past_key_values
        ans = tf_oracle(model, tok, pred, pkv2, S, device)
        if ans is None: continue
        oracle_norm = C.oracle_to_norm(ans, S)
        oracle_keep = oracle_norm[:, :, :head_len].topk(sel, dim=-1).indices
        w_tan, j_opt, _ = C.tanimoto_optimal(pf_kv, oracle_norm, G, C.MAX_WINDOW, C.CHUNK)
        wt = torch.tensor(w_tan, dtype=torch.float32)
        print(f"\n[{r['dataset']}] S={S}")
        print("  k     soft(search obj)  hard(top-112)")
        for k in KS:
            nc = max(1, min(G, k // C.CHUNK))
            acc = (pf_kv * wt.view(G, 1, 1, 1) * (torch.arange(G) < nc).float().view(G, 1, 1, 1)).sum(0)
            soft = C._tan_scalar(acc, oracle_norm)
            hard = hard_tani(acc, oracle_keep, sel, head_len)
            print(f"  {k:4d}    {soft:.4f}           {hard:.4f}")
    collector.remove_hooks()

if __name__ == "__main__":
    main()
