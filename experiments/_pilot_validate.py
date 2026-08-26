"""De-risk: does the obs1 prefill/oracle/coord-descent machinery run under v5?
One prompt, print shapes. If AttentionCollector breaks (rotary/cache API), fix here."""
import os, sys, json, math, warnings
import torch
REPO = "/home/smp9898/A2SF"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments/paper_figures/observations"))
import common as C
from utils import load_model
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

def compute_window_v5(model, collector, past_kv, device):
    """v5-compatible windowed prefill attention (L,H,W,S). Replaces common.py's
    compute_window_data (which used 4.46.2 attn_mod.rotary_emb + tuple cache)."""
    S = collector._prefill_len
    cfg = model.config
    n_h, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // n_h; g = n_h // n_kv
    out = []
    for i in range(cfg.num_hidden_layers):
        attn = model.model.layers[i].self_attn
        hidden = collector._window_inputs[i]          # (1,W,hidden)
        W = hidden.size(1)
        q = attn.q_proj(hidden).view(1, W, n_h, hd).transpose(1, 2)   # (1,H,W,hd)
        pos = torch.arange(S - W, S, device=device).unsqueeze(0)
        cos, sin = model.model.rotary_emb(q, pos)
        q_rot, _ = apply_rotary_pos_emb(q, q, cos, sin)
        k = past_kv.layers[i].keys                    # (1,n_kv,S,hd)
        q_g = q_rot.view(1, n_kv, g, W, hd)
        scores = torch.matmul(q_g, k.unsqueeze(2).transpose(-1, -2)) / math.sqrt(hd)
        scores = scores.view(1, n_h, W, S)
        kp = torch.arange(S, device=device); qp = torch.arange(S - W, S, device=device)
        scores.masked_fill_(~(kp.unsqueeze(0) <= qp.unsqueeze(1)).unsqueeze(0).unsqueeze(0), float("-inf"))
        out.append(torch.softmax(scores.float(), dim=-1)[0].cpu())    # (H,W,S)
    return torch.stack(out)                            # (L,H,W,S)

_NO_CHAT = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]
def fmt(p, ds, m="llama3-1b"):
    if str(ds).lower() not in _NO_CHAT and "llama" in m: return f"[INST]{p}[/INST]"
    return p

def tf_oracle_eager(model, tok, pred_text, past_kv, seq_len, device):
    pred_ids = tok(pred_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    if pred_ids.size(1) == 0: return None
    cfg = model.config
    ans = torch.zeros(cfg.num_hidden_layers, cfg.num_attention_heads, seq_len)
    def set_attn(impl):
        model.config._attn_implementation = impl
        for mod in model.modules():
            if hasattr(mod, "config"): mod.config._attn_implementation = impl
            if hasattr(mod, "_attn_implementation"): mod._attn_implementation = impl
    set_attn("eager")
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in range(0, pred_ids.size(1), 16):
            chunk = pred_ids[:, s:s+16]
            out = model(chunk, past_key_values=past_kv, use_cache=True, output_attentions=True)
            past_kv = out.past_key_values
            if out.attentions is not None:
                for li, a in enumerate(out.attentions):
                    if a is not None: ans[li] += a[0, :, :, :seq_len].float().sum(1).cpu()
            del out
    set_attn("sdpa")
    return ans

def main():
    model, tok = load_model("llama3-1b")
    device = model.device
    # one hotpotqa prompt, length 2000-6000
    rec = None
    for l in open(os.path.join(REPO, "datasets/longbench/hotpotqa.jsonl")):
        d = json.loads(l)
        if 2000 <= d.get("length", 0) <= 6000: rec = d; break
    prompt = fmt(rec["input_prompt"], "hotpotqa")
    ids = tok(prompt, truncation=True, max_length=32768, return_tensors="pt").input_ids.to(device)
    S = ids.shape[1]
    print(f"prompt len S={S}")

    model.init_cache(None)
    collector = C.AttentionCollector(model, C.MAX_WINDOW)
    collector.reset(S)
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past_kv = out.past_key_values
    print("prefill OK, past_kv type:", type(past_kv).__name__)
    try:
        prefill_attn = compute_window_v5(model, collector, past_kv, device)
        print("compute_window_v5 OK, prefill_attn:", tuple(prefill_attn.shape))
    except Exception as e:
        import traceback; print("compute FAILED:\n", traceback.format_exc()); collector.remove_hooks(); return
    collector.remove_hooks()

    pf_kv = C.prefill_to_pf_kv(prefill_attn, C.CHUNK)
    print("pf_kv:", tuple(pf_kv.shape))   # (G, L, kv, S)

    # oracle: greedy decode pred then teacher-force
    with torch.no_grad():
        gen = model.generate(input_ids=ids, max_new_tokens=int(rec.get("generation_length", 32)),
                             do_sample=False, num_beams=1, pad_token_id=tok.eos_token_id,
                             tokenizer=tok, stop_strings="[/INST]")
    pred = tok.decode(gen[0, S:], skip_special_tokens=True)
    print("pred:", repr(pred[:60]))
    model.init_cache(None)
    with torch.no_grad():
        out = model(ids, use_cache=True); past_kv = out.past_key_values
    ans = tf_oracle_eager(model, tok, pred, past_kv, S, device)
    print("oracle answer_score:", None if ans is None else tuple(ans.shape))
    oracle_norm = C.oracle_to_norm(ans, S)
    print("oracle_norm:", tuple(oracle_norm.shape))

    G = pf_kv.shape[0]; W = C.MAX_WINDOW
    w_tan, j_opt, j_sig = C.tanimoto_optimal(pf_kv, oracle_norm, G, W, C.CHUNK)
    print(f"w_tan nonzero={int((w_tan>1e-9).sum())}/{G}, j_tan_optimal final={j_opt[-1]:.3f}, j_sig final={j_sig[-1]:.3f}")
    print("VALIDATION OK")

if __name__ == "__main__":
    main()
