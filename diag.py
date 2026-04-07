"""Layer-by-layer divergence diagnostic between HF and KVLlama."""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_real_drop.kv_llama import KVLlamaForCausalLM

PROMPT = "The quick brown fox jumps over the lazy dog. " * 80  # ~700 tok

def load(path):
    return path

with open("config/model2path.json") as f:
    mp = json.load(f)["llama3-1b"]
tok = AutoTokenizer.from_pretrained(mp)

@torch.no_grad()
def run(impl_label, hf_impl, use_kv):
    if use_kv:
        m = KVLlamaForCausalLM.from_pretrained(mp, torch_dtype=torch.float16).cuda().eval()
        m.init_cache(compression_config=None)
    else:
        m = AutoModelForCausalLM.from_pretrained(
            mp, torch_dtype=torch.float16, attn_implementation=hf_impl
        ).cuda().eval()
    inp = tok(PROMPT, return_tensors="pt").to("cuda")
    out = m(input_ids=inp.input_ids, attention_mask=inp.attention_mask, use_cache=False, output_hidden_states=True)
    hs = [h.float().cpu() for h in out.hidden_states]
    logits = out.logits[0, -1].float().cpu()
    del m; torch.cuda.empty_cache()
    return hs, logits

print("Running HF eager...")
hs_eager, lg_eager = run("hf-eager", "eager", False)
print("Running HF sdpa...")
hs_sdpa,  lg_sdpa,  = run("hf-sdpa",  "sdpa",  False)
print("Running KVLlama (no compression)...")
hs_kv,    lg_kv     = run("kv",       None,    True)

def diffs(a, b):
    return [(x - y).abs().max().item() for x, y in zip(a, b)]

print("\nLayer-wise max|hidden_state diff|")
print("layer | KV vs HF-eager | KV vs HF-sdpa | HF-sdpa vs HF-eager")
for i, (a, b, c) in enumerate(zip(diffs(hs_kv, hs_eager), diffs(hs_kv, hs_sdpa), diffs(hs_sdpa, hs_eager))):
    print("{:5d} | {:14.4e} | {:13.4e} | {:18.4e}".format(i, a, b, c))

print("\nFinal logit max|D|")
def fmt(a, b):
    return "{:.4e}  argmax_eq={}".format((a-b).abs().max().item(), a.argmax().item()==b.argmax().item())
print("  KV vs HF-eager: " + fmt(lg_kv, lg_eager))
print("  KV vs HF-sdpa : " + fmt(lg_kv, lg_sdpa))
print("  HF-sdpa vs HF-eager: " + fmt(lg_sdpa, lg_eager))
