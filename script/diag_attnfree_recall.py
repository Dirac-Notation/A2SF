"""Diagnostic: do attention-free scorers (KeyDiff / L2-norm / TriAttention) actually
select the attention-important keys? Measure recall@budget of each method's top-budget
keys vs the ORACLE (accumulated last-window attention) and vs RANDOM. Sign-flips test
whether the implemented direction is inverted.

Run on 1 GPU:
  CUDA_VISIBLE_DEVICES=0 python script/diag_attnfree_recall.py
"""
import json, os, sys
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MP = json.load(open(f"{REPO}/config/model2path.json"))["llama3-1b"]
BUDGET, RECENT, WIN = 128, 16, 32
LAYERS = [4, 8, 12]
N_PROMPTS = 6

tok = AutoTokenizer.from_pretrained(MP)
model = AutoModelForCausalLM.from_pretrained(MP, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager", device_map="cuda")
model.eval()

# a few non-LongBench prompts (recipe), truncated to a manageable length
rows = [json.loads(l) for l in open(f"{REPO}/datasets/needle/diag_prompts.jsonl")]
prompts = []
for r in rows:
    p = r.get("input_prompt") or r.get("prompt") or ""
    if 6000 < len(p) < 16000:
        prompts.append(p)
    if len(prompts) >= N_PROMPTS:
        break

def topset(score, seq, exclude_recent):
    # top (BUDGET-RECENT) among the head part [0:seq-RECENT]
    head = seq - exclude_recent
    k = min(BUDGET - exclude_recent, head)
    return set(torch.topk(score[:head], k).indices.tolist())

sys.path.insert(0, REPO)
from utils_real_drop.scorers.triattention import TriAttentionScorer
TRI_STATS = torch.load(f"{REPO}/runs/triattention_stats/llama3-1b_stats.pt", map_location="cuda", weights_only=False)

agg = {m: [] for m in ["keydiff(-cos)", "keydiff(+cos)", "l2(-norm)", "l2(+norm)", "triatt", "random"]}
rng = np.random.RandomState(0)

for pi, p in enumerate(prompts):
    ids = tok(p, return_tensors="pt", truncation=True, max_length=8000).input_ids.cuda()
    seq = ids.shape[1]
    with torch.inference_mode():
        out = model(ids, output_attentions=True, use_cache=True)
    kv = out.past_key_values
    for L in LAYERS:
        # post-RoPE K for layer L: [B, num_kv, Sk, hd]
        K = kv.layers[L].keys[0].float()                      # [num_kv, Sk, hd]
        attn = out.attentions[L][0].float()                 # [num_q_heads, Sq, Sk]
        nq, _, _ = attn.shape
        nkv = K.shape[0]; grp = nq // nkv
        # TriAttention score for this layer (uses calibrated stats + repeated K)
        K_rep = K.repeat_interleave(grp, dim=0)[None]        # [1, nq, Sk, hd]
        tri = TriAttentionScorer(nkv, layer_idx=L, stats=TRI_STATS)
        tri.prepare_prefill(seq, K.device, torch.float32, key=K_rep, num_kv=nkv)
        tri_score = tri.score_keys(None, K_rep, nkv)[0]      # [nkv, Sk]
        for h in range(nkv):
            # ORACLE: accumulated attention from last WIN queries, this kv-group's query-heads
            qh = attn[h*grp:(h+1)*grp]                       # [grp, Sq, Sk]
            oracle = qh[:, -WIN:, :].sum(dim=(0, 1))         # [Sk]
            oset = topset(oracle, seq, RECENT)
            k = K[h]                                         # [Sk, hd]
            anchor = k.mean(0, keepdim=True)
            cos = F.cosine_similarity(k, anchor, dim=-1)     # [Sk]
            nrm = k.norm(dim=-1)                             # [Sk]
            cand = {
                "keydiff(-cos)": -cos, "keydiff(+cos)": cos,
                "l2(-norm)": -nrm, "l2(+norm)": nrm,
            }
            for name, sc in cand.items():
                sset = topset(sc, seq, RECENT)
                agg[name].append(len(oset & sset) / max(1, len(oset)))
            agg["triatt"].append(len(oset & topset(tri_score[h], seq, RECENT)) / max(1, len(oset)))
            # random baseline
            rsel = set(rng.choice(seq - RECENT, size=min(BUDGET - RECENT, seq - RECENT), replace=False).tolist())
            agg["random"].append(len(oset & rsel) / max(1, len(oset)))
    print(f"  prompt {pi+1}/{len(prompts)} (seq={seq}) done", flush=True)

print("\n=== recall@128 vs ORACLE (accumulated attention), mean over heads/layers/prompts ===")
for name in ["keydiff(-cos)", "keydiff(+cos)", "l2(-norm)", "l2(+norm)", "triatt", "random"]:
    print(f"  {name:14}: {np.mean(agg[name]):.3f}")
print("\n해석: oracle보다 충분히 높으면 신호 있음. random과 비슷하면 신호 없음(방법 한계).")
print("      -버전 < random 인데 +버전 > random 이면 부호 반대(구현 버그).")
