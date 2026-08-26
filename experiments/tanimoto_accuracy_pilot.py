"""PILOT: does maximising Tanimoto(keep-set, oracle) raise accuracy (GT)?

Per prompt: build keep-sets from a sweep of weightings on the OPTIMAL-weighting
manifold (flat -> w_tan prefixes -> full w_tan -> sigmoid-fit), plus the per-head
oracle (Tanimoto=1). For each, measure hard Tanimoto to the oracle keep-set AND
decode (budget 128) + score GT. Output per-prompt (tanimoto, gt) for every
weighting so we can check the WITHIN-prompt relationship (avoids Simpson's).

  python experiments/tanimoto_accuracy_pilot.py --n_per 12 --gpu 0
"""
import os, sys, json, math, warnings, argparse
import numpy as np, torch
REPO = "/home/smp9898/A2SF"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments/paper_figures/observations"))
import common as C
from utils import load_model, CompressionConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from longbench_eval import qa_f1_score, rouge_score, classification_score, code_sim_score

_METRIC = {"qa_f1_score": qa_f1_score, "rouge_score": rouge_score,
           "classification_score": classification_score, "code_sim_score": code_sim_score}
DS_METRIC = {"hotpotqa": "qa_f1_score", "2wikimqa": "qa_f1_score",
             "multifieldqa_en": "qa_f1_score", "qasper": "qa_f1_score",
             "narrativeqa": "qa_f1_score", "triviaqa": "qa_f1_score",
             "gov_report": "rouge_score", "samsum": "rouge_score"}
PILOT_DS = ["hotpotqa", "2wikimqa", "multifieldqa_en", "qasper", "narrativeqa", "triviaqa", "samsum"]
_NO_CHAT = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]


def fmt(p, ds, m="llama3-1b"):
    if str(ds).lower() not in _NO_CHAT and "llama" in m:
        return f"[INST]{p}[/INST]"
    return p


def score_gt(pred, answers, metric, all_classes):
    fn = _METRIC.get(metric, qa_f1_score)
    best = 0.0
    for r in (answers or []):
        if not r:
            continue
        try:
            v = float(fn(str(pred), str(r), all_classes=all_classes or []))
        except Exception:
            v = 0.0
        best = max(best, v)
    return best


def score_fc(pred, fc_pred, metric, all_classes):
    """Fidelity to the full-cache output (single reference = fc_pred)."""
    if not fc_pred:
        return 0.0
    fn = _METRIC.get(metric, qa_f1_score)
    try:
        return float(fn(str(pred), str(fc_pred), all_classes=all_classes or []))
    except Exception:
        return 0.0


def compute_window_v5(model, collector, past_kv, device):
    S = collector._prefill_len
    cfg = model.config
    n_h, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // n_h; g = n_h // n_kv
    out = []
    for i in range(cfg.num_hidden_layers):
        attn = model.model.layers[i].self_attn
        hidden = collector._window_inputs[i]
        W = hidden.size(1)
        q = attn.q_proj(hidden).view(1, W, n_h, hd).transpose(1, 2)
        pos = torch.arange(S - W, S, device=device).unsqueeze(0)
        cos, sin = model.model.rotary_emb(q, pos)
        q_rot, _ = apply_rotary_pos_emb(q, q, cos, sin)
        k = past_kv.layers[i].keys
        q_g = q_rot.view(1, n_kv, g, W, hd)
        scores = torch.matmul(q_g, k.unsqueeze(2).transpose(-1, -2)) / math.sqrt(hd)
        scores = scores.view(1, n_h, W, S)
        kp = torch.arange(S, device=device); qp = torch.arange(S - W, S, device=device)
        scores.masked_fill_(~(kp.unsqueeze(0) <= qp.unsqueeze(1)).unsqueeze(0).unsqueeze(0), float("-inf"))
        out.append(torch.softmax(scores.float(), dim=-1)[0].cpu())
    return torch.stack(out)


def tf_oracle_eager(model, tok, pred_text, past_kv, S, device):
    pred_ids = tok(pred_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    if pred_ids.size(1) == 0:
        return None
    cfg = model.config
    ans = torch.zeros(cfg.num_hidden_layers, cfg.num_attention_heads, S)
    def set_attn(impl):
        model.config._attn_implementation = impl
        for mod in model.modules():
            if hasattr(mod, "config"): mod.config._attn_implementation = impl
            if hasattr(mod, "_attn_implementation"): mod._attn_implementation = impl
    set_attn("eager")
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in range(0, pred_ids.size(1), 16):
            out = model(pred_ids[:, s:s+16], past_key_values=past_kv, use_cache=True, output_attentions=True)
            past_kv = out.past_key_values
            if out.attentions is not None:
                for li, a in enumerate(out.attentions):
                    if a is not None: ans[li] += a[0, :, :, :S].float().sum(1).cpu()
            del out
    set_attn("sdpa")
    return ans


def topk_set(score_LkvS, k, head_len):
    """score (L,kv,S) -> per (L,kv) top-k indices in [0,head_len). returns (L,kv,k) long."""
    return score_LkvS[:, :, :head_len].topk(k, dim=-1).indices


def tanimoto(keep_LkvK, oracle_LkvK, k):
    L, kv, _ = keep_LkvK.shape
    tot = 0.0
    for li in range(L):
        for h in range(kv):
            a = set(keep_LkvK[li, h].tolist()); b = set(oracle_LkvK[li, h].tolist())
            tot += len(a & b) / max(1, len(a | b))
    return tot / (L * kv)


def decode_keepset(model, tok, ids, keep_LkvK, budget, local, gen_len, device):
    cfg = CompressionConfig()
    cfg["compression_method"] = "oracle"; cfg["total_budget"] = int(budget)
    cfg["recent_budget"] = int(local)
    cfg["oracle_indices"] = [keep_LkvK[li].unsqueeze(0).to(torch.int64) for li in range(keep_LkvK.shape[0])]
    model.init_cache(cfg)
    am = torch.ones_like(ids)
    with torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=int(gen_len),
                             do_sample=False, num_beams=1, pad_token_id=tok.eos_token_id,
                             tokenizer=tok, stop_strings="[/INST]", num_logits_to_keep=1)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per", type=int, default=12)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--shard_count", type=int, default=1)
    ap.add_argument("--answerable_min", type=float, default=0.0,
                    help="keep only prompts whose backup full-cache GT >= this "
                         "(model can actually answer; avoids GT=0 floor noise).")
    ap.add_argument("--out", default="result_txt/analysis/tanimoto_accuracy/pilot.jsonl")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    import random as rnd; rnd.seed(42)

    model, tok = load_model("llama3-1b")
    device = model.device
    G2L = json.load(open(os.path.join(REPO, "config/dataset2maxlen.json")))
    budget = C.BUDGET; local = int(C.LOCAL_RATIO * C.BUDGET); sel = budget - local
    W = C.MAX_WINDOW

    # sample prompts (length 2000-6000) per dataset
    samples = []
    for ds in PILOT_DS:
        rows = [json.loads(l) for l in open(os.path.join(REPO, f"datasets/longbench/{ds}.jsonl"))]
        pool = [r for r in rows if 2000 <= r.get("length", 0) <= 8000]
        for r in rnd.sample(pool, min(a.n_per, len(pool))):
            r["dataset"] = ds; samples.append(r)
    if a.shard_count > 1:
        samples = [s for i, s in enumerate(samples) if i % a.shard_count == a.shard_id]
    print(f"pilot shard {a.shard_id}/{a.shard_count}: {len(samples)} prompts, "
          f"budget={budget} local={local} sel={sel}", flush=True)

    collector = C.AttentionCollector(model, W)
    fout = open(a.out, "w")
    for n, rec in enumerate(samples):
        ds = rec["dataset"]
        prompt = fmt(rec["input_prompt"], ds)
        ids = tok(prompt, truncation=True, max_length=32768, return_tensors="pt").input_ids.to(device)
        S = ids.shape[1]; head_len = S - local
        gen_len = int(G2L.get(ds, 64)); metric = DS_METRIC[ds]
        ans = rec.get("answers", []); allc = rec.get("all_classes", [])

        try:
            # 1) full-cache decode FIRST -> fc_gt -> skip the expensive sweep if
            #    the model can't answer even with full cache (GT=0 floor = noise).
            model.init_cache(None)
            with torch.no_grad():
                gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                     max_new_tokens=gen_len, do_sample=False, num_beams=1,
                                     pad_token_id=tok.eos_token_id, tokenizer=tok,
                                     stop_strings="[/INST]", num_logits_to_keep=1)
            fc_pred = tok.decode(gen[0, S:], skip_special_tokens=True); del gen
            if not fc_pred.strip():        # no full-cache output -> FC fidelity undefined
                continue
            fc_gt = score_gt(fc_pred, ans, metric, allc)
            if a.answerable_min > 0 and fc_gt < a.answerable_min:
                continue

            # 2) prefill with collector -> windowed attention pf_kv
            model.init_cache(None); collector.reset(S)
            with torch.no_grad():
                past_kv = model(ids, use_cache=True).past_key_values
            prefill_attn = compute_window_v5(model, collector, past_kv, device)
            del past_kv; torch.cuda.empty_cache()
            pf_kv = C.prefill_to_pf_kv(prefill_attn, C.CHUNK); del prefill_attn   # (G,L,kv,S)
            G = pf_kv.shape[0]

            # 3) teacher-forcing oracle (decode-time attention each ctx token gets)
            model.init_cache(None)
            with torch.no_grad():
                past_kv2 = model(ids, use_cache=True).past_key_values
            answer_score = tf_oracle_eager(model, tok, fc_pred, past_kv2, S, device)
            del past_kv2; torch.cuda.empty_cache()
            if answer_score is None:
                continue
            oracle_norm = C.oracle_to_norm(answer_score, S); del answer_score     # (L,kv,S)
            w_tan, j_opt, j_sig_curve = C.tanimoto_optimal(pf_kv, oracle_norm, G, W, C.CHUNK)
            oracle_keep = topk_set(oracle_norm, sel, head_len)                     # (L,kv,sel)

            # query-count sweep: flat@k (uniform window) vs wtan@k (optimal weight
            # truncated to first k recent queries). chunks distance-ordered.
            KS = [16, 32, 64, 128, 256]
            weightings = {}
            for k in KS:
                nc = max(1, min(G, k // C.CHUNK))
                fw = np.zeros_like(w_tan); fw[:nc] = 1.0; weightings[f"flat_q{k}"] = fw
                tw = np.zeros_like(w_tan); tw[:nc] = w_tan[:nc]; weightings[f"wtan_q{k}"] = tw
            results = {"i": n, "dataset": ds, "S": S, "fc_gt": fc_gt,
                       "j_opt_final": float(j_opt[-1]), "n_nonzero": int((w_tan > 1e-9).sum()), "points": []}
            for wname, w in weightings.items():
                wt = torch.tensor(w, dtype=torch.float32).view(G, 1, 1, 1)
                wscore = (pf_kv * wt).sum(0)                        # (L,kv,S)
                keep = topk_set(wscore, sel, head_len)
                tani_hard = tanimoto(keep, oracle_keep, sel)
                tani_soft = C._tan_scalar(wscore, oracle_norm)
                pred = decode_keepset(model, tok, ids, keep, budget, local, gen_len, device)
                gt = score_gt(pred, ans, metric, allc)
                fcs = score_fc(pred, fc_pred, metric, allc)
                results["points"].append({"w": wname, "tani_soft": round(tani_soft, 4),
                                          "tani_hard": round(tani_hard, 4),
                                          "gt": round(gt, 4), "fc": round(fcs, 4)})
            pred = decode_keepset(model, tok, ids, oracle_keep, budget, local, gen_len, device)
            results["points"].append({"w": "oracle", "tani_soft": 1.0, "tani_hard": 1.0,
                                      "gt": round(score_gt(pred, ans, metric, allc), 4),
                                      "fc": round(score_fc(pred, fc_pred, metric, allc), 4)})
            fout.write(json.dumps(results) + "\n"); fout.flush()
            pts = " ".join(f"{p['w']}:{p['tani_soft']:.2f}/{p['gt']:.2f}" for p in results["points"])
            print(f"[{n+1}/{len(samples)}] {ds} S={S} fc={fc_gt:.2f} | {pts}", flush=True)
        except Exception as exc:
            print(f"[{n}] {ds} skip ({type(exc).__name__}: {str(exc)[:50]})", flush=True)
        finally:
            torch.cuda.empty_cache()
    collector.remove_hooks(); fout.close()
    print(f"done -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
