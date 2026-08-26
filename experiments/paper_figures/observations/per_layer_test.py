"""DIAGNOSTIC: if we knew each LAYER's attention, could we find that layer's optimal
forgetting coefficient -- and does per-layer beat a single shared coefficient?

For each prompt: prefill (windowed attention) + teacher-forced ORACLE attention (per layer).
Then Tanimoto-recover the oracle key-distribution with the forgetting weight, two ways:
  shared    = one coord-descent weight over ALL layers (current obs1 result)
  per_layer = an independent coord-descent weight PER LAYER (uses that layer's attention)
Reports mean Tanimoto (higher = better key recovery). per_layer >> shared => knowing each
layer's attention gives real per-layer headroom; per_layer ~= shared => layers want the same
coefficient (no per-layer benefit). single (TOVA) / uniform (H2O) shown for context.

  python experiments/paper_figures/observations/per_layer_test.py --nitems 6
"""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nitems", type=int, default=6, help="prompts per task")
    a = ap.parse_args()

    selected = C.sample_prompts(a.nitems)
    backup_pred = C.load_backup_preds()
    tok, model = C.load_model("cuda")
    collector = C.AttentionCollector(model, C.MAX_WINDOW)
    W = int(C.MAX_WINDOW)

    rows = []
    for task_path, label in C.OBS1_TASKS:
        dataset = task_path.split("/")[1]
        ds_preds = backup_pred.get(dataset, [])
        for si, (row_idx, prompt) in enumerate(selected[dataset]):
            MAXLEN = 3000   # cap so eager teacher-forcing attention fits comfortably
            enc = tok(f"[INST]{prompt}[/INST]", return_tensors="pt")
            ids = enc.input_ids.to(model.device)
            if ids.size(1) > MAXLEN:
                h = MAXLEN // 2; ids = torch.cat([ids[:, :h], ids[:, -h:]], dim=1)
            seq_len = ids.size(1)
            pred_text = ds_preds[row_idx] if row_idx < len(ds_preds) else ""
            if not pred_text:
                continue
            collector.reset(seq_len)
            with torch.no_grad():
                out = model(ids, use_cache=True, num_logits_to_keep=1)
                past_kv = out.past_key_values; del out; torch.cuda.empty_cache()
                data = collector.compute_window_data(past_kv)
            answer_score = C.teacher_forcing_answer_score(model, tok, pred_text, past_kv, seq_len, model.device)
            del past_kv; torch.cuda.empty_cache()

            pf_kv = C.prefill_to_pf_kv(data["prefill_attn"], C.CHUNK)   # (G, L, kv, S)
            oracle_norm = C.oracle_to_norm(answer_score, seq_len)        # (L, kv, S)
            G, L = pf_kv.shape[0], pf_kv.shape[1]

            # shared optimal (all layers, one weight)
            _, j_sh, _ = C.tanimoto_optimal(pf_kv, oracle_norm, G, W, C.CHUNK)
            j_single = float(np.asarray(C.tanimoto_single(pf_kv, oracle_norm, G)).reshape(-1)[-1])
            j_uni = float(np.asarray(C.tanimoto_uniform(pf_kv, oracle_norm, G)).reshape(-1)[-1])
            # per-layer optimal (each layer its own weight, using that layer's attention)
            per = []
            wopt_layers = []
            for li in range(L):
                w_l, j_l, _ = C.tanimoto_optimal(pf_kv[:, li:li+1], oracle_norm[li:li+1], G, W, C.CHUNK)
                per.append(float(j_l[-1])); wopt_layers.append(w_l)
            j_pl = float(np.mean(per))
            # how much do per-layer optimal weights disagree? std of the weight curves across layers
            w_spread = float(np.mean(np.std(np.stack(wopt_layers), axis=0)))
            rows.append((label, float(j_sh[-1]), j_pl, j_single, j_uni, w_spread))
            print(f"  {label[:10]:10} L={seq_len:5d}  shared={j_sh[-1]:.3f} per_layer={j_pl:.3f} "
                  f"single={j_single:.3f} uniform={j_uni:.3f}  w_spread={w_spread:.3f}", flush=True)
            del data, answer_score, pf_kv, oracle_norm; torch.cuda.empty_cache()

    collector.remove_hooks()
    R = np.array([r[1:] for r in rows], dtype=np.float32)
    print(f"\n===== per-layer optimal coefficient test (N={len(rows)} prompts) =====")
    print(f"  shared-optimal Tanimoto   = {R[:,0].mean():.3f}")
    print(f"  PER-LAYER optimal Tanimoto= {R[:,1].mean():.3f}   (gain {R[:,1].mean()-R[:,0].mean():+.3f})")
    print(f"  single (TOVA) Tanimoto    = {R[:,2].mean():.3f}")
    print(f"  uniform (H2O) Tanimoto    = {R[:,3].mean():.3f}")
    print(f"  per-layer weight spread   = {R[:,4].mean():.3f}  (0=all layers same coeff)")


if __name__ == "__main__":
    main()
