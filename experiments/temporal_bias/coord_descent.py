"""Coordinate-descent optimal weights from uniform init.

No hyperparameters: all w_d initialised to 1.0, sequentially refined to
maximise Jaccard. Single pass, modest grid, no tolerance / tie-break.

Saves `coord_descent.npz` next to existing discovery.npz.
"""
import os
import sys
import json
import time
import random
from collections import defaultdict

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from experiments.temporal_bias.optimal import (
    AttentionCollector, gqa_topk, jaccard,
    LOCAL_RATIO, TOKEN_BUDGET, MAX_WINDOW, MAX_SEQ_LEN,
    NUM_ITEMS, LENGTH_MIN, LENGTH_MAX, MODEL_NAME,
    TASK_GROUP, SEED,
)

# Allow per-run overrides without editing optimal.py
TOKEN_BUDGET = int(os.environ.get("CD_BUDGET", str(TOKEN_BUDGET)))
MAX_WINDOW   = int(os.environ.get("CD_WINDOW", str(MAX_WINDOW)))

WORKPATH  = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(WORKPATH))


# Forward-greedy search grid (ascending; ties resolved toward smaller c)
GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CHUNK = 4  # group every CHUNK consecutive distances under a shared weight
MIN_REL_GAIN = 0.01  # block kept only if Jaccard gain ≥ 1% of running J


def _jaccard_gpu(a_idx, b_idx, seq_len, per_lh=False):
    L, H, _ = a_idx.shape
    dev = a_idx.device
    a = torch.zeros(L, H, seq_len, dtype=torch.bool, device=dev)
    b = torch.zeros(L, H, seq_len, dtype=torch.bool, device=dev)
    a.scatter_(2, a_idx, True)
    b.scatter_(2, b_idx, True)
    inter = (a & b).sum(2).float()
    union = (a | b).sum(2).float().clamp(min=1)
    j = inter / union
    if per_lh:
        return j.cpu().numpy()
    return float(j.mean().item())


def coord_descent(prefill, answer_idx, budget, kv_h, gs, seq_len,
                   chunk=CHUNK):
    """Forward-greedy block-wise coefficient search.

    Distances are grouped into consecutive chunks of `chunk` tokens. Starting
    from chunk 0 (closest to the last query) outward to chunk G-1, each chunk
    gets its coefficient picked once by sweeping GRID in ascending order;
    ties on Jaccard resolve toward the smaller c. Coefficients are initialised
    to 0 and built up sequentially. Returns weights of shape (G,) ordered by
    chunk distance from the prompt end (chunk 0 = closest).
    """
    L, H, W, S = prefill.shape
    assert W % chunk == 0, f"W={W} not divisible by chunk={chunk}"
    G = W // chunk
    local_b = max(1, int(budget * LOCAL_RATIO))
    sel_b = budget - local_b

    # (W, L, H, S) indexed by *position*; flip to be indexed by *distance*
    # (distance 0 = most recent), then group into G chunks of `chunk` tokens.
    prefill_w = prefill.permute(2, 0, 1, 3).contiguous()        # (W, L, H, S)
    prefill_d = prefill_w.flip(0)                               # (W, L, H, S) by distance
    prefill_chunked = (
        prefill_d.view(G, chunk, L, H, S).sum(dim=1).contiguous()  # (G, L, H, S)
    )

    def eval_accum(acc):
        """Evaluate Jaccard for a given accumulated score tensor (cloned internally)."""
        t = acc.clone()
        t[:, :, -local_b:] = t.max()
        idx = gqa_topk(t, sel_b, kv_h, gs)
        return _jaccard_gpu(answer_idx, idx, seq_len)

    w = np.zeros(G, dtype=np.float32)
    accumulated = torch.zeros_like(prefill_chunked[0])          # (L, H, S)

    temp = accumulated.clone()
    saved_local = torch.empty_like(temp[:, :, -local_b:])

    def eval_temp():
        saved_local.copy_(temp[:, :, -local_b:])
        temp[:, :, -local_b:] = temp.max()
        idx = gqa_topk(temp, sel_b, kv_h, gs)
        j = _jaccard_gpu(answer_idx, idx, seq_len)
        temp[:, :, -local_b:].copy_(saved_local)
        return j

    base_j = -1.0
    j_weight = np.zeros(G, dtype=np.float32)                    # running J after each chunk
    for g in range(G):                                          # g=0 closest → G-1 farthest
        block = prefill_chunked[g]                              # (L, H, S)

        best_c, best_j = GRID[0], -1.0
        j_zero = -1.0
        prev_c = 0.0
        temp.copy_(accumulated)
        for i, c in enumerate(GRID):
            delta = c - prev_c
            if i == 0:
                if abs(delta) > 1e-9:
                    temp.add_(block, alpha=delta)
            else:
                temp.add_(block, alpha=delta)
            j = eval_temp()
            if i == 0:
                j_zero = j                                      # J with w[g]=0
            # ascending GRID + strict ">" ⇒ ties resolve toward smaller c
            if j > best_j + 1e-9:
                best_j = j
                best_c = c
            prev_c = c

        # Drop the block when its best gain is < MIN_REL_GAIN of the
        # running Jaccard (J achieved by blocks 0..g-1 / equivalently
        # J_zero at this step). Treats marginal blocks as "no optimum".
        ref = max(j_zero, 1e-9)
        if best_c > 0.0 and (best_j - j_zero) < MIN_REL_GAIN * ref:
            best_c = 0.0
            best_j = j_zero

        if best_c > 1e-9:
            accumulated.add_(block, alpha=best_c)
        w[g] = best_c
        base_j = best_j
        j_weight[g] = base_j                                    # non-decreasing by construction

    # ── fixed-weight running Jaccards ─────────────────────────────────────────
    # Compute cumulative Jaccard at each chunk for three fixed weighting schemes:
    #   uniform   : w_g = 1 for all g  (H2O-like: all query positions equal)
    #   snap16    : w_g = 1 for g < 16/chunk, else 0  (SnapKV-16 style)
    #   sigmoid   : w_g = sigmoid(d_g, a, b) fitted to the optimal w per-sample
    snap16_G = max(1, 16 // chunk)
    d_centers = np.arange(G, dtype=np.float32) * chunk + (chunk - 1) / 2.0

    try:
        from scipy.optimize import curve_fit as _cf
        def _sig(x, a, b):
            return 1.0 / (1.0 + np.exp(np.clip(a * (x - b), -30.0, 30.0)))
        popt, _ = _cf(_sig, d_centers, w,
                      p0=[0.05, float(W) / 4.0],
                      bounds=([0.0, 0.0], [5.0, float(W)]),
                      maxfev=5000)
        sig_w = _sig(d_centers, *popt).astype(np.float32)
    except Exception:
        sig_w = np.ones(G, dtype=np.float32) * 0.5

    fixed_schemes = [
        ("uniform",  np.ones(G, dtype=np.float32)),
        ("snap16",   np.where(np.arange(G) < snap16_G, 1.0, 0.0).astype(np.float32)),
        ("sigmoid",  sig_w),
    ]
    j_fixed = {}
    for name, ws in fixed_schemes:
        accum2 = torch.zeros_like(prefill_chunked[0])
        j_run = np.zeros(G, dtype=np.float32)
        for g in range(G):
            if ws[g] > 1e-9:
                accum2.add_(prefill_chunked[g], alpha=float(ws[g]))
            j_run[g] = eval_accum(accum2)
        j_fixed[name] = j_run

    # Return prefill_chunked in float16 for storage (caller saves it for future analysis)
    pf_fp16 = prefill_chunked.cpu().half()   # (G, L, H, S)

    return w, base_j, j_weight, j_fixed, pf_fp16


_TF_BATCH = 16   # tokens per teacher-forcing step; 16× speedup over 1-at-a-time decode


def _teacher_forcing_answer_score(model, tokenizer, pred_text, past_kv, seq_len, device):
    """Batched teacher-forcing oracle attention scores.

    Processes pred tokens in chunks of _TF_BATCH using the prefill KV cache.
    Attention matrix per chunk: (n_heads, _TF_BATCH, S) — tiny, no OOM.
    ~16× speedup over token-by-token decode; same result format as
    collector.answer_score: (num_layers, num_kv_heads, seq_len).

    Returns None on empty pred.
    """
    pred_ids = tokenizer(pred_text, add_special_tokens=False,
                         return_tensors="pt").input_ids.to(device)
    N_pred = pred_ids.size(1)
    if N_pred == 0:
        return None

    cfg = model.config
    n_heads    = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    n_layers   = cfg.num_hidden_layers
    gs = n_heads // n_kv_heads

    # Keep query-head dimension (n_heads) so gqa_topk can do GQA aggregation itself,
    # matching the format of collector.answer_score: (n_layers, n_heads, seq_len).
    answer_score = torch.zeros(n_layers, n_heads, seq_len, dtype=torch.float32)

    import warnings
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for start in range(0, N_pred, _TF_BATCH):
            chunk = pred_ids[:, start:start + _TF_BATCH]      # (1, K)
            out = model(chunk, past_key_values=past_kv,
                        use_cache=True, output_attentions=True)
            past_kv = out.past_key_values
            if out.attentions is not None:
                for li, attn_l in enumerate(out.attentions):
                    if attn_l is None:
                        continue
                    # attn_l: (1, n_heads, K, S_past+K) — causal, sum prompt slice
                    a = attn_l[0, :, :, :seq_len].float()     # (n_heads, K, seq_len)
                    answer_score[li] += a.sum(dim=1).cpu()    # (n_heads, seq_len)
            del out

    return answer_score   # CPU; caller moves to target device


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    n_items_override = int(os.environ.get("CD_N_ITEMS", str(NUM_ITEMS)))

    with open(os.path.join(ROOT_PATH, "config", "model2path.json")) as f:
        model2path = json.load(f)
    with open(os.path.join(ROOT_PATH, "config", "dataset2maxlen.json")) as f:
        dataset2maxlen = json.load(f)
    model_path = model2path[MODEL_NAME]

    longbench_dir = os.path.join(ROOT_PATH, "datasets", "longbench")

    # ── Load backup predictions (pred text) ─────────────────────────────────────
    # backup/{MODEL}/llama3-1b_full/{ds}.jsonl is aligned row-by-row with dataset files.
    # Build: ds -> {row_idx -> pred_text}
    backup_dir = os.path.join(ROOT_PATH, "result_txt", "backup", "llama3-1b", "llama3-1b_full")
    backup_pred = {}   # ds -> list[str|None], indexed by dataset row
    for fname in os.listdir(longbench_dir):
        ds_name = fname.replace(".jsonl", "")
        bak_path = os.path.join(backup_dir, f"{ds_name}.jsonl")
        if not os.path.isfile(bak_path):
            continue
        with open(bak_path) as f:
            bak_rows = [json.loads(l) for l in f]
        backup_pred[ds_name] = [r.get("pred", "") for r in bak_rows]

    # ── Build sample pool: (row_idx, prompt) pairs for each dataset ─────────────
    # Track row index so we can look up the matching backup pred.
    dataset_items = defaultdict(list)   # ds -> list of (row_idx, input_prompt)
    for fname in os.listdir(longbench_dir):
        with open(os.path.join(longbench_dir, fname)) as f:
            rows = [json.loads(l) for l in f]
        for i, item in enumerate(rows):
            if LENGTH_MIN <= item.get("length", 0) <= LENGTH_MAX:
                dataset_items[item["dataset"]].append((i, item["input_prompt"]))

    dataset_filter = os.environ.get("CD_DATASET", "").strip()
    keep = set(s for s in dataset_filter.split(",") if s) if dataset_filter else None

    dataset2task, selected = {}, {}
    for task_name, datasets in TASK_GROUP.items():
        for d in datasets:
            dataset2task[d] = task_name
            if keep is not None and d not in keep:
                continue
            if d in dataset_items:
                pool = dataset_items[d]
                selected[d] = random.sample(pool, min(n_items_override, len(pool)))

    n_gpus = torch.cuda.device_count()
    print(f"Loading {model_path} across {n_gpus} GPUs...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    cfg = AutoConfig.from_pretrained(model_path)
    n_layers = cfg.num_hidden_layers
    device_map = {"model.embed_tokens": 0, "model.rotary_emb": 0,
                  "model.norm": 0, "lm_head": 0}
    for i in range(n_layers):
        device_map[f"model.layers.{i}"] = (i * n_gpus) // n_layers
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map=device_map, attn_implementation="sdpa",
    ).eval()

    collector = AttentionCollector(model, MAX_WINDOW)
    base_plots = os.path.join(WORKPATH, "plots")
    plot_dir = base_plots if TOKEN_BUDGET == 128 else os.path.join(base_plots, f"b{TOKEN_BUDGET}")
    print(f"  KV-cache budget = {TOKEN_BUDGET}; outputs → {plot_dir}")

    for dataset_name, items in selected.items():
        task_name = dataset2task[dataset_name]
        ds_preds = backup_pred.get(dataset_name, [])
        tf_available = len(ds_preds) > 0
        print(f"\n{'=' * 70}\n  {task_name}: {dataset_name}  ({len(items)} samples)"
              f"  [teacher-forcing={'ON' if tf_available else 'OFF (fallback to decode)'}]"
              f"\n{'=' * 70}")

        agg = defaultdict(list)
        for sample_i, (row_idx, prompt) in enumerate(items):
            print(f"  [{sample_i + 1}/{len(items)}] ", end="", flush=True)
            enc = tokenizer(f"[INST]{prompt}[/INST]", return_tensors="pt")
            input_ids = enc.input_ids.to(model.device)
            if input_ids.size(1) > MAX_SEQ_LEN:
                half = MAX_SEQ_LEN // 2
                input_ids = torch.cat([input_ids[:, :half], input_ids[:, -half:]], dim=1)
            seq_len = input_ids.size(1)
            max_new = dataset2maxlen.get(dataset_name, 512)

            # Lookup backup pred for this row (may be None/empty → fallback)
            pred_text = ds_preds[row_idx] if tf_available and row_idx < len(ds_preds) else None

            collector.reset(seq_len)
            t0 = time.time()
            with torch.no_grad():
                pf_out = model(input_ids, use_cache=True, num_logits_to_keep=1)
                past_kv  = pf_out.past_key_values
                next_tok = pf_out.logits[:, -1:].argmax(-1)   # first decode token (fallback)
                del pf_out; torch.cuda.empty_cache()
                data = collector.compute_window_data(past_kv)

            if pred_text:
                # ── Teacher forcing: batched forward (16 tokens/step) ─────────
                t_tf = time.time()
                answer_score = _teacher_forcing_answer_score(
                    model, tokenizer, pred_text, past_kv, seq_len, model.device)
                del past_kv; torch.cuda.empty_cache()
                mode_str = f"TF={time.time()-t_tf:.1f}s"
            else:
                # ── Fallback: auto-regressive decode ─────────────────────────
                # past_kv and next_tok are still valid from the prefill above.
                collector.set_decode()
                with torch.no_grad():
                    for _ in range(max_new):
                        out = model(next_tok, past_key_values=past_kv,
                                    use_cache=True, output_attentions=True)
                        past_kv  = out.past_key_values
                        next_tok = out.logits[:, -1:].argmax(-1)
                        if next_tok.item() == tokenizer.eos_token_id:
                            del out; break
                        del out
                del past_kv; torch.cuda.empty_cache()
                answer_score = collector.answer_score
                mode_str = "decode(fallback)"

            prefill_attn = data["prefill_attn"]
            kv_h = collector.num_kv_heads; gs = collector.group_size
            # Ensure answer_score is on the same device as prefill_attn
            # (model.device is unreliable with device_map)
            answer_score = answer_score.to(prefill_attn.device)
            answer_idx = gqa_topk(answer_score, TOKEN_BUDGET, kv_h, gs)

            t_cd = time.time()
            w_cd, j_cd, j_weight, j_fixed, pf_fp16 = coord_descent(
                prefill_attn, answer_idx, TOKEN_BUDGET, kv_h, gs, seq_len)
            print(f"L={seq_len}  {mode_str}  cd={time.time()-t_cd:.1f}s  J={j_cd:.3f}",
                  flush=True)

            agg["w_cd"].append(w_cd)
            agg["j_cd"].append(j_cd)
            agg["j_weight"].append(j_weight)
            agg["j_uniform"].append(j_fixed["uniform"])
            agg["j_snap16"].append(j_fixed["snap16"])
            agg["j_sigmoid"].append(j_fixed["sigmoid"])
            # Oracle tensors (small: ~60MB total per dataset) — skip decode on future runs
            agg["answer_score"].append(answer_score.cpu().half())  # (L, H, S) fp16
            agg["answer_idx"].append(answer_idx.cpu())             # (L, H, B) int64
            agg["seq_len_raw"].append(seq_len)
            # prefill_chunked (large: ~4GB/dataset uncompressed) — saved separately
            agg["prefill_chunked"].append(pf_fp16)                 # (G, L, H, S) fp16

            del data, prefill_attn, answer_score, pf_fp16; torch.cuda.empty_cache()

        ds_dir = os.path.join(plot_dir, task_name.replace(" ", "_"), dataset_name)
        os.makedirs(ds_dir, exist_ok=True)
        seq_lens = np.array(agg["seq_len_raw"], dtype=np.int32)

        # ── coord_descent.npz (small, <100 MB) ──────────────────────────────────
        # Contains all Jaccards + oracle indices. Load this for plotting.
        save_path = os.path.join(ds_dir, "coord_descent.npz")
        np.savez_compressed(
            save_path,
            # Coord-descent results
            w_cd=np.stack(agg["w_cd"], axis=0),              # (N, G)
            j_cd=np.array(agg["j_cd"], dtype=np.float32),   # (N,)
            j_weight=np.stack(agg["j_weight"], axis=0),      # (N, G)
            # Fixed-weight Jaccards — plot-ready
            j_uniform=np.stack(agg["j_uniform"], axis=0),    # (N, G) H2O-like
            j_snap16=np.stack(agg["j_snap16"], axis=0),      # (N, G) SnapKV-16 style
            j_sigmoid=np.stack(agg["j_sigmoid"], axis=0),    # (N, G) sigmoid-fit weights
            # Oracle token indices (for future Jaccard recomputation)
            answer_idx=np.stack([a.numpy() for a in agg["answer_idx"]]),  # (N, L, H, B)
            seq_lens=seq_lens,                                # (N,) true seq_len per sample
            chunk=np.int32(CHUNK),
            window=np.int32(MAX_WINDOW),
        )
        print(f"  saved → {save_path}")

        # ── oracle_score.npz (medium, ~60 MB) ───────────────────────────────────
        # answer_score: re-derive oracle at a different budget without re-running decode.
        # Padded to max_s along the last (S) dimension; use seq_lens to trim.
        max_s = int(max(t.shape[-1] for t in agg["answer_score"]))
        as_padded = np.zeros((len(agg["answer_score"]),
                              *agg["answer_score"][0].shape[:-1], max_s), dtype=np.float16)
        for i, t in enumerate(agg["answer_score"]):
            as_padded[i, ..., :t.shape[-1]] = t.numpy()
        np.savez_compressed(
            os.path.join(ds_dir, "oracle_score.npz"),
            answer_score=as_padded,   # (N, L, H, max_S) fp16
            seq_lens=seq_lens,
            budget=np.int32(TOKEN_BUDGET),
        )
        print(f"  saved → {ds_dir}/oracle_score.npz")

        # ── prefill_chunked.pt (large, ~1–4 GB compressed) ──────────────────────
        # List of (G, L, H, S_i) fp16 tensors, one per sample. S varies → list, not array.
        # Load only when testing a new weighting scheme; not needed for plotting.
        # Budget-independent: generated once per (dataset, window) combination.
        pf_save = os.path.join(ds_dir, "prefill_chunked.pt")
        torch.save(agg["prefill_chunked"], pf_save)
        pf_mb = sum(t.numel() * 2 for t in agg["prefill_chunked"]) / 1e6
        print(f"  saved → {pf_save}  ({pf_mb:.0f} MB uncompressed)")

    collector.remove_hooks()
    print("\n>>> Coord descent done.")


if __name__ == "__main__":
    main()
