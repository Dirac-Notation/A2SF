"""Shared infrastructure for the Observation reproduction pipeline.

This module holds everything obs1.py and obs2.py need in common:
  - configuration constants (model, seed, lengths, window/chunk, tasks)
  - model + prompt loading (deterministic sampling that mirrors the paper run)
  - AttentionCollector (windowed prefill attention) and gqa_topk
  - teacher-forcing oracle attention
  - Tanimoto similarity primitives (single / uniform / coordinate-descent optimal)
  - sigmoid helper for fitting forgetting curves

The numerical logic here is ported verbatim from the original (now removed)
experiments/temporal_bias/{optimal.py,coord_descent.py} and
experiments/paper_figures/observations/compute_alt_metrics.py so that the
figures reproduce exactly.  The only structural change is that obs1 computes
the small Tanimoto curves inline and discards the large prefill/oracle tensors
instead of persisting ~125 GB of intermediates.
"""
import os
import math
import json
import random

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # .../A2SF
LONGBENCH_DIR = os.path.join(ROOT, "datasets", "longbench")
BACKUP_PRED_DIR = os.path.join(ROOT, "result_txt", "backup", "llama3-1b", "llama3-1b_full")
DATA_DIR = os.path.join(HERE, "data")

# ── Model / GQA ────────────────────────────────────────────────────────────────
MODEL_NAME  = "llama3-1b"
N_KV_HEADS  = 8     # llama3.2-1b
GROUP_SIZE  = 4     # n_heads / n_kv_heads

# ── Shared seed ────────────────────────────────────────────────────────────────
SEED = 42

# ── obs1 config (sigmoid_band + tanimoto_recovery) ─────────────────────────────
# These exact values reproduce the committed figures: verified by matching the
# per-prompt seq_lens against the archived /data2/hyunrae/plots data.
#   window=256 (chunk=4 -> G=64), 40 prompts/dataset (hotpotqa pool caps at 28),
#   global-seed sampling in TASK_GROUP order.
LENGTH_MIN   = 2000
LENGTH_MAX   = 6000
NUM_ITEMS    = 40
MAX_SEQ_LEN  = 32768
MAX_WINDOW   = 256
CHUNK        = 4
BUDGET       = 128
LOCAL_RATIO  = 0.125

# Sampling order MUST match the original coord_descent.py global-seed order:
# random.seed(SEED) once, then iterate TASK_GROUP in insertion order and
# random.sample each dataset's pool.  Keep this dict order frozen.
TASK_GROUP = {
    "Few Shot":      ["samsum"],
    "Single-doc QA": ["qasper"],
    "Multi-doc QA":  ["hotpotqa"],
    "Summarization": ["gov_report"],
}

# Display order for the obs1 figures (4 columns).  Independent of sampling order.
OBS1_TASKS = [
    ("Single-doc_QA/qasper",     "Single-doc QA"),
    ("Multi-doc_QA/hotpotqa",    "Multi-doc QA"),
    ("Summarization/gov_report", "Summarization"),
    ("Few_Shot/samsum",          "Few-Shot"),
]
# Map "Task/dataset" display path -> the flat data filename stem used in data/.
def data_stem(task_path: str) -> str:
    return task_path.replace("/", "__")


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════
def model_path() -> str:
    with open(os.path.join(ROOT, "config", "model2path.json")) as f:
        return json.load(f)[MODEL_NAME]


def load_model(device: str = "cuda", dtype=torch.bfloat16):
    """Load tokenizer + 1B model on a single device.

    obs1 uses bfloat16 (matching the original coord_descent.py prefill);
    obs2 passes torch.float16 (matching the original fig3_window_dominance.py).
    """
    mp = model_path()
    print(f"loading {mp} on {device} ({dtype}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(mp)
    model = AutoModelForCausalLM.from_pretrained(
        mp, torch_dtype=dtype,
        device_map={"": device}, attn_implementation="sdpa",   # efficient prefill
    ).eval()
    return tok, model


# ══════════════════════════════════════════════════════════════════════════════
# Prompt loading (deterministic, mirrors original coord_descent.py)
# ══════════════════════════════════════════════════════════════════════════════
def load_backup_preds():
    """ds -> list[str] of full-cache predictions, row-aligned with dataset files."""
    backup_pred = {}
    for fname in os.listdir(LONGBENCH_DIR):
        ds_name = fname.replace(".jsonl", "")
        bak_path = os.path.join(BACKUP_PRED_DIR, f"{ds_name}.jsonl")
        if not os.path.isfile(bak_path):
            continue
        with open(bak_path) as f:
            bak_rows = [json.loads(l) for l in f]
        backup_pred[ds_name] = [r.get("pred", "") for r in bak_rows]
    return backup_pred


def sample_prompts(n_items: int = NUM_ITEMS):
    """Return {dataset -> list[(row_idx, prompt)]} sampled exactly as the paper run.

    Builds the per-dataset pool from all LongBench files (length-filtered), then
    seeds once and samples in TASK_GROUP insertion order.  This reproduces the
    identical prompt set the committed figures were made from.
    """
    from collections import defaultdict
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    dataset_items = defaultdict(list)   # ds -> [(row_idx, input_prompt)]
    for fname in os.listdir(LONGBENCH_DIR):
        with open(os.path.join(LONGBENCH_DIR, fname)) as f:
            rows = [json.loads(l) for l in f]
        for i, item in enumerate(rows):
            if LENGTH_MIN <= item.get("length", 0) <= LENGTH_MAX:
                dataset_items[item["dataset"]].append((i, item["input_prompt"]))

    selected = {}
    for task_name, datasets in TASK_GROUP.items():
        for d in datasets:
            if d in dataset_items:
                pool = dataset_items[d]
                selected[d] = random.sample(pool, min(n_items, len(pool)))
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# AttentionCollector  (verbatim from optimal.py; prefill-window reconstruction)
# ══════════════════════════════════════════════════════════════════════════════
class AttentionCollector:
    """Reconstructs windowed prefill attention (last MAX_WINDOW queries × all keys).

    A forward pre-hook stores the last-W hidden states per layer during prefill;
    compute_window_data() then recomputes Q (post-rotary), reloads K from the KV
    cache, and returns per-layer softmax attention of shape (L, H, W, S).
    The decode-path machinery is retained for API parity but unused here (oracle
    attention comes from teacher forcing, not the collector).
    """

    def __init__(self, model, max_window):
        self.model = model
        self.max_window = max_window
        cfg = model.config
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", self.num_heads)
        self.group_size = self.num_heads // self.num_kv_heads
        self.head_dim = cfg.hidden_size // self.num_heads
        self.hidden_size = cfg.hidden_size

        self._window_inputs = {}
        self._answer = None
        self._prefill_len = 0
        self._is_prefill = True

        self._hooks = []
        for i, layer in enumerate(model.model.layers):
            h1 = layer.self_attn.register_forward_pre_hook(
                self._pre_hook(i), with_kwargs=True
            )
            self._hooks.append(h1)

    def reset(self, prefill_len):
        self._window_inputs.clear()
        self._answer = torch.zeros(self.num_layers, self.num_heads, prefill_len)
        self._prefill_len = prefill_len
        self._is_prefill = True

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _pre_hook(self, layer_idx):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states")
            if hidden is None:
                hidden = args[0] if args else None
            if hidden is None:
                return
            if self._is_prefill and hidden.size(1) > 1:
                w = min(self.max_window, hidden.size(1))
                self._window_inputs[layer_idx] = hidden[:, -w:, :].detach()
        return hook

    def compute_window_data(self, past_kv):
        """Return {'prefill_attn': (L, H, W, S)} reconstructed from stored hidden + KV."""
        S = self._prefill_len
        prefill_attn = []
        for i in range(self.num_layers):
            attn_mod = self.model.model.layers[i].self_attn
            hidden = self._window_inputs[i]
            device = hidden.device
            W = hidden.size(1)

            q = attn_mod.q_proj(hidden)
            q = q.view(1, W, self.num_heads, self.head_dim).transpose(1, 2)
            pos_ids = torch.arange(S - W, S, device=device).unsqueeze(0)
            # transformers v5: rotary_emb moved from per-attn-module to model level
            rotary = getattr(attn_mod, "rotary_emb", None) or self.model.model.rotary_emb
            cos, sin = rotary(q, pos_ids)
            q_rot, _ = apply_rotary_pos_emb(q, q, cos, sin)

            # v5 cache: layer keys via .layers[i].keys (fallback to legacy tuple access)
            try:
                k = past_kv.layers[i].keys             # (1, nkv, S, D)
            except (AttributeError, TypeError):
                k = past_kv[i][0]
            q_g = q_rot.view(1, self.num_kv_heads, self.group_size, W, self.head_dim)
            k_t = k.unsqueeze(2).transpose(-1, -2)
            scores = torch.matmul(q_g, k_t) / math.sqrt(self.head_dim)
            scores = scores.view(1, self.num_heads, W, S)

            kp = torch.arange(S, device=device)
            qp = torch.arange(S - W, S, device=device)
            mask = kp.unsqueeze(0) <= qp.unsqueeze(1)
            scores.masked_fill_(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            attn_w = torch.softmax(scores.float(), dim=-1).to(q_rot.dtype)
            prefill_attn.append(attn_w[0].cpu())       # (H, W, S)

            del q, q_rot, q_g, k_t, scores, attn_w, hidden, k
            self._window_inputs[i] = None
            torch.cuda.empty_cache()

        return {"prefill_attn": torch.stack(prefill_attn)}   # (L, H, W, S)


def gqa_topk(scores, k, num_kv_heads, group_size):
    if group_size <= 1:
        return scores.topk(k, dim=-1).indices
    *batch, _H, S = scores.shape
    g = scores.view(*batch, num_kv_heads, group_size, S).sum(dim=-2)
    idx = g.topk(k, dim=-1).indices
    return idx.repeat_interleave(group_size, dim=-2)


# ══════════════════════════════════════════════════════════════════════════════
# Teacher-forcing oracle attention  (verbatim from coord_descent.py)
# ══════════════════════════════════════════════════════════════════════════════
_TF_BATCH = 16   # tokens per teacher-forcing step


def teacher_forcing_answer_score(model, tokenizer, pred_text, past_kv, seq_len, device):
    """Batched teacher-forcing oracle attention scores: (n_layers, n_heads, seq_len)."""
    pred_ids = tokenizer(pred_text, add_special_tokens=False,
                         return_tensors="pt").input_ids.to(device)
    N_pred = pred_ids.size(1)
    if N_pred == 0:
        return None

    cfg = model.config
    n_heads  = cfg.num_attention_heads
    n_layers = cfg.num_hidden_layers
    answer_score = torch.zeros(n_layers, n_heads, seq_len, dtype=torch.float32)

    import warnings
    # teacher-forcing needs attention weights -> eager just for these (few-query) forwards;
    # prefill stays sdpa (efficient). restore afterwards.
    _prev_impl = model.config._attn_implementation
    model.config._attn_implementation = "eager"
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for start in range(0, N_pred, _TF_BATCH):
            chunk = pred_ids[:, start:start + _TF_BATCH]
            out = model(chunk, past_key_values=past_kv,
                        use_cache=True, output_attentions=True)
            past_kv = out.past_key_values
            if out.attentions is not None:
                for li, attn_l in enumerate(out.attentions):
                    if attn_l is None:
                        continue
                    a = attn_l[0, :, :, :seq_len].float()   # (n_heads, K, seq_len)
                    answer_score[li] += a.sum(dim=1).cpu()
            del out
    model.config._attn_implementation = _prev_impl
    return answer_score


# ══════════════════════════════════════════════════════════════════════════════
# Sigmoid helper  (verbatim)
# ══════════════════════════════════════════════════════════════════════════════
def sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(np.clip(a * (x - b), -30.0, 30.0)))


def fit_sigmoid_to_row(w_row, d_centers, W):
    """Fit descending sigmoid to a single weight vector. Returns sig_w (G,)."""
    from scipy.optimize import curve_fit
    try:
        popt, _ = curve_fit(sigmoid, d_centers, w_row,
                            p0=[0.05, float(W) / 4.0],
                            bounds=([0.0, 0.0], [5.0, float(W)]),
                            maxfev=5000)
        return sigmoid(d_centers, *popt).astype(np.float32)
    except Exception:
        return np.ones_like(w_row) * 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Tanimoto curves  (verbatim numerics from compute_alt_metrics.py)
# ══════════════════════════════════════════════════════════════════════════════
_GRID_NZERO = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                           dtype=torch.float32)


def prefill_to_pf_kv(prefill_attn, chunk=CHUNK):
    """(L, H, W, S) windowed attention -> (G, L, kv, S) chunked, kv-head space.

    Mirrors coord_descent.coord_descent (permute/flip/chunk) followed by the
    GQA reduction in compute_alt_metrics (view+sum over group dim).
    """
    L, H, W, S = prefill_attn.shape
    assert W % chunk == 0, f"W={W} not divisible by chunk={chunk}"
    G = W // chunk
    pf_w = prefill_attn.permute(2, 0, 1, 3).contiguous()        # (W, L, H, S)
    pf_d = pf_w.flip(0)                                         # by distance
    pf_chunked = pf_d.view(G, chunk, L, H, S).sum(dim=1)        # (G, L, H, S)
    pf_kv = (pf_chunked.float()
             .view(G, L, N_KV_HEADS, GROUP_SIZE, S).sum(dim=3))  # (G, L, kv, S)
    return pf_kv


def oracle_to_norm(answer_score, seq_len):
    """(L, H, S) oracle attention -> normalised (L, kv, S_i) Tanimoto reference."""
    L, H, S = answer_score.shape
    oracle_qh = answer_score[:, :, :seq_len].float()
    oracle_kv = oracle_qh.view(L, N_KV_HEADS, GROUP_SIZE, seq_len).sum(dim=2)
    o_raw = oracle_kv.clamp(min=0.0)
    return o_raw / (o_raw.sum(-1, keepdim=True) + 1e-12)        # (L, kv, S_i)


def _tan_scalar(acc, oracle_norm):
    m = acc.clamp(min=0.0)
    mn = m / (m.sum(-1, keepdim=True) + 1e-12)
    return float((torch.minimum(oracle_norm, mn).sum(-1) /
                  torch.maximum(oracle_norm, mn).sum(-1).clamp(1e-12)).mean())


def tanimoto_optimal(pf_kv, oracle_norm, G, W, chunk=CHUNK):
    """Coord-descent weights maximising Tanimoto (g=0 forced to 1.0).

    Returns (w_tan (G,), j_tan_optimal (G,), j_tan_sigmoid (G,)).
    """
    L = pf_kv.shape[1]
    kv = pf_kv.shape[2]
    S_i = pf_kv.shape[3]
    d_centers = np.arange(G, dtype=np.float32) * chunk + (chunk - 1) / 2.0

    w_tan    = np.zeros(G, dtype=np.float32)
    accum    = torch.zeros(L, kv, S_i, dtype=torch.float32)
    j_weight = np.zeros(G, dtype=np.float32)

    for g in range(G):
        block = pf_kv[g]
        if g == 0:
            best_w = 1.0
        else:
            base_j = _tan_scalar(accum, oracle_norm)
            best_w = 0.0
            temps  = (accum.unsqueeze(0) +
                      block.unsqueeze(0) * _GRID_NZERO.view(-1, 1, 1, 1))
            m      = temps.clamp(min=0.0)
            mn     = m / (m.sum(-1, keepdim=True) + 1e-12)
            o_exp  = oracle_norm.unsqueeze(0)
            scores = ((torch.minimum(o_exp, mn).sum(-1) /
                       torch.maximum(o_exp, mn).sum(-1).clamp(1e-12))
                      .mean(dim=(1, 2)))
            best_idx = int(scores.argmax())
            if float(scores[best_idx]) > base_j + 1e-9:
                best_w = float(_GRID_NZERO[best_idx])
        if best_w > 1e-9:
            accum = accum + block * best_w
        w_tan[g]    = best_w
        j_weight[g] = _tan_scalar(accum, oracle_norm)

    sig_w     = fit_sigmoid_to_row(w_tan, d_centers, W)
    accum_sig = torch.zeros(L, kv, S_i, dtype=torch.float32)
    j_sig     = np.zeros(G, dtype=np.float32)
    for g in range(G):
        if sig_w[g] > 1e-9:
            accum_sig = accum_sig + pf_kv[g] * float(sig_w[g])
        j_sig[g] = _tan_scalar(accum_sig, oracle_norm)
    return w_tan, j_weight, j_sig


def tanimoto_uniform(pf_kv, oracle_norm, G):
    """Running Tanimoto with uniform weights (all chunks weight 1.0). (G,)."""
    L, kv, S_i = pf_kv.shape[1], pf_kv.shape[2], pf_kv.shape[3]
    accum = torch.zeros(L, kv, S_i, dtype=torch.float32)
    j = np.zeros(G, dtype=np.float32)
    for g in range(G):
        accum = accum + pf_kv[g]
        j[g] = _tan_scalar(accum, oracle_norm)
    return j


def tanimoto_single(pf_kv, oracle_norm, G):
    """Per-chunk (non-accumulated) Tanimoto similarity. (G,)."""
    j = np.zeros(G, dtype=np.float32)
    for g in range(G):
        m_raw = pf_kv[g].clamp(min=0.0)
        m_norm = m_raw / (m_raw.sum(-1, keepdim=True) + 1e-12)
        inter = torch.minimum(oracle_norm, m_norm).sum(-1)
        union = torch.maximum(oracle_norm, m_norm).sum(-1).clamp(1e-12)
        j[g] = float((inter / union).mean())
    return j
