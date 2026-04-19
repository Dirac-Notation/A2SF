"""AJ: per-query forgetting weight = Jaccard overlap of top-k with the most-recent query.

Motivation (from experiments/temporal_bias analysis):
  - The optimal per-query forgetting coefficient along the prefill window is
    almost perfectly predicted by a single prefill-only quantity:
        aj_recent[i] = Jaccard( topk(attn[i, :]), topk(attn[most_recent, :]) )
    with an affine map of coefficients a ≈ 0.97, b ≈ -0.1 across diverse tasks.
  - Unlike A2SF / sigmoid / exponential, AJ is content-adaptive: each layer and
    KV-head gets its own forgetting curve, computed on the fly from attention
    itself. No tuning parameters (α, w) to search.

Algorithm (per layer, per KV-head):
  1. During prefill, observe softmax probs for queries inside the window
     (last ``window_size`` positions); store their top-k key indices.
  2. After prefill, the reference is the last window query's top-k.
     aj[pos] = |topk(pos) ∩ topk(ref)| / |topk(pos) ∪ topk(ref)|
     w[pos] = clip(aj[pos] - offset, 0, 1)
  3. Build aj-weighted key scores via scatter-add:
         score[k] = Σ_{pos ∈ window} w[pos] · 1[k ∈ topk(pos)]
  4. Apply the standard "top-k + always-keep recent" selection on score.

Queries BEFORE the window contribute zero weight (they're never observed).
"""
import torch

from .base import CompressionPolicy


class AjPolicy(CompressionPolicy):
    """Content-adaptive forgetting via top-k Jaccard to the most-recent query."""

    # ── weight functions (set via self.weight_fn string) ──
    VALID_WEIGHT_FNS = {
        "aj_offset",       # clip(aj - offset, 0, 1)
        "aj",              # raw aj
        "aj_sqrt",         # aj ** 0.5
        "aj_fastrise",     # 1 - (1-aj)**2
        "aj_quartic",      # aj ** 0.25
        "aj_floor30",      # max(aj, 0.3)
        "aj_floor50",      # max(aj, 0.5)
        "aj_floor70",      # max(aj, 0.7)
        "aj_mix50",        # 0.5 * aj + 0.5  (blend with uniform)
        "aj_mix25",        # 0.75 * aj + 0.25
        "aj_mix75",        # 0.25 * aj + 0.75
        "aj_gate30",       # aj if aj > 0.3 else 0
        "aj_gate50",       # aj if aj > 0.5 else 0
        "aj_sqrt_gate30",  # sqrt(aj) if aj > 0.3 else 0
        "aj_norm_sqrt",    # sqrt( (aj - min_per_head) / (max_per_head - min + eps) )
    }

    def __init__(
        self,
        num_key_value_heads: int,
        total_budget: int,
        window_size: int = 128,
        offset: float = 0.1,
        recent_budget: int = 16,
        weight_fn: str = "aj_offset",
    ):
        super().__init__(num_key_value_heads, total_budget, recent_budget)
        self.window_size = int(window_size)
        self.offset = float(offset)
        if weight_fn not in self.VALID_WEIGHT_FNS:
            raise ValueError(f"unknown weight_fn={weight_fn!r}; choose {self.VALID_WEIGHT_FNS}")
        self.weight_fn = weight_fn
        self._window_start_abs = 0
        self._seq_len_q = 0
        # Per-block buffers (cleared at the start of every prefill):
        self._topk_blocks: list = []   # (B, num_kv, qb_win, budget) int64 top-k indices
        self._prob_blocks: list = []   # (B, num_kv, qb_win, K)  fp/bf probs (GQA-reduced)

    # ---- lifecycle ----
    def prepare_prefill(self, seq_len_q, device, dtype):
        self._seq_len_q = int(seq_len_q)
        self._window_start_abs = max(0, self._seq_len_q - self.window_size)
        self._topk_blocks = []
        self._prob_blocks = []

    def needs_scores(self) -> bool:
        # We still need the score-accumulating path so we get `probs` in the kernel
        # (even though we won't use the accumulated scores). Once prefilled, the
        # fast SDPA path is used for decode.
        return not self.is_prefilled

    # ---- per-block probs hook (called from attention.py) ----
    def observe_probs(self, q_start: int, q_end: int, probs: torch.Tensor) -> None:
        """probs: (B, num_heads, qb, K) — softmax output for this Q-block.

        For queries INSIDE the window we store both:
          (a) per-query top-k indices (used to compute the aj weight)
          (b) the full per-query attention probs (used to build the final
              selection score: score[k] = Σ aj[pos] * probs[pos, k]).
        Heads are reduced to KV-head space by sum-over-group (standard GQA).
        """
        if q_end <= self._window_start_abs:
            return  # entire block is before window

        local_start = max(0, self._window_start_abs - q_start)
        win_probs = probs[:, :, local_start:, :]  # (B, H, qb_win, K)
        B, H, qb_win, K = win_probs.shape
        if qb_win == 0:
            return

        num_kv = self.num_key_value_heads
        if num_kv > 0 and H % num_kv == 0 and H != num_kv:
            group = H // num_kv
            # (B, num_kv, group, qb_win, K) → sum over group
            win_probs = win_probs.view(B, num_kv, group, qb_win, K).sum(dim=2)
        # else: either already in kv-head space, or shapes mismatch → use as-is

        k = min(self.total_budget, K)
        topk_idx = win_probs.topk(k, dim=-1).indices  # (B, num_kv, qb_win, k)
        self._topk_blocks.append(topk_idx.detach())
        self._prob_blocks.append(win_probs.detach())

    # ---- weighting hook (called from attention.py per block) ----
    def get_query_weights(self, q_start, q_end, device, dtype):
        # Return None → the kernel skips its score accumulation. We build scores
        # from our stored top-k masks in select().
        return None

    # ---- final selection ----
    def select(self, scores: torch.Tensor, seq_len_k: int):
        # Fallback if nothing was observed
        if not self._topk_blocks or not self._prob_blocks:
            return self._topk_with_recent(scores, seq_len_k)

        topk = torch.cat(self._topk_blocks, dim=2)          # (B, num_kv, W, budget)
        win_probs = torch.cat(self._prob_blocks, dim=2)     # (B, num_kv, W, K)
        B, num_kv, W, budget = topk.shape
        K = win_probs.size(-1)
        device = topk.device

        # ── aj weight per window query ──
        # Top-k masks
        ref_idx = topk[:, :, -1:, :]                         # (B, num_kv, 1, budget)
        ref_mask = torch.zeros(B, num_kv, 1, K, dtype=torch.bool, device=device)
        ref_mask.scatter_(-1, ref_idx, True)
        win_mask = torch.zeros(B, num_kv, W, K, dtype=torch.bool, device=device)
        win_mask.scatter_(-1, topk, True)

        inter = (win_mask & ref_mask).sum(-1).float()        # (B, num_kv, W)
        union = (win_mask | ref_mask).sum(-1).float().clamp(min=1)
        aj = inter / union                                   # (B, num_kv, W)

        # Configurable weight function
        if self.weight_fn == "aj_offset":
            w = (aj - self.offset).clamp(0.0, 1.0)
        elif self.weight_fn == "aj":
            w = aj
        elif self.weight_fn == "aj_sqrt":
            w = aj.clamp(min=0.0).sqrt()
        elif self.weight_fn == "aj_fastrise":
            w = 1.0 - (1.0 - aj).pow(2)
        elif self.weight_fn == "aj_quartic":
            w = aj.clamp(min=0.0).pow(0.25)
        elif self.weight_fn == "aj_floor30":
            w = aj.clamp(min=0.3)
        elif self.weight_fn == "aj_floor50":
            w = aj.clamp(min=0.5)
        elif self.weight_fn == "aj_floor70":
            w = aj.clamp(min=0.7)
        elif self.weight_fn == "aj_mix50":
            w = 0.5 * aj + 0.5
        elif self.weight_fn == "aj_mix25":
            w = 0.75 * aj + 0.25
        elif self.weight_fn == "aj_mix75":
            w = 0.25 * aj + 0.75
        elif self.weight_fn == "aj_gate30":
            w = torch.where(aj > 0.3, aj, torch.zeros_like(aj))
        elif self.weight_fn == "aj_gate50":
            w = torch.where(aj > 0.5, aj, torch.zeros_like(aj))
        elif self.weight_fn == "aj_sqrt_gate30":
            s = aj.clamp(min=0.0).sqrt()
            w = torch.where(aj > 0.3, s, torch.zeros_like(s))
        elif self.weight_fn == "aj_norm_sqrt":
            # Per (B, num_kv) min-max normalise over the window dim, then sqrt
            amin = aj.amin(dim=-1, keepdim=True)
            amax = aj.amax(dim=-1, keepdim=True)
            w = ((aj - amin) / (amax - amin + 1e-8)).clamp(min=0.0).sqrt()
        else:
            w = aj  # pragma: no cover

        # ── aj-weighted continuous score  (SnapKV-style sum, but per-query weighted) ──
        #   score[k] = Σ_pos w[pos] · probs[pos, k]
        aj_scores = (w.unsqueeze(-1) * win_probs.float()).sum(dim=2)   # (B, num_kv, K)

        # Free the per-block buffers so we don't carry them across layers
        self._topk_blocks.clear()
        self._prob_blocks.clear()

        return self._topk_with_recent(aj_scores, seq_len_k)
