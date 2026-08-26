"""TriAttention-style scorer using calibrated representative query statistics.

Scoring formula (pytorch, "half" RoPE layout):
    score(k_pos) = Σ_f freq_scale²[f] * (A*cos(t*ω[f]) - B*sin(t*ω[f])) + extra

where:
    A + iB  = Σ_f q_mean_complex[f] * conj(K_rot[k_pos, f])
    extra   = Σ_f (q_abs_mean[f] - |q_mean_complex[f]|) * |K_rot[k_pos, f]| * freq_scale²[f]
    t       = representative query position (calibrated mean seq length)

Key difference from Q-tiled loop: scores are computed in O(N) from calibrated
q_mean stats + actual K_rot, without needing the actual Q tensor.
This allows the attention output to use FlashAttention (no Q-tiled loop).
"""

import torch

from .base import Scorer


class TriAttentionScorer(Scorer):
    """Prefill-time KV scoring using calibrated mean-query statistics.

    Stats are loaded once per layer from a shared dict (produced by
    script/calibrate_triattention.py):
        q_mean_real  [num_layers, num_kv_heads, freq_count]
        q_mean_imag  [num_layers, num_kv_heads, freq_count]
        q_abs_mean   [num_layers, num_kv_heads, freq_count]
        omega        [freq_count]
        rep_position (scalar float)
        freq_scale_sq  [num_layers, num_kv_heads, freq_count]  (optional)
    """

    def __init__(self, num_kv_heads: int, layer_idx: int, stats: dict):
        super().__init__(num_kv_heads)
        L = layer_idx
        self.layer_idx = layer_idx

        self._q_mean_real  = stats["q_mean_real"][L]   # [H, F]
        self._q_mean_imag  = stats["q_mean_imag"][L]   # [H, F]
        self._q_abs_mean   = stats["q_abs_mean"][L]    # [H, F]
        self._omega        = stats["omega"]            # [F]
        self._rep_position = float(stats.get("rep_position", 512))

        fss = stats.get("freq_scale_sq", None)
        self._freq_scale_sq = fss[L] if fss is not None else None  # [H, F] or None

        self._precomputed_scores: torch.Tensor | None = None

    def prepare_prefill(self, seq_len_q, device, dtype, key=None, num_kv=None, **kwargs):
        """Compute per-key scores from calibrated Q stats + actual K_rot.

        key: [batch, num_q_heads, seq_len_k, head_dim]  (post-repeat_kv, post-RoPE)
        num_kv: number of KV heads in the original cache (before repeat)
        """
        self._precomputed_scores = None
        if key is None:
            return

        batch, num_q_heads, seq_len_k, head_dim = key.shape
        H = self.num_key_value_heads
        group = num_q_heads // H if num_q_heads > H else 1
        # Take first head of each group to recover KV-head space
        K = key[:, ::group, :, :].contiguous()  # [batch, H, seq_len_k, head_dim]

        freq_count = head_dim // 2
        # LLaMA "half" RoPE layout: real = K[..., :F], imag = K[..., F:]
        k_real = K[..., :freq_count].float()   # [B, H, N, F]
        k_imag = K[..., freq_count:].float()   # [B, H, N, F]

        q_r  = self._q_mean_real.to(device=device, dtype=torch.float32)   # [H, F]
        q_i  = self._q_mean_imag.to(device=device, dtype=torch.float32)   # [H, F]
        q_ab = self._q_abs_mean.to(device=device, dtype=torch.float32)    # [H, F]
        w    = self._omega.to(device=device, dtype=torch.float32)         # [F]

        # Q_mean * conj(K_rot): [B, H, N, F]
        q_r4 = q_r[None, :, None, :]
        q_i4 = q_i[None, :, None, :]
        prod_real = q_r4 * k_real + q_i4 * k_imag
        prod_imag = q_i4 * k_real - q_r4 * k_imag

        # freq_scale_sq weighting (optional)
        if self._freq_scale_sq is not None:
            fs = self._freq_scale_sq.to(device=device, dtype=torch.float32)  # [H, F]
            prod_real = prod_real * fs[None, :, None, :]
            prod_imag = prod_imag * fs[None, :, None, :]

        # RoPE phase for representative query position t
        t     = self._rep_position
        phase = t * w                            # [F]
        c     = torch.cos(phase)[None, None, None, :]  # broadcast [1,1,1,F]
        s     = torch.sin(phase)[None, None, None, :]

        base_scores = (prod_real * c - prod_imag * s).sum(dim=-1)  # [B, H, N]

        # Magnitude-linear-regression (MLR) additive term
        q_mean_abs = torch.sqrt(q_r ** 2 + q_i ** 2 + 1e-8)  # [H, F]
        k_abs      = torch.sqrt(k_real ** 2 + k_imag ** 2)    # [B, H, N, F]
        extra_coef = (q_ab - q_mean_abs)[None, :, None, :]    # [1, H, 1, F]
        if self._freq_scale_sq is not None:
            fs = self._freq_scale_sq.to(device=device, dtype=torch.float32)
            extra_coef = extra_coef * fs[None, :, None, :]
        extra = (extra_coef * k_abs).sum(dim=-1)               # [B, H, N]

        self._precomputed_scores = base_scores + extra         # [B, H, N]  fp32

    def score_keys(self, query, key, num_kv):
        # attention-free: scores were computed in prepare_prefill from calibrated Q stats + K
        return self._precomputed_scores  # [B, num_kv, Sk] fp32

    def get_query_weights(self, q_start, q_end, device, dtype):
        return None  # not used — fast path bypasses Q-tiled loop
