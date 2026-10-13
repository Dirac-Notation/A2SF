import math

import torch

from .base import Scorer


class QTrustScorer(Scorer):
    """Q-Trust: the request's attention decides which prefill queries are trusted.

    First hop (computed by compress.py and handed over through `set_first_hop`):
        s(q) = sum of the attention the last W queries (the request) give to position q   (= SnapKV score)
    Query weights (per KV head):
        window_one=False : w(q) = s(q) / sum_q s(q)
        window_one=True  : request queries get raw weight 1, the others s(q), then normalized   (paper, Eq. 6)
    Second hop:
        score(i) = sum_{q >= i} w(q) A(q, i)

    Options:
      temperature = T -> w(q) ∝ s(q)^(1/T)  (T -> inf: uniform weights = H2O accumulation). Needs window_one=False.
      skip        = τ -> queries with w(q) < τ / N are dropped from the second hop; if no KV head keeps a query,
                         its attention row is never computed. 0 = dense.

    Hooks used only by the observation experiments (experiments/observation):
      ext_sel [num_kv, N] bool  -> only these document queries enter the sum; the document mass the weighting gave
                                   is spread evenly over them (ext_sel_uniform=True: sinks, window and the selected
                                   document queries all get the same weight, i.e. an H2O-style sum over a chosen set).
      ext_doc_w [num_kv, n_doc] -> replace the document-query weights by given non-negative weights.
    """

    def __init__(self, num_key_value_heads: int, observation_window: int, temperature: float = 1.0,
                 skip: float = 0.0, window_one: bool = False):
        super().__init__(num_key_value_heads)
        self.first_hop_window = int(observation_window)
        self.temperature = float(temperature)
        self.skip = float(skip)
        self.window_one = bool(window_one)
        if self.window_one and self.temperature != 1.0:
            raise ValueError("window_one is only defined with temperature=1")
        self.ext_sel = None
        self.ext_sel_uniform = False
        self.ext_doc_w = None
        self._w = None
        self._qidx = None
        self.kept_query_frac = []           # per layer: share of query rows computed (selective computation)

    def set_first_hop(self, scores):
        if scores.shape[0] != 1:
            raise ValueError("QTrustScorer supports batch size 1")
        s = scores[0].to(torch.float32)
        T = self.temperature
        if math.isinf(T):
            w = torch.ones_like(s)
            w = w / w.sum(dim=-1, keepdim=True)
        elif T != 1.0:
            # s^(1/T) normalized == softmax(log s / T); log-space so small T does not underflow
            logit = torch.log(s.clamp_min(0.0)) / T                     # zeros -> -inf
            w = torch.softmax(logit, dim=-1).nan_to_num(0.0)
        else:
            w = s / s.sum(dim=-1, keepdim=True).clamp_min(1e-12)     # [num_kv, Sk]
        if self.window_one:
            n, W = s.shape[-1], self.first_hop_window
            raw = s.clone(); raw[:, n - W:] = 1.0
            w = raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if self.ext_sel is not None:
            w = self._apply_ext_sel(w)
        if self.ext_doc_w is not None:
            self._w = w; self.set_content_weights(self.ext_doc_w.to(w.device)); w = self._w
        self._qidx = None
        self._w = self._apply_skip(w)

    def _apply_skip(self, w):
        """Selective computation: zero the queries below skip x uniform share and record the rows to compute."""
        if self.skip > 0:
            keep = w >= self.skip / w.shape[-1]
            if self.window_one:
                keep[:, w.shape[-1] - self.first_hop_window:] = True
            w = w * keep
            self._qidx = keep.any(dim=0).nonzero().flatten()          # union over KV heads, sorted
            self.kept_query_frac.append(float(self._qidx.numel()) / w.shape[-1])
        return w

    def _apply_ext_sel(self, w, n_sink=4, n_win=16):
        n = w.shape[-1]
        doc = torch.zeros(n, dtype=torch.bool, device=w.device); doc[n_sink:n - n_win] = True
        sel = self.ext_sel.to(w.device) & doc
        if self.ext_sel_uniform:
            out = (sel | ~doc).to(w.dtype).expand_as(w).clone()
            return out / out.sum(-1, keepdim=True).clamp_min(1e-30)
        out = w.clone(); out[:, doc] = 0.0
        m = (w * doc).sum(-1)                                    # dense document mass, per KV head
        k = sel.sum(-1).clamp_min(1)
        out = out + sel * (m / k).unsqueeze(-1)
        return out / out.sum(-1, keepdim=True).clamp_min(1e-30)

    def set_content_weights(self, doc_w, n_sink=4, n_win=16, apply_skip=False):
        """Replace the document-query weights by doc_w [num_kv, n_doc] (>= 0), keeping the document mass."""
        w = self._w.clone()
        n = w.shape[-1]
        m = w[:, n_sink:n - n_win].sum(-1, keepdim=True)
        d = doc_w.clamp_min(0).to(w.dtype)
        w[:, n_sink:n - n_win] = d / d.sum(-1, keepdim=True).clamp_min(1e-30) * m
        self._qidx = None
        self._w = self._apply_skip(w) if apply_skip else w

    def query_indices(self):
        """Query positions whose rows must be computed (None = all, dense path)."""
        return self._qidx

    def get_query_weights_at(self, idx):
        return self._w[:, idx]

    def is_per_head(self):
        return True

    def get_query_weights(self, q_start, q_end, device, dtype):
        return self._w[:, q_start:q_end]

    def finalize_prefill(self):
        super().finalize_prefill()
        self._w = None
        self._qidx = None
