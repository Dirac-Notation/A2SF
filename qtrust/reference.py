"""Q-Trust reference implementation in plain PyTorch.

This file is the specification. It favors clarity over speed: it materializes attention
probabilities block by block. `qtrust.fast` computes the same scores with Triton kernels.

Notation (one layer, batch 1, prefill of N tokens):
    q : [H, N, D]  queries after RoPE        (H query heads)
    k : [KV, N, D] keys after RoPE           (KV key/value heads, H = KV * G under GQA)
    A_h(t, i)      causal attention probability of query t to key i in query head h

Q-Trust scores every key i of each KV head in two hops:

    s(j)   = sum_{t in last W} sum_{h in group} A_h(t, j)            first hop  (= SnapKV score)
    w(q)  ∝ 1     if q is in the last W queries (the request)
            s(q)  otherwise                                          query weight, sum_q w(q) = 1
    score(i) = sum_{q >= i} w(q) sum_{h in group} A_h(q, i)          second hop (weighted accumulation)

Because sum_q s(q) = W * G for each KV head, score(i) is proportional to
    s(i) + sum_{q not in last W} s(q) A(q, i),
i.e. the request's direct attention plus the attention of the queries the request attends to.

Selective computation: queries with w(q) < tau / N (tau times the uniform weight) are dropped
from the second hop; the window queries are always kept. tau = 0 is the dense score.

Selection: per KV head, the last `recent` keys are always kept and the remaining
`budget - recent` slots go to the highest-scoring keys among [0, N - recent).
"""
import torch


def _attention_rows(q, k, t0, t1, scale):
    """Causal attention probabilities of query rows [t0, t1) to all N keys: [H, t1 - t0, N] fp32."""
    H, N, _ = q.shape
    G = H // k.shape[0]
    kk = k.repeat_interleave(G, dim=0)                                      # [H, N, D]
    logits = torch.matmul(q[:, t0:t1].float(), kk.float().transpose(-1, -2)) * scale
    rows = torch.arange(t0, t1, device=q.device)[:, None]
    cols = torch.arange(N, device=q.device)[None, :]
    logits = logits.masked_fill(cols > rows, float("-inf"))                 # causal mask
    return torch.softmax(logits, dim=-1)


def first_hop(q, k, window=16, scale=None):
    """s(j): attention the last `window` queries pay to each key, summed over the GQA group. [KV, N]."""
    H, N, D = q.shape
    KV = k.shape[0]
    scale = D ** -0.5 if scale is None else scale
    W = min(window, N)
    A = _attention_rows(q, k, N - W, N, scale)                             # [H, W, N]
    return A.view(KV, H // KV, W, N).sum(dim=(1, 2))


def query_weights(s, window=16, tau=1.0, window_weight="one"):
    """w(q) per KV head from the first-hop score s [KV, N].

    window_weight = "one": request (window) queries get raw weight 1, others s(q)   (paper, Eq. 6)
                    "s"  : every query, including the window, gets raw weight s(q)
    Returns (w [KV, N], keep [KV, N] bool): the normalized weights after dropping queries below tau / N.
    """
    N = s.shape[-1]
    W = min(window, N)
    raw = s.float().clone()
    if window_weight == "one":
        raw[:, N - W:] = 1.0
    elif window_weight != "s":
        raise ValueError(f"window_weight must be 'one' or 's', got {window_weight!r}")
    w = raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    keep = torch.ones_like(w, dtype=torch.bool)
    if tau > 0:
        keep = w >= tau / N
        keep[:, N - W:] = True                                              # the request is always computed
        w = w * keep
    return w, keep


def second_hop(q, k, w, scale=None, block=256):
    """score(i) = sum_{q >= i} w(q) sum_{h in group} A_h(q, i). w: [KV, N]. Returns [KV, N] fp32."""
    H, N, D = q.shape
    KV = k.shape[0]
    G = H // KV
    scale = D ** -0.5 if scale is None else scale
    score = torch.zeros(KV, N, device=q.device, dtype=torch.float32)
    for t0 in range(0, N, block):
        t1 = min(t0 + block, N)
        wb = w[:, t0:t1]
        if not torch.any(wb > 0):
            continue                                                        # every row of this block is dropped
        A = _attention_rows(q, k, t0, t1, scale).view(KV, G, t1 - t0, N)
        score += torch.einsum("kq,kgqn->kn", wb.float(), A)
    return score


def qtrust_scores(q, k, window=16, tau=1.0, window_weight="one", scale=None):
    """Q-Trust key scores for one layer. q [H, N, D], k [KV, N, D] -> [KV, N] fp32."""
    s = first_hop(q, k, window, scale)
    w, _ = query_weights(s, window, tau, window_weight)
    return second_hop(q, k, w, scale)


def select_keys(score, budget, recent=16):
    """Kept key indices per KV head, sorted: top (budget - recent) of [0, N - recent) + the last `recent`.
    score [KV, N] -> [KV, budget] (or None if N <= budget, i.e. nothing to evict)."""
    KV, N = score.shape
    if N <= budget:
        return None
    head = N - recent
    top = score[:, :head].topk(budget - recent, dim=-1).indices.sort(dim=-1).values
    tail = torch.arange(head, N, device=score.device).expand(KV, recent)
    return torch.cat([top, tail], dim=-1)
