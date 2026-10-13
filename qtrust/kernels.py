"""Triton kernels for Q-Trust, reusing the FlashAttention-2 row log-sum-exp (LSE).

With LSE_q = logsumexp_i(q . k_i * scale) returned by the prefill forward, the exact causal attention
probability is A(q, i) = exp(q . k_i * scale - LSE_q), so neither kernel recomputes a softmax.
Both kernels are key-major (one program per key block and KV head), like the dK/dV pass of the
FlashAttention-2 backward: each program owns its output slots, so no atomics are needed. GQA is handled
by indexing the KV head of query head h as h // G (no repeat_kv copy).

  window_score : s(i)     = sum_{t in last W} sum_{h in group} A_h(t, i)                  (first hop)
  hop2_score   : score(i) = sum_{q >= i} w(q) sum_{h in group} A_h(q, i)                  (second hop)
                 dense over all queries, or sparse over a sorted list of query positions

Shapes (batch 1, prefill, Sq == Sk == N):
  q [1, H, N, D]   k [1, KV, N, D]   lse [1, H, N] fp32   w [KV, N] fp32   ->   out [1, KV, N] fp32
"""
import functools

import torch
import triton
import triton.language as tl


@triton.jit
def _window_kernel(Q, K, LSE, OUT,
                   sqh, sqs, sqd, skh, sks, skd, slh, sls,
                   N, scale,
                   G: tl.constexpr, W: tl.constexpr, WP: tl.constexpr, D: tl.constexpr, BLOCK_K: tl.constexpr):
    jb = tl.program_id(0)
    kv = tl.program_id(1)
    offs_k = jb * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, D)
    kmask = offs_k < N
    k = tl.load(K + kv * skh + offs_k[:, None] * sks + offs_d[None, :] * skd, mask=kmask[:, None], other=0.0)
    # WP >= 16 rows so tl.dot is legal even for a window smaller than 16; the padded rows are masked off.
    # A window longer than the prompt is clipped to the prompt here, at run time, so N never forces a recompile.
    offs_t = N - WP + tl.arange(0, WP)                                 # window query positions (padded)
    tmask = offs_t >= tl.maximum(N - W, 0)
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for gi in tl.static_range(G):
        h = kv * G + gi
        q = tl.load(Q + h * sqh + offs_t[:, None] * sqs + offs_d[None, :] * sqd, mask=tmask[:, None], other=0.0)
        lse = tl.load(LSE + h * slh + offs_t * sls, mask=tmask, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale                            # [WP, BLOCK_K] fp32
        p = tl.exp(s - lse[:, None])
        valid = (offs_k[None, :] <= offs_t[:, None]) & kmask[None, :] & tmask[:, None]
        acc += tl.sum(tl.where(valid, p, 0.0), axis=0)
    tl.store(OUT + kv * N + offs_k, acc, mask=kmask)


LOG2E = 1.4426950408889634


@triton.jit
def _hop2_tile(acc, Q, LSE, k, pos, rmask, w, offs_k, kmask, kv, sqh, sqs, sqd, slh, sls, qk_scale,
               G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, BQ: tl.constexpr, MASK: tl.constexpr):
    # rows r = gi * BQ + t stack the G query heads of this KV head: one [GP*BQ, D] x [D, BK] product.
    # GP is G rounded up to a power of two (tl.arange needs one); the gi >= G rows are masked off, which
    # is what lets an odd GQA ratio such as Qwen's 7 compile at all.
    r = tl.arange(0, GP * BQ)
    gi = r // BQ
    t = r % BQ
    p_r = tl.reshape(tl.broadcast_to(pos[None, :], (GP, BQ)), (GP * BQ,))
    m_r = tl.reshape(tl.broadcast_to(rmask[None, :], (GP, BQ)), (GP * BQ,)) & (gi < G)
    w_r = tl.reshape(tl.broadcast_to(w[None, :], (GP, BQ)), (GP * BQ,))
    h = kv * G + tl.minimum(gi, G - 1)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + h[:, None] * sqh + p_r[:, None] * sqs + offs_d[None, :] * sqd, mask=m_r[:, None], other=0.0)
    lse = tl.load(LSE + h * slh + p_r * sls, mask=m_r, other=0.0)
    s = tl.dot(q, tl.trans(k)) * qk_scale                              # log2-domain logits (qk_scale = scale*log2e)
    p = tl.exp2(s - lse[:, None] * 1.4426950408889634)                 # = exp(q.k*scale - LSE) = A(q, i)
    if MASK:
        valid = (offs_k[None, :] <= p_r[:, None]) & m_r[:, None] & kmask[None, :]
        p = tl.where(valid, p, 0.0)
    else:
        p = tl.where(m_r[:, None], p, 0.0)
    return acc + tl.sum(p * w_r[:, None], axis=0)


@triton.jit
def _hop2_dense_kernel(Q, K, LSE, Wt, OUT,
                       sqh, sqs, sqd, skh, sks, skd, slh, sls,
                       N, qk_scale,
                       G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr):
    kv = tl.program_id(0)                                              # KV fastest: long (early) key blocks first
    jb = tl.program_id(1)
    k_start = jb * BK
    offs_k = k_start + tl.arange(0, BK)
    offs_d = tl.arange(0, D)
    kmask = offs_k < N
    k = tl.load(K + kv * skh + offs_k[:, None] * sks + offs_d[None, :] * skd, mask=kmask[:, None], other=0.0)
    acc = tl.zeros([BK], dtype=tl.float32)
    q_begin = (k_start // BQ) * BQ
    diag_end = tl.minimum(((k_start + BK + BQ - 1) // BQ) * BQ, N)     # query blocks that overlap the key block
    for qs in range(q_begin, diag_end, BQ):                            # diagonal part: causal mask
        pos = qs + tl.arange(0, BQ)
        rmask = pos < N
        w = tl.load(Wt + kv * N + pos, mask=rmask, other=0.0)
        acc = _hop2_tile(acc, Q, LSE, k, pos, rmask, w, offs_k, kmask, kv, sqh, sqs, sqd, slh, sls, qk_scale,
                         G=G, GP=GP, D=D, BQ=BQ, MASK=True)
    for qs in range(diag_end, N, BQ):                                  # strictly below: no causal mask
        pos = qs + tl.arange(0, BQ)
        rmask = pos < N
        w = tl.load(Wt + kv * N + pos, mask=rmask, other=0.0)
        acc = _hop2_tile(acc, Q, LSE, k, pos, rmask, w, offs_k, kmask, kv, sqh, sqs, sqd, slh, sls, qk_scale,
                         G=G, GP=GP, D=D, BQ=BQ, MASK=False)
    tl.store(OUT + kv * N + offs_k, acc, mask=kmask)


@triton.jit
def _hop2_sparse_kernel(Q, K, LSE, Wt, OUT, QIDX, QSTART, QDIAG,
                         sqh, sqs, sqd, skh, sks, skd, slh, sls,
                         N, M, qk_scale,
                         G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr):
    kv = tl.program_id(0)
    jb = tl.program_id(1)
    offs_k = jb * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, D)
    kmask = offs_k < N
    k = tl.load(K + kv * skh + offs_k[:, None] * sks + offs_d[None, :] * skd, mask=kmask[:, None], other=0.0)
    acc = tl.zeros([BK], dtype=tl.float32)
    start = tl.load(QSTART + jb)                                       # first listed query >= block start
    diag = tl.load(QDIAG + jb)                                         # first listed query >= block end
    for qb in range(start, diag, BQ):                                  # rows that may precede some key: masked
        offs = qb + tl.arange(0, BQ)
        rmask = offs < diag
        pos = tl.load(QIDX + offs, mask=rmask, other=0)
        w = tl.load(Wt + kv * N + pos, mask=rmask, other=0.0)
        acc = _hop2_tile(acc, Q, LSE, k, pos, rmask, w, offs_k, kmask, kv, sqh, sqs, sqd, slh, sls, qk_scale,
                         G=G, GP=GP, D=D, BQ=BQ, MASK=True)
    for qb in range(diag, M, BQ):                                      # all rows after the block: no mask
        offs = qb + tl.arange(0, BQ)
        rmask = offs < M
        pos = tl.load(QIDX + offs, mask=rmask, other=0)
        w = tl.load(Wt + kv * N + pos, mask=rmask, other=0.0)
        acc = _hop2_tile(acc, Q, LSE, k, pos, rmask, w, offs_k, kmask, kv, sqh, sqs, sqd, slh, sls, qk_scale,
                         G=G, GP=GP, D=D, BQ=BQ, MASK=False)
    tl.store(OUT + kv * N + offs_k, acc, mask=kmask)


def _on_input_device(fn):
    """Launch on the device that holds the inputs (layers of a device_map="auto" model sit on several GPUs)."""
    @functools.wraps(fn)
    def wrapped(q, *args, **kwargs):
        with torch.cuda.device(q.device):
            return fn(q, *args, **kwargs)
    return wrapped


def _check(q, k, lse):
    assert q.dim() == 4 and q.shape[0] == 1 and k.shape[0] == 1, "batch 1 only"
    H, N, D = q.shape[1], q.shape[2], q.shape[3]
    KV = k.shape[1]
    assert k.shape[2] == N and H % KV == 0 and lse.shape[-1] == N
    return H, KV, N, D


@_on_input_device
def window_score(q, k, lse, scale, window=16, block_k=64):
    """s(i) = sum over the last `window` queries and the GQA group of A(t, i). Returns [1, KV, N] fp32."""
    H, KV, N, D = _check(q, k, lse)
    wp = max(int(window), 16)                # tl.dot needs at least 16 rows; W > N is clipped in the kernel
    assert N >= 1
    lse = lse.reshape(H, N)
    out = torch.zeros(KV, N, device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(N, block_k), KV)
    _window_kernel[grid](q, k, lse, out,
                         q.stride(1), q.stride(2), q.stride(3), k.stride(1), k.stride(2), k.stride(3),
                         lse.stride(0), lse.stride(1), N, float(scale),
                         G=H // KV, W=window, WP=wp, D=D, BLOCK_K=block_k, num_warps=4)
    return out.unsqueeze(0)



# launch parameters tuned on an RTX 3090 at 16K tokens
BQ, BK = 16, 128                       # query rows per tile (per head), keys per program


@_on_input_device
def hop2_score(q, k, lse, w, scale, qidx=None):
    """score(i) = sum_{q >= i} w(q) sum_{h in group} A_h(q, i). w: [KV, N] fp32 (per KV head).
    qidx: optional sorted tensor of query positions to visit (sparse second hop). Returns [1, KV, N] fp32."""
    H, KV, N, D = _check(q, k, lse)
    G = H // KV
    GP = triton.next_power_of_2(G)                 # tl.arange needs a power of two; padded rows are masked
    lse = lse.reshape(H, N)
    w = w.to(torch.float32).contiguous()
    out = torch.zeros(KV, N, device=q.device, dtype=torch.float32)
    grid = (KV, triton.cdiv(N, BK))
    common = (q.stride(1), q.stride(2), q.stride(3), k.stride(1), k.stride(2), k.stride(3), lse.stride(0), lse.stride(1))
    qk_scale = float(scale) * LOG2E
    if qidx is None:
        _hop2_dense_kernel[grid](q, k, lse, w, out, *common, N, qk_scale, G=G, GP=GP, D=D, BQ=BQ, BK=BK,
                                 num_warps=4, num_stages=2)
    else:
        qidx = qidx.to(device=q.device, dtype=torch.int32).contiguous()
        kst = torch.arange(triton.cdiv(N, BK), device=q.device, dtype=torch.int32) * BK
        qstart = torch.searchsorted(qidx, kst).to(torch.int32)       # first listed query >= block start
        qdiag = torch.searchsorted(qidx, kst + BK).to(torch.int32)   # first listed query >= block end
        _hop2_sparse_kernel[grid](q, k, lse, w, out, qidx, qstart, qdiag, *common, N, qidx.numel(), qk_scale,
                                  G=G, GP=GP, D=D, BQ=BQ, BK=BK, num_warps=4, num_stages=1)
    return out.unsqueeze(0)
