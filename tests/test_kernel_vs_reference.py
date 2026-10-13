"""The Triton path must reproduce the PyTorch reference.

    python tests/test_kernel_vs_reference.py          (needs a CUDA GPU with Triton)

Random bf16 queries/keys with GQA; the LSE comes from the same FlashAttention-2 call used in qtrust.hf.
"""
import sys, os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qtrust.reference import qtrust_scores, select_keys
from qtrust.fast import qtrust_scores_fast


def run(N, H=32, KV=8, D=128, tau=1.0, window_weight="one", seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(1, H, N, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, KV, N, D, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, KV, N, D, device="cuda", dtype=torch.bfloat16, generator=g)
    scale = D ** -0.5
    G = H // KV
    _, lse = torch.ops.aten._scaled_dot_product_flash_attention(
        q, k.repeat_interleave(G, 1), v.repeat_interleave(G, 1), 0.0, True, False, scale=scale)[:2]
    ref = qtrust_scores(q[0], k[0], window=16, tau=tau, window_weight=window_weight, scale=scale)
    fast = qtrust_scores_fast(q, k, lse, scale, window=16, tau=tau, window_weight=window_weight)
    rel = ((fast - ref).abs().max() / ref.abs().max()).item()
    a, b = select_keys(ref, 128), select_keys(fast, 128)
    overlap = sum(len(set(a[h].tolist()) & set(b[h].tolist())) for h in range(KV)) / a.numel()
    print(f"N={N:6d} tau={tau} weight={window_weight}: max rel. error {rel:.2e}, kept-key overlap {overlap:.4f}")
    assert rel < 1e-3 and overlap > 0.99


if __name__ == "__main__":
    for N in (1000, 4096):
        for tau in (0.0, 1.0):
            for ww in ("one", "s"):
                run(N, tau=tau, window_weight=ww)
    print("OK")
