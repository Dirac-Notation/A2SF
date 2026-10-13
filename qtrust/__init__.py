"""Q-Trust: request-guided query weighting for KV cache compression.

    qtrust.reference : the algorithm in plain PyTorch (read this first)
    qtrust.kernels   : Triton kernels for the two score passes
    qtrust.fast      : scores with the kernels (same result as the reference)
    qtrust.hf        : drop-in use with Hugging Face transformers v5 (import it to register "qtrust")
"""
from .reference import first_hop, query_weights, second_hop, qtrust_scores, select_keys

__all__ = ["first_hop", "query_weights", "second_hop", "qtrust_scores", "select_keys"]
