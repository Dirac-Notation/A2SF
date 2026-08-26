"""KV-cache compression (transformers v5), model-agnostic.

The compression mechanism lives in `compress.py`: ONE custom attention function
("waits") registered into transformers' `ALL_ATTENTION_FUNCTIONS` plus a
`CompressedCache`, reusing `scorers/` + `selectors/`. There are no per-model
reimplementations. Load any HF model via `utils.load_model(shortname)` /
`utils.load_compressed_lm(path)`; the returned model exposes the
`model.init_cache(cfg)` + `model.generate(...)` interface.
"""
from utils_real_drop.compress import (
    CompressionConfig,
    CompressedCache,
    init_cache,
    make_cache,
    attach_pipeline_api,
    load_pipeline_model,
    load_compressed_model,
)

__all__ = [
    "CompressionConfig",
    "CompressedCache",
    "init_cache",
    "make_cache",
    "attach_pipeline_api",
    "load_pipeline_model",
    "load_compressed_model",
]
