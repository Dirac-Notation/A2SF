# The per-model reimplementations (kv_llama/kv_qwen/kv_opt) target transformers
# 4.46.2 internals; under transformers v5 those APIs are gone, so guard the import.
# The v5 path (utils_real_drop.v5_compress) uses only scorers/ + selectors/.
try:
    from utils_real_drop.kv_llama import KVLlamaForCausalLM
    from utils_real_drop.kv_qwen import KVQwen2ForCausalLM
    from utils_real_drop.kv_opt import KVOPTForCausalLM
    from utils_real_drop.cache import CompressedKVCache
    _KV_446_AVAILABLE = True
except Exception:  # e.g. transformers v5
    _KV_446_AVAILABLE = False

# HF config.model_type -> KV-compression model class (4.46.2 path only).
_MODEL_TYPE_TO_KV_CLASS = {
    "llama": KVLlamaForCausalLM,
    "qwen2": KVQwen2ForCausalLM,
    "opt": KVOPTForCausalLM,
} if _KV_446_AVAILABLE else {}


def get_kv_class(model_type: str):
    """Return the KV-compression model class for an HF `config.model_type`."""
    if model_type not in _MODEL_TYPE_TO_KV_CLASS:
        raise ValueError(
            f"Unsupported model_type {model_type!r}. "
            f"Supported: {sorted(_MODEL_TYPE_TO_KV_CLASS)}"
        )
    return _MODEL_TYPE_TO_KV_CLASS[model_type]


def load_kv_model(model_path: str, **from_pretrained_kwargs):
    """Load the right KV-compression model for `model_path` by inspecting its config."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_path)
    return get_kv_class(cfg.model_type).from_pretrained(model_path, **from_pretrained_kwargs)


__all__ = [
    "KVLlamaForCausalLM",
    "KVQwen2ForCausalLM",
    "KVOPTForCausalLM",
    "CompressedKVCache",
    "get_kv_class",
    "load_kv_model",
]
