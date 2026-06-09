from utils_real_drop.kv_llama import KVLlamaForCausalLM
from utils_real_drop.kv_qwen import KVQwen2ForCausalLM
from utils_real_drop.kv_opt import KVOPTForCausalLM
from utils_real_drop.cache import CompressedKVCache

# HF config.model_type -> KV-compression model class.
_MODEL_TYPE_TO_KV_CLASS = {
    "llama": KVLlamaForCausalLM,
    "qwen2": KVQwen2ForCausalLM,
    "opt": KVOPTForCausalLM,
}


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
