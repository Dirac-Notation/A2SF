# A2SF (Accumulative Attention Score with Forgetting)

## Overview

A key-value (KV) cache compression technique using accumulative attention scores
with a forgetting mechanism. The repo also ships SnapKV and a sigmoid-window
variant under the same compression interface, plus a Reinforcement Learning
agent that learns per-layer compression parameters.

---

## Preparation

### Python Version

- Python 3.8 (other versions are untested).

### Environment setup

```bash
conda env create -n A2SF python=3.8
conda activate A2SF
pip install -r pip.txt
```

Key pinned dependencies: `transformers==4.46.2`, `datasets<4.0.0`,
`sentence-transformers==2.7.0`.

---

## Example

```python
import torch
from types import SimpleNamespace
from transformers import AutoTokenizer

from utils_real_drop import KVLlamaForCausalLM

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = KVLlamaForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16
).to("cuda").eval()

# Pass `compression_config=None` for the un-compressed baseline.
# For A2SF compression, build a config object with the fields the policy needs:
compression_config = SimpleNamespace(
    compression_method="a2sf",   # one of: "full"/None, "a2sf", "snap", "sigmoid"
    total_budget=256,            # total tokens kept per layer (recent_budget=16 fixed)
    forgetting_factor=0.95,
)
model.init_cache(compression_config)

prompt = "Summarize the following passage in one sentence: ..."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

`compression_method` accepts `"full"` (or `None`), `"a2sf"`, `"snap"`,
`"sigmoid"`. Each method needs a few extra fields on the config object:

| method     | required fields                                |
|------------|------------------------------------------------|
| `a2sf`     | `total_budget`, `forgetting_factor`            |
| `snap`     | `total_budget`, `observation_window`           |
| `sigmoid`  | `total_budget`, `a`, `b`                       |

---

## Architecture (`utils_real_drop/`)

```
utils_real_drop/
├── kv_llama.py      # KVLlamaForCausalLM — Llama wiring (RoPE, repeat_kv, layer loop)
├── attention.py     # compressed_attention(q, k, v, *, policy, ...) — model-agnostic kernel
├── cache.py         # CompressedKVCache — HF-compatible KV storage + compress hook
└── policies/
    ├── base.py      # CompressionPolicy abstract base
    ├── a2sf.py
    ├── snap.py
    ├── sigmoid.py
    └── __init__.py  # build_policies() dispatcher / registry
```

The attention kernel uses K-tiled online softmax (real flash-attention style),
so prefill memory does **not** scale with the full sequence length; only with
the q-block × k-block tile. Causal masking is generated per-block — no
`[B, 1, S_q, S_k]` mask tensor is ever materialized.

### Adding a new compression method

1. Subclass `CompressionPolicy` in `utils_real_drop/policies/foo.py`.
2. Implement `prepare_prefill`, `get_query_weights`, `select`.
3. Register it in `utils_real_drop/policies/__init__.py::_REGISTRY`.

Nothing in `attention.py`, `cache.py`, or `kv_llama.py` needs to change.

### Adding a new model (Qwen, Mistral, …)

Create `utils_real_drop/kv_<model>.py` and use the same three calls inside its
attention forward:

```python
key, value = cache.update(key, value, layer_idx)
key   = repeat_kv(key,   num_groups)
value = repeat_kv(value, num_groups)
out, selected = compressed_attention(
    q, key, value,
    policy=cache.get_policy(layer_idx),
    attn_mask=mask, head_dim=head_dim,
)
if selected is not None:
    cache.compress(layer_idx, selected)
```

`cache`, `attention`, and `policies` are model-agnostic and can be reused as-is.
