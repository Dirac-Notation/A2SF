# A2SF (Accumulative Attention Score with Forgetting)

## Overview

A key-value (KV) cache compression technique utilizing accumulative attention scores. 

---

## Preparation

### Python Version

- Python: 3.8 (Another versions are under the test.)

### To set up the environment, follow the below:

   ```bash
   conda env create -n A2SF python=3.8
   conda activate A2SF
   pip install -r pip.txt
   ```

## Example

   ```python
   import torch

   from transformers import AutoTokenizer

   from utils import load_datasets
   from utils_real_drop.kv_llama import LlamaForCausalLM

   model_name = "meta-llama/Llama-2-7b-hf"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = LlamaForCausalLM.from_pretrained(model_name).half().to("cuda")

   model.init_cache(
      use_compression=True,
      select_budget=128, # Cache selection budget
      recent_budget=128, # Recent cache budget
   )

   prompts, answers = load_datasets(dataset_path="datasets/cnn_dailymail-3shot.jsonl", tokenizer=tokenizer)

   input_ids = prompts[0].to(model.device)

   print(tokenizer.decode(model.generate(input_ids, max_new_tokens=64).flatten()[input_ids.numel():].tolist()))
   ```