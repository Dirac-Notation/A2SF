import os
import time
from dataclasses import dataclass
from typing import Any, Dict

import torch
import sys

# Ensure project root is on sys.path for `utils` import when executed from
# outside the repository root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import CompressionConfig, load_model

@dataclass
class ModelResult:
    inference_time: float
    pred_text: str
    output_ids: torch.Tensor


class A2SFModelRunner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        self.model, self.tokenizer = load_model(config.model)
        self.num_layers = self.model.config.num_hidden_layers
        self.debug_shapes = os.environ.get("A2SF_DEBUG_SHAPES", "0") == "1"

    def run_with_compression(
        self,
        prompt: str,
        a: float,
        b: float,
        token_budget: int,
        **kwargs: Any,
    ) -> ModelResult:
        start_time = time.time()

        input_tensor = self.tokenizer(prompt, truncation=False, return_tensors="pt")
        input_ids = input_tensor.input_ids.to(self.model.device)
        attention_mask = input_tensor.attention_mask.to(self.model.device)

        compression_config = self._create_compression_config(a, b, token_budget)
        self.model.init_cache(compression_config)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )[0]

            pred = self.tokenizer.decode(
                output_ids[input_ids.size(-1) :],
                skip_special_tokens=True,
            )
            if self.debug_shapes:
                print("[A2SF_DEBUG] pred_text_len:", len(pred))

        inference_time = time.time() - start_time

        return ModelResult(
            inference_time=inference_time,
            pred_text=pred,
            output_ids=output_ids.detach().cpu(),
        )

    def _create_compression_config(self, a: float, b: float, token_budget: int) -> Dict[str, Any]:
        base_config = CompressionConfig()
        base_config.compression_method = "sigmoid"
        base_config.total_budget = token_budget
        base_config.local_ratios = 0.125
        base_config.a = float(a)
        base_config.b = float(b)
        return base_config


__all__ = ["A2SFModelRunner", "ModelResult"]

