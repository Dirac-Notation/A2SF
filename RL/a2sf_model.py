from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch

from .agent.neural_ucb_agent import NeuralUCBAgent
from .env import A2SFEnv, A2SFModelRunner
from longbench_eval import dataset2metric


@dataclass
class ModelConfig:
    # ----- Model Configuration -----
    model: str = "llama3-1b"

    # ----- Agent Action Space -----
    a_values: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
    )
    b_values: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0.0], dtype=torch.float32)
    )

    # Note: KVLlama internally uses `device_map="auto"`; we still keep a logical device
    # for tensor placement before we re-align using the first shard device.
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class A2SFGenerateOutput:
    sequences: torch.Tensor
    pred_text: str
    reward: float
    info: Dict[str, Any]


class A2SFModel:
    """
    Wrap RL components (Agent + Env + KV model runner) behind a Transformers-like generate().

    Usage (inference):
      model = A2SFModel(config=..., state_dict=checkpoint["agent_state_dict"])
      out = model.generate(prompt, metric_type="qa_f1_score", token_budget=128, max_new_tokens=64)
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
        arch_config: Optional[Dict[str, Any]] = None,
    ):
        if config is None:
            config = ModelConfig()
        self.config = config

        self.model_runner = A2SFModelRunner(self.config)
        self.env = A2SFEnv(self.model_runner, self.config)

        # Align env/agent tensor device with the actual model shard placement.
        # (KVLlama uses device_map="auto", so config.device("cuda") may not reflect cuda:x.)
        first_layer_device = next(self.model_runner.model.model.layers[0].parameters()).device
        self.env.device = first_layer_device
        self._agent_device = first_layer_device

        if arch_config is not None:
            # 체크포인트의 arch_config를 단일 정보원(source of truth)으로 사용.
            state_dim = int(arch_config["state_dim"])
            num_heads = int(arch_config["num_heads"])
            num_task_types = int(arch_config["num_task_types"])
            metric_heads = list(arch_config["metric_heads"])
            a_values = arch_config["a_values"].to(dtype=torch.float32).clone()
            b_values = arch_config["b_values"].to(dtype=torch.float32).clone()
        else:
            # Legacy path: encoder로부터 차원 유도 + state_dict에서 힌트 추출.
            state_dim = int(self.env.context_encoder.output_dim)
            num_heads = int(self.env.context_encoder.num_heads)
            num_task_types = int(self.env.context_encoder.num_task_types)
            if state_dict is not None:
                head_names = sorted({
                    k.split(".")[1] for k in state_dict.keys()
                    if k.startswith("reward_heads.")
                })
                metric_heads = head_names if head_names else sorted({fn.__name__ for fn in dataset2metric.values()})
            else:
                metric_heads = sorted({fn.__name__ for fn in dataset2metric.values()})
            if state_dict is not None and "a_values" in state_dict and "b_values" in state_dict:
                a_values = state_dict["a_values"].to(dtype=torch.float32).clone()
                b_values = state_dict["b_values"].to(dtype=torch.float32).clone()
            else:
                a_values = self.config.a_values
                b_values = self.config.b_values

        self.agent = NeuralUCBAgent(
            state_dim=state_dim,
            a_values=a_values,
            b_values=b_values,
            metric_heads=metric_heads,
            num_heads=num_heads,
            num_task_types=num_task_types,
        ).to(first_layer_device)

        if state_dict is not None:
            # Final checkpoint은 inverse_lambdas/action_counts 버퍼가 빠져 있을 수 있으므로 strict=False.
            missing, unexpected = self.agent.load_state_dict(state_dict, strict=False)
            if unexpected:
                print(f"[A2SFModel] unexpected keys in state_dict: {unexpected}")
            self.agent.eval()
            # Keep config in sync with what was actually loaded so downstream code sees the right space.
            self.config.a_values = a_values
            self.config.b_values = b_values

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        metric_type: str,
        token_budget: int = 128,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Union[A2SFGenerateOutput, torch.Tensor, str]:
        """
        Transformers-like generate() with RL action selection.

        Required inputs:
          - prompt
          - metric_type
          - token_budget

        Extra generation/runtime options are forwarded via kwargs.
        """
        resolved_metric_type = str(metric_type or "qa_f1_score")
        generation_length = int(kwargs.get("max_new_tokens", 64))
        kwargs.setdefault("num_logits_to_keep", 1)
        answers = kwargs.pop("answers", None)
        all_classes = kwargs.pop("all_classes", None)
        dataset = kwargs.pop("dataset", None)
        task_type = kwargs.pop("task_type", None)

        state = self.env.get_state(
            prompt=prompt,
            metric_type=resolved_metric_type,
            token_budget=token_budget,
            answers=answers,
            all_classes=all_classes,
            generation_length=generation_length,
            dataset=dataset,
            task_type=task_type,
        )

        action, _ = self.agent.act(
            state.to(self._agent_device, dtype=torch.float32),
            metric_type=resolved_metric_type,
        )
        reward_t, info = self.env.run_with_action(action, **kwargs)

        sequences = info["output_ids"]
        pred_text = info["pred"]
        reward = float(reward_t.item()) if isinstance(reward_t, torch.Tensor) else float(reward_t)

        if not return_dict:
            return pred_text

        return A2SFGenerateOutput(
            sequences=sequences,
            pred_text=pred_text,
            reward=reward,
            info=info,
        )


__all__ = ["A2SFModel", "A2SFGenerateOutput", "ModelConfig"]

