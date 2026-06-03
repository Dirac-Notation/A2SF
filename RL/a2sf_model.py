from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch

from .agent.neural_ucb_agent import NeuralUCBAgent
from .env import A2SFEnv, A2SFModelRunner


# Sigmoid action space: flat pair lists (paired actions, NOT cartesian).
# a=0 makes b irrelevant (σ(0)=0.5 uniform), so only one (0, 1) entry kept.
# 13 unique (a, b) pairs (matches champion ckpt simple_ucb_v5_maxo, LB128=26.95):
#   (0, 1) + cartesian({0.01, 0.1, 10} × {1, 16, 32, 128}).
_A_BASE = [0.01, 0.1, 10.0]
_B_BASE = [1.0, 16.0, 32.0, 128.0]
SIGMOID_A_VALUES = [0.0]
SIGMOID_B_VALUES = [1.0]
for _a in _A_BASE:
    for _b in _B_BASE:
        SIGMOID_A_VALUES.append(_a)
        SIGMOID_B_VALUES.append(_b)
assert len(SIGMOID_A_VALUES) == 13

# Indices of "hard-like + a=0" actions, used by ablation Config A/B.
HARD_LIKE_INDICES = [0, 9, 10, 11, 12]  # (0,1), (10,1), (10,16), (10,32), (10,128)


@dataclass
class ModelConfig:
    model: str = "llama3-1b"

    # Compression method: "sigmoid" (champion), "a2sf" / "snap" (baselines), "full" (no compression).
    compression_method: str = "sigmoid"

    # Encoder (frozen mini-attn).
    encoder_topk: int = 16            # top-K positions per head in stats feature mode
    encoder_views: str = "both"       # tova | snap | both
    mini_attn_ckpt: str = ""          # path to MiniAttnEncoder checkpoint
    encoder_max_input_length: int = 32768
    encoder_feature_mode: str = "stats"
    # NeuralUCBAgent expects 2 views (tova + snap concatenated). MiniAttnEncoder
    # with single_view=False duplicates its single mini-attn output into 2-view
    # layout. Keep False to stay consistent with the agent's num_views=2.
    single_view: bool = False
    # Champion always-on (history file 9): adds hidden_size content channel to state.
    encoder_include_hidden_pool: bool = True
    encoder_hidden_pool_window: int = 0
    include_metric_oh: bool = True
    include_seq_len: bool = True
    include_task_oh: bool = True
    # Extra view appended after mini_attn_stats (experiments A-D).
    extra_view: str = 'none'         # none|pre_rope_mean|pre_rope_max|pre_rope_z_max|snap_l0|position_prior
    pre_rope_query_window: int = 16  # query window for pre_rope_* and snap_l0

    # Sigmoid 34-action paired space.
    a_values: torch.Tensor = field(
        default_factory=lambda: torch.tensor(SIGMOID_A_VALUES, dtype=torch.float32)
    )
    b_values: torch.Tensor = field(
        default_factory=lambda: torch.tensor(SIGMOID_B_VALUES, dtype=torch.float32)
    )
    paired_actions: bool = True

    # Orthogonal compression modifiers (apply on top of any base method).
    chunk_size: int = 0           # ChunkKV: 0 = off, >0 = group head tokens into chunks of this size.
    pyramid_kv: bool = False      # PyramidKV: vary per-layer budget (layer 0 max, layer L-1 min).
    pyramid_ratio: float = 4.0    # PyramidKV bottom/top ratio.

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def sigmoid(cls, model: str = "llama3-1b") -> "ModelConfig":
        return cls(model=model)


@dataclass
class A2SFGenerateOutput:
    sequences: torch.Tensor
    pred_text: str
    reward: float
    info: Dict[str, Any]


class A2SFModel:
    """Wrap RL components (NeuralUCBAgent + Env + KV model runner) behind a Transformers-like generate().

    Usage (inference):
      model = A2SFModel(config=..., state_dict=ckpt["agent_state_dict"], arch_config=ckpt["arch_config"])
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

        first_layer_device = next(self.model_runner.model.model.layers[0].parameters()).device
        self.env.device = first_layer_device
        self._agent_device = first_layer_device

        if arch_config is None:
            raise ValueError("A2SFModel requires arch_config from the checkpoint.")

        a_values = arch_config["a_values"].to(dtype=torch.float32).clone()
        b_values = arch_config["b_values"].to(dtype=torch.float32).clone()

        variant = str(arch_config.get("variant", "")).lower()
        if variant == "simple_ucb":
            from .agent import SimpleUCBAgent
            self.agent = SimpleUCBAgent(
                state_dim=int(arch_config["state_dim"]),
                a_values=a_values,
                b_values=b_values,
                num_metric_types=int(arch_config["num_metric_types"]),
                num_task_types=int(arch_config["num_task_types"]),
                hidden=int(arch_config.get("hidden", 256)),
                dropout=float(arch_config.get("dropout", 0.0)),
                paired_actions=bool(arch_config.get("paired_actions", True)),
                include_seq_len=bool(arch_config.get("include_seq_len", True)),
            ).to(first_layer_device)
        elif variant == "lora_ucb":
            from .agent import LoRAUCBAgent
            self.agent = LoRAUCBAgent(
                state_dim=int(arch_config["state_dim"]),
                a_values=a_values,
                b_values=b_values,
                num_metric_types=int(arch_config["num_metric_types"]),
                num_task_types=int(arch_config["num_task_types"]),
                metric_heads=list(arch_config.get("metric_heads", []) or []),
                hidden=int(arch_config.get("hidden", 256)),
                block_hidden=int(arch_config.get("block_hidden", 512)),
                lora_rank=int(arch_config.get("lora_rank", 4)),
                dropout=float(arch_config.get("dropout", 0.0)),
                paired_actions=bool(arch_config.get("paired_actions", True)),
                include_seq_len=bool(arch_config.get("include_seq_len", True)),
                backbone_route_by=str(arch_config.get("backbone_route_by", "task")),
                head_route_by=str(arch_config.get("head_route_by", "metric")),
            ).to(first_layer_device)
        else:
            # Infer paired_actions from ckpt: if num_actions in inverse_lambdas == len(a_values), it's paired.
            paired_inferred = True
            if state_dict is not None and "inverse_lambdas" in state_dict:
                inv_shape = state_dict["inverse_lambdas"].shape
                # Shape: (metric_heads, num_actions, feat, feat) for new NeuralUCB
                ckpt_num_actions = int(inv_shape[1]) if len(inv_shape) >= 2 else int(inv_shape[0])
                if ckpt_num_actions == len(a_values) * len(b_values):
                    paired_inferred = False
                elif ckpt_num_actions == len(a_values):
                    paired_inferred = True
            paired = bool(arch_config.get("paired_actions", paired_inferred))
            self.agent = NeuralUCBAgent(
                state_dim=int(arch_config["state_dim"]),
                a_values=a_values,
                b_values=b_values,
                num_metric_types=int(arch_config["num_metric_types"]),
                num_task_types=int(arch_config["num_task_types"]),
                side_dim=int(arch_config["side_dim"]),
                num_heads=int(arch_config["num_heads"]),
                num_hidden_pool=int(arch_config.get("num_hidden_pool", 0)),
                backbone_depth=int(arch_config.get("backbone_depth", 2)),
                dropout=float(arch_config.get("dropout", 0.0)),
                paired_actions=paired,
                task_cond_head=bool(arch_config.get("task_cond_head", True)),
                task_head_mlp=bool(arch_config.get("task_head_mlp", False)),
                output_activation=str(arch_config.get("output_activation", "sigmoid")),
                num_views=int(arch_config.get("num_views", 2)),
                include_seq_len=bool(arch_config.get("include_seq_len", True)),
            ).to(first_layer_device)

        if state_dict is not None:
            missing, unexpected = self.agent.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[A2SFModel] missing keys: {missing}")
            if unexpected:
                print(f"[A2SFModel] unexpected keys: {unexpected}")
            self.agent.eval()
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
        resolved_metric_type = str(metric_type or "qa_f1_score")
        generation_length = int(kwargs.get("max_new_tokens", 64))
        kwargs.setdefault("num_logits_to_keep", 1)
        answers = kwargs.pop("answers", None)
        all_classes = kwargs.pop("all_classes", None)
        dataset = kwargs.pop("dataset", None)
        task_type = kwargs.pop("task_type", None)

        fixed_action_idx = kwargs.pop("fixed_action_idx", None)

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

        if fixed_action_idx is not None:
            # Bypass RL agent: pick a fixed (a, b) from the agent's grid by index.
            i = int(fixed_action_idx)
            a_val = self.agent.a_values[i:i + 1].to(self._agent_device, dtype=torch.float32)
            b_val = self.agent.b_values[i:i + 1].to(self._agent_device, dtype=torch.float32)
            action = (a_val, b_val)
        else:
            action, _ = self.agent.act(
                state.to(self._agent_device, dtype=torch.float32),
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
