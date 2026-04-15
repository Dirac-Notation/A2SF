from typing import Any, Dict, List, Optional, Tuple

import torch

from .encoder import AttentionEncoder
from longbench_eval import (
    qa_f1_score,
    qa_f1_zh_score,
    rouge_score,
    rouge_zh_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

METRIC_FN_REGISTRY = {
    "qa_f1_score": qa_f1_score,
    "qa_f1_zh_score": qa_f1_zh_score,
    "rouge_score": rouge_score,
    "rouge_zh_score": rouge_zh_score,
    "classification_score": classification_score,
    "retrieval_score": retrieval_score,
    "retrieval_zh_score": retrieval_zh_score,
    "count_score": count_score,
    "code_sim_score": code_sim_score,
}


class A2SFEnv:
    """RL Environment for A2SF model (single-step / bandit)."""

    def __init__(self, runner, config):
        self.runner = runner
        self.config = config
        self.device = torch.device(config.device)

        # Metadata encoder used to build compact RL state features
        self.context_encoder = AttentionEncoder(
            target_model=runner.model,
            target_tokenizer=runner.tokenizer,
            device=config.device,
            output_dim=-1,
            num_query_tokens=16,
        )

        # Current episode cache
        self.current_prompt = None
        self.current_metric_type: str = "qa_f1_score"
        self.current_token_budget = None
        self.current_answers: List[str] = []
        self.current_all_classes: List[str] = []

    def get_state(
        self,
        prompt: str,
        metric_type: str,
        token_budget: int,
        answers: Optional[List[str]] = None,
        all_classes: Optional[List[str]] = None,
        generation_length: int = 64,
        dataset: str = None,
        task_type: Optional[str] = None,
    ) -> torch.Tensor:
        # Cache episode metadata used by `run_with_action()`.
        self.current_prompt = prompt
        self.current_metric_type = str(metric_type or "qa_f1_score")
        self.current_token_budget = token_budget
        self.current_answers = answers or []
        self.current_all_classes = all_classes or []

        return self.context_encoder.encode_context(
            prompt,
            generation_length,
            token_budget,
            metric_type=self.current_metric_type,
            task_type=task_type,
            dataset=dataset,
        ).to(self.device, dtype=torch.float32)

    def run_with_action(
        self,
        action: Tuple[torch.Tensor, torch.Tensor],
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Runs model generation for the given action (a, b), then computes reward in-env.
        """
        a_val, b_val = action
        a_val = float(a_val.item() if isinstance(a_val, torch.Tensor) else a_val)
        b_val = float(b_val.item() if isinstance(b_val, torch.Tensor) else b_val)

        with torch.no_grad():
            result = self.runner.run_with_compression(
                prompt=self.current_prompt,
                a=a_val,
                b=b_val,
                token_budget=self.current_token_budget,
                **kwargs,
            )

        pred_text = result.pred_text
        metric_fn = METRIC_FN_REGISTRY.get(self.current_metric_type, qa_f1_score)

        reward_val = 0.0
        if self.current_answers:
            for gt in self.current_answers:
                reward_val = max(
                    reward_val,
                    float(metric_fn(pred_text, gt, all_classes=self.current_all_classes)),
                )

        reward = torch.tensor(reward_val, device=self.device)

        info = {
            "a": a_val,
            "b": b_val,
            "reward": reward_val,
            "pred": pred_text,
            "output_ids": result.output_ids.detach().cpu(),
        }

        return reward, info

    def run_with_actions(
        self,
        action: Tuple[torch.Tensor, torch.Tensor],
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Runs model generation for batched actions over the same prompt."""
        a_vals, b_vals = action
        if not isinstance(a_vals, torch.Tensor):
            a_vals = torch.tensor([a_vals], device=self.device, dtype=torch.float32)
        if not isinstance(b_vals, torch.Tensor):
            b_vals = torch.tensor([b_vals], device=self.device, dtype=torch.float32)

        a_vals = a_vals.to(self.device, dtype=torch.float32).view(-1)
        b_vals = b_vals.to(self.device, dtype=torch.float32).view(-1)
        if a_vals.numel() != b_vals.numel():
            raise ValueError(
                f"Action tensor sizes must match, got a:{a_vals.numel()} and b:{b_vals.numel()}"
            )

        batch_size = int(a_vals.numel())
        prompts = [self.current_prompt] * batch_size

        with torch.no_grad():
            result = self.runner.run_with_compression(
                prompt=prompts,
                a=a_vals,
                b=b_vals,
                token_budget=self.current_token_budget,
                **kwargs,
            )

        pred_texts = result.pred_text
        if isinstance(pred_texts, str):
            pred_texts = [pred_texts]
        if len(pred_texts) != batch_size:
            raise ValueError(
                f"Batched prediction size mismatch, expected {batch_size}, got {len(pred_texts)}"
            )

        metric_fn = METRIC_FN_REGISTRY.get(self.current_metric_type, qa_f1_score)
        rewards: List[float] = []
        for pred_text in pred_texts:
            reward_val = 0.0
            if self.current_answers:
                for gt in self.current_answers:
                    reward_val = max(
                        reward_val,
                        float(metric_fn(pred_text, gt, all_classes=self.current_all_classes)),
                    )
            rewards.append(reward_val)

        reward_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        info = {
            "a": a_vals.detach().cpu(),
            "b": b_vals.detach().cpu(),
            "reward": rewards,
            "pred": pred_texts,
            "output_ids": result.output_ids.detach().cpu(),
        }
        return reward_tensor, info


__all__ = ["A2SFEnv"]

