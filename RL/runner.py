import torch
import os
from typing import Dict, Any
from dataclasses import dataclass
import time

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_model, CompressionConfig
from .main import A2SFRLConfig

@dataclass
class ModelResult:
    reward: float
    inference_time: float

class A2SFModelRunner:
    def __init__(self, config: A2SFRLConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.model, self.tokenizer = load_model(config.model_name)
        
        self.num_layers = self.model.config.num_hidden_layers
        self.debug_shapes = os.environ.get("A2SF_DEBUG_SHAPES", "0") == "1"
    
    def run_with_compression(
        self,
        prompt: str,
        a: float,
        b: float,
        token_budget: int,
        target_prob_data: Dict[str, torch.Tensor],
        dataset: str = None,
    ) -> ModelResult:
        start_time = time.time()
        
        input_tensor = self.tokenizer(prompt, truncation=False, return_tensors="pt")
        input_ids = input_tensor.input_ids.to(self.model.device)
        attention_mask = input_tensor.attention_mask.to(self.model.device)
        
        compression_config = self._create_compression_config(a, b, token_budget)
        self.model.init_cache(compression_config)

        answer_token_ids = target_prob_data["answer_token_ids"].to(self.model.device)
        teacher_topk_indices = target_prob_data["teacher_topk_indices"].to(self.model.device)
        teacher_topk_probs = target_prob_data["teacher_topk_probs"].to(self.model.device)
        
        with torch.no_grad():
            prefill_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_attentions=False,
            )
            prefill_logits = prefill_outputs.logits[:, -1, :]  # (1, vocab)
            past_key_values = prefill_outputs.past_key_values

            answer_len = int(answer_token_ids.size(0))
            if answer_len == 0:
                kl_value = 0.0
            else:
                # 첫 번째 정답 토큰 확률은 prefill 마지막 logits에서 계산
                logits_steps = [prefill_logits.unsqueeze(1)]  # (1, 1, vocab)

                # 나머지 정답 토큰은 teacher forcing으로 한 번에 계산
                if answer_len > 1:
                    decode_input_ids = answer_token_ids[:-1].unsqueeze(0)
                    decode_attention_mask = torch.cat(
                        [
                            attention_mask,
                            attention_mask.new_ones((attention_mask.size(0), answer_len - 1)),
                        ],
                        dim=-1,
                    )
                    decode_outputs = self.model(
                        input_ids=decode_input_ids,
                        attention_mask=decode_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_attentions=False,
                    )
                    logits_steps.append(decode_outputs.logits)  # (1, answer_len-1, vocab)

                if self.debug_shapes:
                    print("[A2SF_DEBUG] answer_len:", answer_len)
                    print("[A2SF_DEBUG] prefill_logits.shape:", tuple(prefill_logits.shape))
                    if answer_len > 1:
                        print("[A2SF_DEBUG] decode_input_ids.shape:", tuple(decode_input_ids.shape))
                        print("[A2SF_DEBUG] decode_outputs.logits.shape:", tuple(decode_outputs.logits.shape))
                    for idx, t in enumerate(logits_steps):
                        print(f"[A2SF_DEBUG] logits_steps[{idx}].shape:", tuple(t.shape))

                student_logits = torch.cat(logits_steps, dim=1)  # (1, answer_len, vocab)
                kl_value = self._compute_sparse_kl_divergence(
                    student_logits=student_logits,
                    teacher_topk_indices=teacher_topk_indices,
                    teacher_topk_probs=teacher_topk_probs,
                )
        
        inference_time = time.time() - start_time
        reward = -kl_value
        
        return ModelResult(reward=reward, inference_time=inference_time)

    def _compute_sparse_kl_divergence(
        self,
        student_logits: torch.Tensor,
        teacher_topk_indices: torch.Tensor,
        teacher_topk_probs: torch.Tensor,
    ) -> float:
        """
        Compute KL(P_teacher || P_student) on teacher top-k support only.
        """
        if teacher_topk_indices.numel() == 0 or teacher_topk_probs.numel() == 0:
            return 0.0

        # Align by shortest available sequence length for stability
        seq_len = min(student_logits.size(1), teacher_topk_indices.size(0), teacher_topk_probs.size(0))
        if seq_len == 0:
            return 0.0

        student_log_probs = torch.log_softmax(student_logits[:, :seq_len, :], dim=-1)
        support_idx = teacher_topk_indices[:seq_len].unsqueeze(0).long()
        support_teacher_probs = teacher_topk_probs[:seq_len].to(student_log_probs.dtype)

        # Normalize sparse teacher probs so KL is well-defined on support
        support_teacher_probs = support_teacher_probs / support_teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        support_teacher_log_probs = torch.log(support_teacher_probs.clamp_min(1e-12))

        support_student_log_probs = torch.gather(student_log_probs, dim=-1, index=support_idx).squeeze(0)
        token_kl = torch.sum(
            support_teacher_probs * (support_teacher_log_probs - support_student_log_probs),
            dim=-1,
        )
        return float(token_kl.mean().item())
    
    def _create_compression_config(self, a: float, b: float, token_budget: int) -> Dict[str, Any]:
        base_config = CompressionConfig()
        
        base_config.compression_method = "sigmoid"
        base_config.total_budget = token_budget
        base_config.layerwise_ratios = [1.0 for _ in range(self.num_layers)]
        base_config.local_ratios = 0.125
        base_config.a = float(a)
        base_config.b = float(b)
        
        return base_config
