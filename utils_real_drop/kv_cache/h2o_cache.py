import torch

from . import BaseCompressor


class H2OCompressor(BaseCompressor):
    """H2O: 첫 prefill에서 점수 누적 후 선택."""

    def __init__(self, num_key_value_heads: int, device: torch.device):
        super().__init__(num_key_value_heads=num_key_value_heads, device=device)
        self.seq_length = 0
        self.total_budget = 0
        self.recent_budget = 0
        self.select_budget = 0
        self.selected_indices = None
        self.prompt = False

    def init_cache(self, compression_config, layer_idx):
        self.seq_length = 0
        self.selected_indices = None
        self.prompt = False
        self.total_budget = compression_config.total_budget
        self.recent_budget = 16
        self.select_budget = self.total_budget - self.recent_budget

    def should_accumulate_scores(self, seq_len_q: int, seq_len_k: int) -> bool:
        return not self.prompt

    def accumulate_scores(
        self,
        attn_probs: torch.Tensor,
        q_start: int,
        q_end: int,
        acc_scores: torch.Tensor,
    ):
        acc_scores.add_(attn_probs.to(torch.float32).sum(dim=2))

    def select(self, scores: torch.Tensor, seq_len_k: int):
        if seq_len_k <= self.total_budget:
            return None
        scores = scores.clone()
        scores[:, :, -self.recent_budget :] = scores.max()
        selected_indices = scores.topk(self.total_budget, dim=-1).indices
        self.selected_indices = selected_indices
        return selected_indices


# Backward compatibility alias
H2OCache = H2OCompressor