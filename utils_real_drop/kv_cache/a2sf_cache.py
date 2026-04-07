import torch

from . import BaseCompressor


class A2SFCompressor(BaseCompressor):
    """A2SF: forgetting window 기반 누적 + 선택."""

    def __init__(self, num_key_value_heads: int, device: torch.device):
        super().__init__(num_key_value_heads=num_key_value_heads, device=device)
        self.seq_length = 0
        self.total_budget = 0
        self.recent_budget = 0
        self.select_budget = 0
        self.selected_indices = None
        self.prompt = False
        self.forgetting_factor = None
        self.exponents = None
        self.window = None

    def init_cache(self, compression_config, layer_idx):
        self.seq_length = 0
        self.selected_indices = None
        self.prompt = False
        self.total_budget = max(round(float(compression_config.total_budget), 2), 2)
        self.recent_budget = 16
        self.select_budget = self.total_budget - self.recent_budget
        self.forgetting_factor = compression_config.forgetting_factor
        self.exponents = None
        self.window = None

    def should_accumulate_scores(self, seq_len_q: int, seq_len_k: int) -> bool:
        return not self.prompt

    def prepare_prefill(self, seq_len_q: int, seq_len_k: int, device: torch.device, dtype: torch.dtype):
        self.prompt = True
        if self.exponents is None or self.exponents.shape[-1] != seq_len_q:
            self.exponents = torch.arange(0, seq_len_q, device=device).view(1, 1, seq_len_q)
        else:
            self.exponents = self.exponents.to(device)
        self.window = (self.forgetting_factor ** (self.exponents.max() - self.exponents)).to(dtype=torch.float32)

    def get_query_weights(self, q_start, q_end, device, dtype):
        # window shape: [1, 1, seq_len_q] (or [B,1,seq_len_q]); collapse to [qb]
        w = self.window[..., q_start:q_end].reshape(-1, q_end - q_start)[0]
        return w.to(device=device, dtype=torch.float32)

    def select(self, scores: torch.Tensor, seq_len_k: int):
        if seq_len_k <= self.total_budget:
            return None
        scores = scores.clone()
        scores[:, :, -self.recent_budget :] = scores.max()
        selected_indices = scores.topk(self.total_budget, dim=-1).indices.sort().values
        self.selected_indices = selected_indices
        return selected_indices


# Backward compatibility alias
A2SFCache = A2SFCompressor