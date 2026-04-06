import torch

from . import BaseCompressor


class SigmoidCompressor(BaseCompressor):
    """Sigmoid: 시그모이드 창 기반 누적 + 선택."""

    def __init__(self, num_key_value_heads: int, device: torch.device):
        super().__init__(num_key_value_heads=num_key_value_heads, device=device)
        self.seq_length = 0
        self.total_budget = 0
        self.recent_budget = 0
        self.select_budget = 0
        self.selected_indices = None
        self.prompt = False
        self.a = None
        self.b = None
        self.exponents = None
        self.window = None

    @staticmethod
    def _as_1d_tensor(value, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Accept scalar/tensor input and normalize to a 1D tensor."""
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype).reshape(-1)
        return torch.tensor([value], device=device, dtype=dtype)

    def init_cache(self, compression_config, layer_idx):
        self.seq_length = 0
        self.selected_indices = None
        self.prompt = False
        self.total_budget = compression_config.total_budget
        self.recent_budget = 16
        self.select_budget = self.total_budget - self.recent_budget
        self.a = self._as_1d_tensor(compression_config.a, device=self.device, dtype=torch.float32)
        self.b = self._as_1d_tensor(compression_config.b, device=self.device, dtype=torch.float32)
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
        # 기존 구현 수식 보존
        a = self.a.to(device=device, dtype=torch.float32).view(-1, 1, 1)
        b = self.b.to(device=device, dtype=torch.float32).view(-1, 1, 1)
        self.window = 1 / (torch.exp(-a * (self.exponents - (seq_len_q - b - 1.0))))
        self.window = self.window.to(dtype=torch.float32)

    def accumulate_scores(
        self,
        attn_probs: torch.Tensor,
        q_start: int,
        q_end: int,
        acc_scores: torch.Tensor,
    ):
        chunk_len = q_end - q_start
        forgetting = self.window[:, :, q_start:q_end].view(self.window.shape[0], 1, chunk_len, 1)
        weighted = forgetting * attn_probs.to(torch.float32)
        acc_scores.add_(weighted.sum(dim=2))

    def select(self, scores: torch.Tensor, seq_len_k: int):
        if seq_len_k <= self.total_budget:
            return None
        scores = scores.clone()
        scores[:, :, -self.recent_budget :] = scores.max()
        selected_indices = scores.topk(self.total_budget, dim=-1).indices
        self.selected_indices = selected_indices
        return selected_indices


# Backward compatibility alias
SigmoidCache = SigmoidCompressor