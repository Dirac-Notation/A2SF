from typing import Any, Callable, Dict, List, Optional

from torch.utils.data import Dataset
from tqdm import tqdm


class RLDataset(Dataset):
    """Dataset for RL training episodes (bandit-style single-step)."""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        state_builder: Optional[Callable[[Dict[str, Any], int], Any]] = None,
        token_budgets: Optional[List[int]] = None,
    ):
        self.data = data
        self.state_builder = state_builder
        self.token_budgets = list(token_budgets or [])

        if self.state_builder is not None and len(self.token_budgets) > 0:
            self._precompute_states()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    def _precompute_states(self) -> None:
        total = len(self.data)
        progress = tqdm(
            self.data,
            total=total,
            desc="[dataset] precomputing states",
            unit="sample",
        )
        for sample_idx, sample in enumerate(progress, start=1):
            cached_states: Dict[str, Any] = {}
            for token_budget in self.token_budgets:
                state = self.state_builder(sample, int(token_budget))
                # Keep cache on CPU to avoid occupying model GPU memory.
                cached_states[str(int(token_budget))] = state.detach().cpu()
            sample["cached_states"] = cached_states
            progress.set_postfix_str(f"{sample_idx}/{total}")


def rl_collate_fn(batch):
    """
    Custom collate function for RL dataset.
    Returns the batch as a list of dictionaries (no tensor conversion).
    """
    return batch


__all__ = ["RLDataset", "rl_collate_fn"]

