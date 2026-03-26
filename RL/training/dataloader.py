from typing import Any, Dict, List

from torch.utils.data import Dataset


class RLDataset(Dataset):
    """Dataset for RL training episodes (bandit-style single-step)."""

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def rl_collate_fn(batch):
    """
    Custom collate function for RL dataset.
    Returns the batch as a list of dictionaries (no tensor conversion).
    """
    return batch


__all__ = ["RLDataset", "rl_collate_fn"]

