"""WAITS RL package — the (task, metric) routing policy (D-I-A-R-L).

  action_grid.py  the 13 paired (a, b) actions
  metadata.py     task/metric vocab + one-hot index helpers (the Input)
  dataset.py      generate the (task, metric, action_scores_gt) training data (D)
  model.py        RoutingNeuralUCB — per-arm LinUCB over [task | metric] (A)
  train.py        bandit training + Sherman-Morrison + waits_table export (R, L)
"""
from .action_grid import SIGMOID_A_VALUES, SIGMOID_B_VALUES, HARD_LIKE_INDICES, NUM_ACTIONS
from .metadata import (
    TASK_TYPE_ORDER, METRIC_TYPE_ORDER,
    task_type_to_index, metric_type_to_index, dataset_metric,
)
from .model import RoutingNeuralUCB

__version__ = "3.0.0"
__all__ = [
    "RoutingNeuralUCB",
    "SIGMOID_A_VALUES", "SIGMOID_B_VALUES", "HARD_LIKE_INDICES", "NUM_ACTIONS",
    "TASK_TYPE_ORDER", "METRIC_TYPE_ORDER",
    "task_type_to_index", "metric_type_to_index", "dataset_metric",
]
