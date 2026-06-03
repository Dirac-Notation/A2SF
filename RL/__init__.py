"""A2SF RL package: NeuralUCB agent + sigmoid compression (champion)."""

from .a2sf_model import A2SFModel, ModelConfig
from .agent.neural_ucb_agent import NeuralUCBAgent
from .env import A2SFEnv, AttentionEncoder, A2SFModelRunner

__version__ = "2.0.0"
__all__ = [
    "A2SFModel",
    "ModelConfig",
    "NeuralUCBAgent",
    "A2SFEnv",
    "A2SFModelRunner",
]
