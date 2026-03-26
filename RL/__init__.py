"""
A2SF RL Package

This package contains the reinforcement learning components for training
an A2SF model to dynamically adjust KV cache compression ratios.
"""

from .agent.neural_ucb_agent import NeuralUCBAgent
from .env import A2SFEnv, AttentionEncoder
from .env import A2SFModelRunner
from .training.trainer import A2SFTrainer
from .a2sf_model import ModelConfig

__version__ = "1.0.0"
__all__ = [
    "NeuralUCBAgent",
    "A2SFEnv",
    "A2SFModelRunner",
    "A2SFTrainer",
    "AttentionEncoder",
    "ModelConfig",
]
