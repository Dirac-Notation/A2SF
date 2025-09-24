from dataclasses import dataclass, field
from typing import List

@dataclass
class A2SFRLConfig:
    # ----- Model Configuration -----
    model_name: str = "llama3"  # llama, llama2, llama3, opt
    gpus: List[int] = field(default_factory=lambda: [0])
    
    # ----- Context Features -----
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    context_window: int = 64
    max_context: int = 128
    
    # ----- PPO Hyperparameters -----
    ppo_clip: float = 0.2
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.05
    max_grad_norm: float = 1.0
    
    # Training configuration
    episodes_per_update: int = 128
    update_epochs: int = 4
    minibatch_size: int = 64
    
    # ----- Evaluation Configuration -----
    eval_frequency: int = 100
    eval_samples: int = 50
    
    # ----- Misc -----
    seed: int = 42
    device: str = "cuda"
    save_dir: str = "runs/a2sf_rl"
    log_frequency: int = 5
