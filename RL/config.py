from dataclasses import dataclass, field
from typing import List

@dataclass
class A2SFRLConfig:
    # ----- Model Configuration -----
    model_name: str = "llama2"  # llama, llama2, llama3, opt
    gpus: List[int] = field(default_factory=lambda: [0])
    
    # ----- RL Action Space -----
    # Single continuous action: forgetting factor (0.0 to 1.0)
    # Forgetting factor controls how much past information is retained
    action_min: float = 0.0
    action_max: float = 1.0
    
    # ----- Reward Configuration -----
    accuracy_weight: float = 1.0  # weight for accuracy-based reward
    
    # ----- Context Features -----
    context_window: int = 64  # number of recent tokens to encode
    sentence_transformer_model: str = "all-MiniLM-L6-v2"  # sentence transformer model
    
    # ----- PPO Hyperparameters -----
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    lr: float = 3e-4
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    
    # Training configuration
    episodes_per_update: int = 256
    update_epochs: int = 4
    minibatch_size: int = 128
    
    # ----- Dataset Configuration -----
    tasks: List[str] = field(default_factory=lambda: [
        "Code Complete", "Few Shot", "Single-doc QA", 
        "Multi-doc QA", "Passage Retrieval", "Summarization"
    ])
    max_samples_per_task: int = 100  # limit samples per task for training
    
    # ----- Evaluation Configuration -----
    eval_frequency: int = 10  # evaluate every N iterations
    eval_samples: int = 50  # number of samples for evaluation
    
    # ----- Misc -----
    seed: int = 42
    device: str = "cuda"
    save_dir: str = "runs/a2sf_rl"
    log_frequency: int = 5  # log every N iterations
