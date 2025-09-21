from dataclasses import dataclass, field
from typing import List

@dataclass
class A2SFRLConfig:
    model_name: str = "llama2"
    gpus: List[int] = field(default_factory=lambda: [0])
    
    action_min: float = 0.0
    action_max: float = 1.0
    
    accuracy_weight: float = 1.0
    
    context_window: int = 64
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.05
    max_grad_norm: float = 1.0
    
    episodes_per_update: int = 256
    update_epochs: int = 4
    minibatch_size: int = 128
    
    def __post_init__(self):
        if self.episodes_per_update < self.minibatch_size:
            self.minibatch_size = self.episodes_per_update
            print(f"Adjusted minibatch_size to {self.minibatch_size} to match episodes_per_update")
    
    tasks: List[str] = field(default_factory=lambda: [
        "Code Complete", "Few Shot", "Single-doc QA", 
        "Multi-doc QA", "Passage Retrieval", "Summarization"
    ])
    max_samples_per_task: int = 100
    
    eval_frequency: int = 10
    eval_samples: int = 50
    
    seed: int = 42
    device: str = "cuda"
    save_dir: str = "runs/a2sf_rl"
    log_frequency: int = 5