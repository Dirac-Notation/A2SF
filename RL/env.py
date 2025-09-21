import torch
from typing import Dict, Any, Tuple
from dataclasses import dataclass

from .features import ContextEncoder, build_state_from_context
from .config import A2SFRLConfig

@dataclass
class EpisodeResult:
    accuracy_score: float
    forgetting_factor: float
    total_reward: float
    metrics: Dict[str, Any]

class A2SFEnv:
    def __init__(self, runner, config: A2SFRLConfig):
        self.runner = runner
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        self.context_encoder = ContextEncoder(
            model_name=config.sentence_transformer_model,
            device=config.device
        )
        
        self.current_prompt = None
        self.current_task = None
        self.current_dataset = None
        self.current_tokens = None
        self.current_answers = None
        self._last_state = None
    
    def _encode_to_state(self, tokens) -> torch.Tensor:
        context_embedding = self.context_encoder.encode_context(
            tokens, max_tokens=self.config.context_window
        )
        state = build_state_from_context(context_embedding)

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device, dtype=torch.float32)
        return state
    
    def reset(self, prompt: str, task: str, tokens: list, answers: list, dataset: str = None) -> torch.Tensor:
        self.current_prompt = prompt
        self.current_task = task
        self.current_dataset = dataset
        self.current_tokens = tokens
        self.current_answers = answers
        
        state = self._encode_to_state(tokens)
        self._last_state = state
        return state
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool, Dict[str, Any]]:
        if isinstance(action, torch.Tensor):
            action = action.to(self.device).float().view(-1)
            a_scalar = action[0].item()
        else:
            a_scalar = float(action)

        forgetting_factor = float(
            max(self.config.action_min, min(self.config.action_max, a_scalar))
        )

        with torch.no_grad():
            result = self.runner.run_with_compression(
                prompt=self.current_prompt,
                task=self.current_task,
                forgetting_factor=forgetting_factor,
                answers=self.current_answers,
                dataset=self.current_dataset
            )
        
        reward = float(result.accuracy_score)
        reward_tensor = torch.tensor(reward, device=self.device, dtype=torch.float32)
        
        next_state = self._last_state if self._last_state is not None else self._encode_to_state(self.current_tokens)
        done = True
        
        info = {
            "accuracy_score": result.accuracy_score,
            "forgetting_factor": forgetting_factor,
            "metrics": result.metrics
        }
        
        return next_state, reward_tensor, done, info