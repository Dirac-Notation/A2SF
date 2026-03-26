import os
import sys
import json
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from ..a2sf_model import ModelConfig
from .training_config import TrainingConfig
from ..agent.neural_ucb_agent import NeuralUCBAgent
from ..env import A2SFEnv, A2SFModelRunner
from .dataloader import RLDataset, rl_collate_fn
from longbench_eval import dataset2metric

TOKEN_BUDGET_CANDIDATES = [128]


class A2SFTrainer:
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.device = None  # will be aligned to the actual model shard device

        torch.manual_seed(training_config.seed)
        random.seed(training_config.seed)

        self.model_runner = A2SFModelRunner(self.model_config)
        self.env = A2SFEnv(self.model_runner, self.model_config)

        # KVLlama uses device_map="auto", so config.device may not equal cuda:x.
        first_layer_device = next(self.model_runner.model.model.layers[0].parameters()).device
        self.device = first_layer_device
        self.env.device = first_layer_device

        # State dimension: [seq_length] + [head entropy, head max_pos] per attention head
        state_dim = int(self.env.context_encoder.output_dim)
        metric_heads = sorted({fn.__name__ for fn in dataset2metric.values()})

        self.agent = NeuralUCBAgent(
            state_dim=state_dim,
            a_values=self.model_config.a_values,
            b_values=self.model_config.b_values,
            metric_heads=metric_heads,
        ).to(self.device)

        # Optimizer only includes agent parameters
        # Context encoder is metadata-based and frozen
        all_params = list(self.agent.parameters())
        self.optimizer = optim.SGD(all_params, lr=self.training_config.lr)

        # Load pre-split data generated under `RL/training/data/`.
        training_data_list, eval_data_list = self.load_training_data()

        # Create datasets
        self.training_dataset = RLDataset(training_data_list)
        self.eval_dataset = RLDataset(eval_data_list)

        # Create data loaders
        # For training: shuffle and use episodes_per_update as batch size
        self.train_loader = DataLoader(
            self.training_dataset,
            batch_size=self.training_config.episodes_per_update,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues with model
            pin_memory=False,
            collate_fn=rl_collate_fn,  # Use custom collate function
        )

        # For evaluation: no shuffle, process all at once
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=1,  # Process one at a time for evaluation
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=rl_collate_fn,  # Use custom collate function
        )

        print(f"Loaded {len(self.training_dataset)} training samples")
        print(f"Loaded {len(self.eval_dataset)} evaluation samples")
        print(f"Training batches per epoch: {len(self.train_loader)}")
        os.makedirs(self.training_config.save_dir, exist_ok=True)

        self.iterations_per_epoch = len(self.train_loader)
        self.total_epochs = max(1, int(self.training_config.epochs))
        self.total_iterations = max(1, self.total_epochs * max(1, self.iterations_per_epoch))
        self.scheduler_t_max = self.total_epochs

        # Initialize learning rate scheduler after total_iterations is known
        self.scheduler = self._create_scheduler()

    @staticmethod
    def _format_fixed(value: float, digits: int) -> str:
        return f"{float(value):.{digits}f}"

    def _format_list_fixed(self, values: List[Any], digits: int) -> List[Any]:
        formatted = []
        for v in values:
            if isinstance(v, (float, int)):
                formatted.append(self._format_fixed(v, digits))
            else:
                formatted.append(v)
        return formatted

    def _serialize_with_fixed_precision(
        self, payload: Dict[str, Any], digits_map: Dict[str, int]
    ) -> Dict[str, Any]:
        serialized = {}
        for key, value in payload.items():
            if key in digits_map and isinstance(value, (float, int)):
                serialized[key] = self._format_fixed(value, digits_map[key])
            elif key in digits_map and isinstance(value, list):
                serialized[key] = self._format_list_fixed(value, digits_map[key])
            else:
                serialized[key] = value
        return serialized

    def _format_reward_pairs(
        self, pairs: List[Tuple[float, float]], digits: int = 4
    ) -> List[List[str]]:
        return [
            [self._format_fixed(gt_reward, digits), self._format_fixed(pred_reward, digits)]
            for gt_reward, pred_reward in pairs
        ]

    def load_training_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        train_path = self.training_config.train_data_path
        eval_path = self.training_config.eval_data_path

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training split not found: {train_path}")
        if not os.path.exists(eval_path):
            raise FileNotFoundError(
                f"Evaluation split not found: {eval_path}. "
                "Run `python -m RL.training.data_generation.make_training_dataset` "
                "to generate fixed split files."
            )

        with open(train_path, "r", encoding="utf-8") as f:
            training_data = [json.loads(line) for line in f if line.strip()]
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = [json.loads(line) for line in f if line.strip()]

        for sample in training_data + eval_data:
            sample["metric_type"] = self._resolve_metric_type(sample.get("dataset"))
            if "answers" not in sample:
                sample["answers"] = []
            if "all_classes" not in sample:
                sample["all_classes"] = []

        print(f"Loaded {len(training_data)} training samples from {train_path}")
        print(f"Loaded {len(eval_data)} evaluation samples from {eval_path}")
        return training_data, eval_data

    @staticmethod
    def _resolve_metric_type(dataset_name: str) -> str:
        dataset_key = str(dataset_name or "").strip().lower()
        fn = dataset2metric.get(dataset_key)
        if fn is None:
            return "qa_f1_score"
        return fn.__name__

    def _neural_ucb_update_batch(
        self,
        samples: List[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, str]],
        config,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, Any]:
        if len(samples) == 0:
            return {
                "prediction_loss": 0.0,
                "l2_loss": 0.0,
                "total_loss": 0.0,
                "grad_norm": 0.0,
                "reward_pairs": [],
            }

        self.agent.train()
        self.env.context_encoder.eval()

        prediction_mse_losses = []
        reward_pairs: List[Tuple[float, float]] = []
        feature_action_pairs = []

        for state, action, reward, metric_type in samples:
            if state.ndim == 1:
                state = state.unsqueeze(0)

            out = self.agent.forward(state, metric_type=metric_type)
            reward_pred = out["reward_pred"]
            feature_vector = out["feature_vector"]

            action_idx = self.agent._get_action_indices(action)
            if action_idx.ndim > 0:
                action_idx = action_idx[0]

            selected_predict = reward_pred[0, action_idx].unsqueeze(0)
            actual_reward = reward.unsqueeze(0) if reward.ndim == 0 else reward
            prediction_mse_losses.append(F.mse_loss(selected_predict, actual_reward))
            reward_pairs.append(
                (
                    float(actual_reward.view(-1)[0].detach().item()),
                    float(selected_predict.view(-1)[0].detach().item()),
                )
            )
            feature_action_pairs.append((feature_vector.detach(), action_idx))

        mean_mse = torch.stack(prediction_mse_losses).mean()
        prediction_loss = torch.sqrt(mean_mse + 1e-8)

        l2_loss = 0.0
        for param in self.agent.parameters():
            if param.requires_grad:
                l2_loss += torch.sum(param ** 2)
        l2_loss = config.l2_coef * l2_loss

        total_loss = prediction_loss + l2_loss

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm_sq = 0.0
        for param in self.agent.parameters():
            if param.grad is not None:
                g = param.grad.detach()
                grad_norm_sq += float(torch.sum(g * g).item())
        grad_norm = grad_norm_sq ** 0.5
        optimizer.step()

        with torch.no_grad():
            for feature_vector, action_idx in feature_action_pairs:
                self.agent._update_inverse_covariances(feature_vector, action_idx)

        return {
            "prediction_loss": float(prediction_loss.item()),
            "l2_loss": float(l2_loss.item()),
            "total_loss": float(total_loss.item()),
            "grad_norm": float(grad_norm),
            "reward_pairs": reward_pairs,
        }

    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        return lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.scheduler_t_max,
            eta_min=self.training_config.lr * 0.01,
        )

    def _get_ucb_beta(self, iteration: int, total_iterations: int) -> float:
        if total_iterations <= 1:
            return float(self.training_config.ucb_beta_max)

        progress = iteration / float(total_iterations - 1)
        beta = self.training_config.ucb_beta_max + (
            self.training_config.ucb_beta_min - self.training_config.ucb_beta_max
        ) * progress
        return float(beta)

    def train(self, num_epochs: int = 1) -> int:
        print(f"Starting training for {num_epochs} epochs")

        global_iteration = 0
        total_epochs = max(1, int(num_epochs))

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            current_beta = self._get_ucb_beta(epoch, total_epochs)

            for batch in self.train_loader:
                best_rewards: List[float] = []
                worst_rewards: List[float] = []
                best_actions_a: List = []
                best_actions_b: List = []
                worst_actions_a: List = []
                worst_actions_b: List = []
                best_input_seq_lengths: List[int] = []
                best_task_types: List[str] = []
                worst_input_seq_lengths: List[int] = []
                worst_task_types: List[str] = []
                update_samples = []

                start_time = time.time()

                for episode_data in batch:
                    prompt = episode_data["input_prompt"]
                    tokenized_prompt = self.model_runner.tokenizer(
                        prompt, truncation=False, return_tensors="pt"
                    )
                    input_seq_len = int(tokenized_prompt.input_ids.size(1))
                    task_type_str = str(episode_data.get("task_type") or "unknown")
                    metric_type = str(episode_data.get("metric_type") or "qa_f1_score")

                    for token_budget in TOKEN_BUDGET_CANDIDATES:
                        state = self.env.get_state(
                            prompt=prompt,
                            generation_length=episode_data["generation_length"],
                            answers=episode_data.get("answers", []),
                            all_classes=episode_data.get("all_classes", []),
                            metric_type=metric_type,
                            token_budget=token_budget,
                            dataset=episode_data["dataset"],
                            task_type=episode_data.get("task_type"),
                        )
                        state = state.to(self.device)

                        best_action, _ = self.agent.get_best_action(
                            state,
                            beta=current_beta,
                            metric_type=metric_type,
                        )
                        worst_action, _ = self.agent.get_worst_action(
                            state,
                            beta=current_beta,
                            metric_type=metric_type,
                        )

                        best_reward, _best_info = self.env.run_with_action(best_action)
                        worst_reward, _worst_info = self.env.run_with_action(worst_action)

                        best_gt_reward_val = (
                            best_reward.item() if isinstance(best_reward, torch.Tensor) else float(best_reward)
                        )
                        worst_gt_reward_val = (
                            worst_reward.item() if isinstance(worst_reward, torch.Tensor) else float(worst_reward)
                        )

                        update_samples.append((state, best_action, best_reward, metric_type))
                        update_samples.append((state, worst_action, worst_reward, metric_type))

                        best_rewards.append(best_gt_reward_val)
                        worst_rewards.append(worst_gt_reward_val)

                        best_a_val, best_b_val = best_action
                        worst_a_val, worst_b_val = worst_action

                        best_actions_a.append(
                            round(best_a_val.item(), 2) if isinstance(best_a_val, torch.Tensor) else best_a_val
                        )
                        best_actions_b.append(
                            int(best_b_val.item()) if isinstance(best_b_val, torch.Tensor) else int(best_b_val)
                        )
                        worst_actions_a.append(
                            round(worst_a_val.item(), 2) if isinstance(worst_a_val, torch.Tensor) else worst_a_val
                        )
                        worst_actions_b.append(
                            int(worst_b_val.item()) if isinstance(worst_b_val, torch.Tensor) else int(worst_b_val)
                        )
                        best_input_seq_lengths.append(input_seq_len)
                        worst_input_seq_lengths.append(input_seq_len)
                        best_task_types.append(task_type_str)
                        worst_task_types.append(task_type_str)

                end_time = time.time()
                print(f"Time taken for one iteration: {end_time - start_time} seconds")

                avg_loss_stats = self._neural_ucb_update_batch(
                    samples=update_samples,
                    config=self.training_config,
                    optimizer=self.optimizer,
                )
                all_reward_pairs = avg_loss_stats.get("reward_pairs", [])
                best_reward_pairs = all_reward_pairs[0::2]
                worst_reward_pairs = all_reward_pairs[1::2]

                self._log_progress(
                    iteration=global_iteration,
                    loss_stats=avg_loss_stats,
                    iteration_rewards=best_rewards,
                    iteration_actions_a=best_actions_a,
                    iteration_actions_b=best_actions_b,
                    iteration_reward_pairs=best_reward_pairs,
                    iteration_input_seq_lengths=best_input_seq_lengths,
                    iteration_task_types=best_task_types,
                    current_beta=current_beta,
                    log_filename="training_progress_best.jsonl",
                    log_prefix="BEST",
                )
                self._log_progress(
                    iteration=global_iteration,
                    loss_stats=avg_loss_stats,
                    iteration_rewards=worst_rewards,
                    iteration_actions_a=worst_actions_a,
                    iteration_actions_b=worst_actions_b,
                    iteration_reward_pairs=worst_reward_pairs,
                    iteration_input_seq_lengths=worst_input_seq_lengths,
                    iteration_task_types=worst_task_types,
                    current_beta=current_beta,
                    log_filename="training_progress_worst.jsonl",
                    log_prefix="WORST",
                )
                global_iteration += 1

            self.scheduler.step()

            last_iter = max(0, global_iteration - 1)
            self._save_checkpoint(last_iter)
            self._evaluate(last_iter, current_beta=current_beta, total_iterations=total_epochs)
            self._plot_training_progress()

        return max(0, global_iteration - 1)

    def _plot_training_progress(self) -> None:
        try:
            from .plot_training_pregress import plot_training_progress
        except Exception as e:
            print(f"[plot] failed to import plot module: {e}")
            return

        plot_training_progress(
            save_dir=self.training_config.save_dir,
            output_path=os.path.join(self.training_config.save_dir, "training_progress.png"),
            iterations_per_epoch=self.iterations_per_epoch,
            epochs=self.total_epochs,
        )

    def _evaluate(
        self,
        iteration: int,
        current_beta: Optional[float] = None,
        total_iterations: Optional[int] = None,
    ):
        print(f"Evaluating at iteration {iteration}")

        self.agent.eval()
        self.env.context_encoder.eval()

        eval_rewards = []
        eval_actions_a = []
        eval_actions_b = []
        eval_reward_pairs = []
        eval_input_seq_lengths: List[int] = []
        eval_task_types: List[str] = []

        for batch in tqdm(self.eval_loader, desc="Evaluating"):
            episode_data = batch[0]

            prompt = episode_data["input_prompt"]
            tokenized_prompt = self.model_runner.tokenizer(prompt, truncation=False, return_tensors="pt")
            prompt_length = tokenized_prompt.input_ids.size(1)

            for token_budget in TOKEN_BUDGET_CANDIDATES:
                metric_type = str(episode_data.get("metric_type") or "qa_f1_score")
                state = self.env.get_state(
                    prompt=prompt,
                    generation_length=episode_data["generation_length"],
                    answers=episode_data.get("answers", []),
                    all_classes=episode_data.get("all_classes", []),
                    metric_type=metric_type,
                    token_budget=token_budget,
                    dataset=episode_data["dataset"],
                    task_type=episode_data.get("task_type"),
                )
                state = state.to(self.device)

                with torch.no_grad():
                    action, _ = self.agent.act(state, metric_type=metric_type)
                    pred_reward = self.agent.predict_reward(state, action, metric_type=metric_type)
                    pred_reward_val = float(pred_reward.view(-1)[0].item())

                reward, info = self.env.run_with_action(action)

                gt_reward_val = float(reward.item())
                eval_rewards.append(gt_reward_val)
                a_val, b_val = action
                eval_actions_a.append(round(a_val.item(), 2) if isinstance(a_val, torch.Tensor) else a_val)
                eval_actions_b.append(int(b_val.item()) if isinstance(b_val, torch.Tensor) else int(b_val))
                eval_reward_pairs.append((gt_reward_val, pred_reward_val))
                eval_input_seq_lengths.append(int(prompt_length))
                eval_task_types.append(str(episode_data.get("task_type") or "unknown"))

        avg_eval_reward = sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0.0
        avg_eval_reward = round(avg_eval_reward, 4)

        print("Evaluation Results:")
        print(f"  Avg Reward:   {avg_eval_reward:.4f}")
        print()

        eval_data = {
            "iteration": iteration,
            "eval_avg_reward": avg_eval_reward,
            "eval_actions_a": eval_actions_a,
            "eval_actions_b": eval_actions_b,
            "eval_input_seq_lengths": eval_input_seq_lengths,
            "eval_task_types": eval_task_types,
            "eval_reward_pairs": self._format_reward_pairs(eval_reward_pairs, digits=4),
        }
        eval_data = self._serialize_with_fixed_precision(
            eval_data,
            digits_map={"eval_avg_reward": 4, "eval_actions_a": 4},
        )

        with open(
            os.path.join(self.training_config.save_dir, "evaluation_progress.jsonl"), "a"
        ) as f:
            f.write(json.dumps(eval_data) + "\n")

    def _log_progress(
        self,
        iteration: int,
        loss_stats: Dict[str, float],
        iteration_rewards: List,
        iteration_actions_a: List,
        iteration_actions_b: List,
        iteration_reward_pairs: List[Tuple[float, float]],
        iteration_input_seq_lengths: List[int],
        iteration_task_types: List[str],
        current_beta: float,
        log_filename: str,
        log_prefix: str,
    ):
        avg_reward = sum(iteration_rewards) / len(iteration_rewards) if iteration_rewards else 0.0

        prediction_loss = round(loss_stats.get("prediction_loss", 0.0), 4)
        total_loss = round(loss_stats.get("total_loss", 0.0), 4)
        l2_loss = round(loss_stats.get("l2_loss", 0.0), 4)
        grad_norm = round(loss_stats.get("grad_norm", 0.0), 4)

        print(f"Iteration {iteration} ({log_prefix}):")
        print(f"  Avg Reward:        {avg_reward:.4f}")
        print(f"  Prediction Loss:   {prediction_loss:.4f}")
        print(f"  L2 Loss:           {l2_loss:.4f}")
        print(f"  Total Loss:        {total_loss:.4f}")
        print(f"  Grad Norm:         {grad_norm:.4f}")
        print(f"  UCB Beta:          {current_beta:.4f}")
        print()

        current_lr = self.optimizer.param_groups[0]["lr"]

        progress_data = {
            "iteration": iteration,
            "avg_reward": avg_reward.item() if isinstance(avg_reward, torch.Tensor) else avg_reward,
            "prediction_loss": prediction_loss,
            "l2_loss": l2_loss,
            "total_loss": total_loss,
            "grad_norm": grad_norm,
            "learning_rate": current_lr,
            "ucb_beta": current_beta,
            "input_seq_lengths": iteration_input_seq_lengths,
            "task_types": iteration_task_types,
            "actions_a": iteration_actions_a,
            "actions_b": iteration_actions_b,
            "reward_pairs": self._format_reward_pairs(iteration_reward_pairs, digits=4),
        }
        progress_data = self._serialize_with_fixed_precision(
            progress_data,
            digits_map={
                "avg_reward": 4,
                "prediction_loss": 4,
                "l2_loss": 4,
                "total_loss": 4,
                "grad_norm": 4,
                "learning_rate": 4,
                "ucb_beta": 4,
                "actions_a": 4,
            },
        )

        with open(
            os.path.join(self.training_config.save_dir, str(log_filename)), "a"
        ) as f:
            f.write(json.dumps(progress_data) + "\n")

    def _save_checkpoint(self, iteration: int):
        checkpoint_path = os.path.join(self.training_config.save_dir, f"policy_{iteration}.pt")
        checkpoint_data = {
            "iteration": iteration,
            "agent_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model_config,
            "training_config": self.training_config,
        }
        checkpoint_data["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint_data, checkpoint_path)

        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Update configs from checkpoint if available (best-effort).
        if "model_config" in checkpoint:
            self.model_config = checkpoint["model_config"]
        if "training_config" in checkpoint:
            self.training_config = checkpoint["training_config"]

        state_dict = checkpoint.get("agent_state_dict")
        if state_dict is None:
            # Backward compatibility for older checkpoints.
            state_dict = checkpoint.get("policy_state_dict")
        if state_dict is None:
            raise ValueError("Checkpoint missing 'agent_state_dict' (and legacy 'policy_state_dict').")
        self.agent.load_state_dict(state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
        return checkpoint["iteration"]


__all__ = ["A2SFTrainer"]

