import os
import sys
import json
import random
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from ..a2sf_model import ModelConfig
from .training_config import TrainingConfig
from ..agent.neural_ucb_agent import NeuralUCBAgent
from ..env import A2SFEnv, A2SFModelRunner
from .dataloader import RLDataset, rl_collate_fn
from longbench_eval import (
    dataset2metric,
    qa_f1_score,
    qa_f1_zh_score,
    rouge_score,
    rouge_zh_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

TOKEN_BUDGET_CANDIDATES = [1024]


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
        self.metric_fn_registry = {
            "qa_f1_score": qa_f1_score,
            "qa_f1_zh_score": qa_f1_zh_score,
            "rouge_score": rouge_score,
            "rouge_zh_score": rouge_zh_score,
            "classification_score": classification_score,
            "retrieval_score": retrieval_score,
            "retrieval_zh_score": retrieval_zh_score,
            "count_score": count_score,
            "code_sim_score": code_sim_score,
        }

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
        training_data_list = self.load_training_data()

        # Create datasets
        self.training_dataset = RLDataset(training_data_list)

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

        print(f"Loaded {len(self.training_dataset)} training samples")
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

    def load_training_data(self) -> List[Dict[str, Any]]:
        train_path = self.training_config.train_data_path

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training split not found: {train_path}")

        with open(train_path, "r", encoding="utf-8") as f:
            training_data = [json.loads(line) for line in f if line.strip()]

        for sample in training_data:
            sample["metric_type"] = self._resolve_metric_type(sample.get("dataset"))
            if "answers" not in sample:
                sample["answers"] = []
            if "all_classes" not in sample:
                sample["all_classes"] = []

        print(f"Loaded {len(training_data)} training samples from {train_path}")
        return training_data

    @staticmethod
    def _resolve_metric_type(dataset_name: str) -> str:
        dataset_key = str(dataset_name or "").strip().lower()
        fn = dataset2metric.get(dataset_key)
        if fn is None:
            return "qa_f1_score"
        return fn.__name__

    def _compute_reward_from_prediction(
        self,
        pred_text: str,
        metric_type: str,
        answers: List[str],
        all_classes: List[str],
    ) -> float:
        metric_fn = self.metric_fn_registry.get(str(metric_type), qa_f1_score)
        reward_val = 0.0
        if answers:
            for gt in answers:
                reward_val = max(
                    reward_val,
                    float(metric_fn(pred_text, gt, all_classes=all_classes)),
                )
        return reward_val

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
            feature_action_pairs.append((feature_vector.detach(), action_idx, str(metric_type)))

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
            for feature_vector, action_idx, mt in feature_action_pairs:
                self.agent._update_inverse_covariances(feature_vector, action_idx, mt)

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
                label_order = ["best1", "best2", "worst1", "worst2"]
                per_label_data: Dict[str, Dict[str, List[Any]]] = {
                    label: {
                        "rewards": [],
                        "actions_a": [],
                        "actions_b": [],
                        "reward_pairs": [],
                    }
                    for label in label_order
                }
                common_input_seq_lengths: List[int] = []
                common_task_types: List[str] = []
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
                    answers = episode_data.get("answers", []) or []
                    all_classes = episode_data.get("all_classes", []) or []
                    action_outputs = episode_data.get("action_outputs")
                    if not isinstance(action_outputs, list):
                        raise ValueError("training_data must include cached 'action_outputs'")

                    for token_budget in TOKEN_BUDGET_CANDIDATES:
                        state = self.env.get_state(
                            prompt=prompt,
                            metric_type=metric_type,
                            token_budget=token_budget,
                            answers=episode_data.get("answers", []),
                            all_classes=episode_data.get("all_classes", []),
                            generation_length=episode_data["generation_length"],
                            dataset=episode_data["dataset"],
                            task_type=episode_data.get("task_type"),
                        )
                        state = state.to(self.device)

                        with torch.no_grad():
                            reward_pred_all = self.agent.forward(
                                state,
                                metric_type=metric_type,
                            )["reward_pred"][0]
                            _, ucb_scores = self.agent._compute_ucb_scores(
                                state=state,
                                beta=current_beta,
                                metric_type=metric_type,
                            )
                            ucb_scores = ucb_scores[0]

                            desc_idx = torch.argsort(ucb_scores, descending=True)
                            asc_idx = torch.argsort(ucb_scores, descending=False)
                            best1_idx = int(desc_idx[0].item())
                            best2_idx = int(desc_idx[1].item()) if desc_idx.numel() > 1 else best1_idx
                            worst1_idx = int(asc_idx[0].item())
                            worst2_idx = int(asc_idx[1].item()) if asc_idx.numel() > 1 else worst1_idx
                            # Requested order: best1, best2, worst1, worst2
                            selected_indices = torch.tensor(
                                [best1_idx, best2_idx, worst1_idx, worst2_idx],
                                device=ucb_scores.device,
                                dtype=torch.long,
                            )

                            a_idx = selected_indices // self.agent.num_b_values
                            b_idx = selected_indices % self.agent.num_b_values
                            selected_a = self.agent.a_values[a_idx].to(self.device)
                            selected_b = self.agent.b_values[b_idx].to(self.device)
                        selected_rewards: List[float] = []
                        for local_idx in range(selected_indices.numel()):
                            action_idx = int(selected_indices[local_idx].item())
                            pred_text = str(action_outputs[action_idx])
                            selected_rewards.append(
                                self._compute_reward_from_prediction(
                                    pred_text=pred_text,
                                    metric_type=metric_type,
                                    answers=answers,
                                    all_classes=all_classes,
                                )
                            )
                        selected_rewards_t = torch.tensor(
                            selected_rewards, device=self.device, dtype=torch.float32
                        ).view(-1)

                        for local_idx, label in enumerate(label_order):
                            action = (
                                selected_a[local_idx].view(1),
                                selected_b[local_idx].view(1),
                            )
                            reward_val = selected_rewards_t[local_idx]
                            update_samples.append((state, action, reward_val, metric_type))
                            selected_idx = int(selected_indices[local_idx].item())
                            gt_reward_val = float(reward_val.item())
                            pred_reward_val = float(reward_pred_all[selected_idx].item())
                            per_label_data[label]["rewards"].append(gt_reward_val)
                            per_label_data[label]["actions_a"].append(round(float(selected_a[local_idx].item()), 4))
                            per_label_data[label]["actions_b"].append(int(selected_b[local_idx].item()))
                            per_label_data[label]["reward_pairs"].append((gt_reward_val, pred_reward_val))

                        common_input_seq_lengths.append(input_seq_len)
                        common_task_types.append(task_type_str)

                end_time = time.time()
                print(f"Time taken for one iteration: {end_time - start_time} seconds")

                avg_loss_stats = self._neural_ucb_update_batch(
                    samples=update_samples,
                    config=self.training_config,
                    optimizer=self.optimizer,
                )
                best_rewards = per_label_data["best1"]["rewards"]
                best2_rewards = per_label_data["best2"]["rewards"]
                worst1_rewards = per_label_data["worst1"]["rewards"]
                worst2_rewards = per_label_data["worst2"]["rewards"]
                self._log_progress_common(
                    iteration=global_iteration,
                    loss_stats=avg_loss_stats,
                    best_iteration_rewards=best_rewards,
                    best2_iteration_rewards=best2_rewards,
                    worst1_iteration_rewards=worst1_rewards,
                    worst2_iteration_rewards=worst2_rewards,
                    iteration_input_seq_lengths=common_input_seq_lengths,
                    iteration_task_types=common_task_types,
                    current_beta=current_beta,
                )
                for label in label_order:
                    self._log_progress_action_detail(
                        iteration=global_iteration,
                        iteration_actions_a=per_label_data[label]["actions_a"],
                        iteration_actions_b=per_label_data[label]["actions_b"],
                        iteration_reward_pairs=per_label_data[label]["reward_pairs"],
                        log_filename=f"training_progress_{label}.jsonl",
                    )
                global_iteration += 1

            self.scheduler.step()

            last_iter = max(0, global_iteration - 1)
            ce = int(getattr(self.training_config, "checkpoint_every_epochs", 100) or 0)
            if ce > 0 and (epoch + 1) % ce == 0:
                self._save_checkpoint(iteration=last_iter, epoch=epoch + 1)
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

    def _log_progress_common(
        self,
        iteration: int,
        loss_stats: Dict[str, float],
        best_iteration_rewards: List,
        best2_iteration_rewards: List,
        worst1_iteration_rewards: List,
        worst2_iteration_rewards: List,
        iteration_input_seq_lengths: List[int],
        iteration_task_types: List[str],
        current_beta: float,
    ):
        best_avg_reward = (
            sum(best_iteration_rewards) / len(best_iteration_rewards) if best_iteration_rewards else 0.0
        )
        best2_avg_reward = (
            sum(best2_iteration_rewards) / len(best2_iteration_rewards) if best2_iteration_rewards else 0.0
        )
        worst1_avg_reward = (
            sum(worst1_iteration_rewards) / len(worst1_iteration_rewards) if worst1_iteration_rewards else 0.0
        )
        worst2_avg_reward = (
            sum(worst2_iteration_rewards) / len(worst2_iteration_rewards) if worst2_iteration_rewards else 0.0
        )

        prediction_loss = round(loss_stats.get("prediction_loss", 0.0), 4)
        total_loss = round(loss_stats.get("total_loss", 0.0), 4)
        l2_loss = round(loss_stats.get("l2_loss", 0.0), 4)
        grad_norm = round(loss_stats.get("grad_norm", 0.0), 4)

        print(f"Iteration {iteration} (COMMON):")
        print(f"  Best1 Avg Reward:  {best_avg_reward:.4f}")
        print(f"  Best2 Avg Reward:  {best2_avg_reward:.4f}")
        print(f"  Worst2 Avg Reward: {worst2_avg_reward:.4f}")
        print(f"  Worst1 Avg Reward: {worst1_avg_reward:.4f}")
        print(f"  Prediction Loss:   {prediction_loss:.4f}")
        print(f"  L2 Loss:           {l2_loss:.4f}")
        print(f"  Total Loss:        {total_loss:.4f}")
        print(f"  Grad Norm:         {grad_norm:.4f}")
        print(f"  UCB Beta:          {current_beta:.4f}")
        print()

        current_lr = self.optimizer.param_groups[0]["lr"]

        progress_data = {
            "iteration": iteration,
            "best1_avg_reward": (
                best_avg_reward.item() if isinstance(best_avg_reward, torch.Tensor) else best_avg_reward
            ),
            "best2_avg_reward": (
                best2_avg_reward.item() if isinstance(best2_avg_reward, torch.Tensor) else best2_avg_reward
            ),
            "worst1_avg_reward": (
                worst1_avg_reward.item() if isinstance(worst1_avg_reward, torch.Tensor) else worst1_avg_reward
            ),
            "worst2_avg_reward": (
                worst2_avg_reward.item() if isinstance(worst2_avg_reward, torch.Tensor) else worst2_avg_reward
            ),
            "prediction_loss": prediction_loss,
            "l2_loss": l2_loss,
            "total_loss": total_loss,
            "grad_norm": grad_norm,
            "learning_rate": current_lr,
            "ucb_beta": current_beta,
            "input_seq_lengths": iteration_input_seq_lengths,
            "task_types": iteration_task_types,
        }
        progress_data = self._serialize_with_fixed_precision(
            progress_data,
            digits_map={
                "best1_avg_reward": 4,
                "best2_avg_reward": 4,
                "worst1_avg_reward": 4,
                "worst2_avg_reward": 4,
                "prediction_loss": 4,
                "l2_loss": 4,
                "total_loss": 4,
                "grad_norm": 4,
                "learning_rate": 4,
                "ucb_beta": 4,
            },
        )

        with open(
            os.path.join(self.training_config.save_dir, "training_progress.jsonl"), "a"
        ) as f:
            f.write(json.dumps(progress_data) + "\n")

    def _log_progress_action_detail(
        self,
        iteration: int,
        iteration_actions_a: List,
        iteration_actions_b: List,
        iteration_reward_pairs: List[Tuple[float, float]],
        log_filename: str,
    ):
        detail_data = {
            "iteration": iteration,
            "reward_pairs": self._format_reward_pairs(iteration_reward_pairs, digits=4),
            "actions_a": iteration_actions_a,
            "actions_b": iteration_actions_b,
        }
        detail_data = self._serialize_with_fixed_precision(
            detail_data,
            digits_map={
                "actions_a": 4,
            },
        )

        with open(
            os.path.join(self.training_config.save_dir, str(log_filename)), "a"
        ) as f:
            f.write(json.dumps(detail_data) + "\n")

    def _save_checkpoint(self, iteration: int, epoch: int):
        checkpoint_path = os.path.join(self.training_config.save_dir, f"policy_epoch_{epoch}.pt")
        checkpoint_data = {
            "iteration": iteration,
            "epoch": epoch,
            "agent_state_dict": self.agent.state_dict(),
            "attention_encoder_state_dict": {},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model_config,
            "training_config": self.training_config,
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
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

