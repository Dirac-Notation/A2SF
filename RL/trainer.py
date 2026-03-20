import os
import sys
import json
import random
import time
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

from .main import A2SFRLConfig
from .policy import NeuralUCBPolicy
from .env import A2SFEnv
from .runner import A2SFModelRunner

TOKEN_BUDGET_CANDIDATES = [128]

class RLDataset(Dataset):
    """Dataset for RL training episodes"""
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def rl_collate_fn(batch):
    """
    Custom collate function for RL dataset.
    Returns the batch as a list of dictionaries (no tensor conversion).
    """
    return batch

class A2SFTrainer:
    def __init__(self, config: A2SFRLConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        torch.manual_seed(config.seed)
        random.seed(config.seed)        
        
        self.model_runner = A2SFModelRunner(config)
        self.env = A2SFEnv(self.model_runner, config)
        self._target_prob_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        
        # State dimension: [sequence_length_feature, task_type_feature]
        state_dim = 2
        
        self.policy = NeuralUCBPolicy(
            state_dim=state_dim,
            a_values=config.a_values,
            b_values=config.b_values,
        ).to(self.device)
        
        # Optimizer only includes policy parameters
        # Context encoder is metadata-based and frozen
        all_params = list(self.policy.parameters())
        self.optimizer = optim.SGD(all_params, lr=config.lr)
        
        # Load pre-split data produced by datasets/make_training_dataset.py
        training_data_list, eval_data_list = self.load_training_data()
        
        # Create datasets
        self.training_dataset = RLDataset(training_data_list)
        self.eval_dataset = RLDataset(eval_data_list)
        
        # Create data loaders
        # For training: shuffle and use episodes_per_update as batch size
        self.train_loader = DataLoader(
            self.training_dataset,
            batch_size=config.episodes_per_update,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues with model
            pin_memory=False,
            collate_fn=rl_collate_fn  # Use custom collate function
        )
        
        # For evaluation: no shuffle, process all at once
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=1,  # Process one at a time for evaluation
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=rl_collate_fn  # Use custom collate function
        )
        
        print(f"Loaded {len(self.training_dataset)} training samples")
        print(f"Loaded {len(self.eval_dataset)} evaluation samples")
        print(f"Training batches per epoch: {len(self.train_loader)}")
        os.makedirs(config.save_dir, exist_ok=True)

        self.iterations_per_epoch = len(self.train_loader)
        self.total_epochs = max(1, int(config.epochs))
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

    def _serialize_with_fixed_precision(self, payload: Dict[str, Any], digits_map: Dict[str, int]) -> Dict[str, Any]:
        serialized = {}
        for key, value in payload.items():
            if key in digits_map and isinstance(value, (float, int)):
                serialized[key] = self._format_fixed(value, digits_map[key])
            elif key in digits_map and isinstance(value, list):
                serialized[key] = self._format_list_fixed(value, digits_map[key])
            else:
                serialized[key] = value
        return serialized

    def _format_reward_pairs(self, pairs: List[Tuple[float, float]], digits: int = 4) -> List[List[str]]:
        return [
            [self._format_fixed(gt_reward, digits), self._format_fixed(pred_reward, digits)]
            for gt_reward, pred_reward in pairs
        ]

    def _plot_training_progress(self) -> None:
        """Overwrite training progress plot from training/eval jsonl files."""
        train_file = os.path.join(self.config.save_dir, "training_progress.jsonl")
        eval_file = os.path.join(self.config.save_dir, "evaluation_progress.jsonl")
        if not os.path.exists(train_file):
            return

        iterations: List[int] = []
        avg_rewards: List[float] = []
        total_losses: List[float] = []

        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                iterations.append(int(row["iteration"]))
                avg_rewards.append(float(row["avg_reward"]))
                total_losses.append(float(row["total_loss"]))

        if len(iterations) == 0:
            return

        train_iters = np.array(iterations, dtype=np.int64)
        y_reward = np.array(avg_rewards, dtype=np.float64)
        y_loss = np.array(total_losses, dtype=np.float64)

        # Original plot style from runs/plot_training_progress.py
        plt.rcParams.update({
            "font.family": "serif",
            "figure.figsize": (14, 14),
            "figure.dpi": 150,
            "font.size": 22,
            "axes.labelsize": 26,
            "axes.titlesize": 28,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.5,
        })

        # Chunk-average smoothing (same style as original script)
        window_size = max(1, self.iterations_per_epoch)

        def _smooth_chunk(vals: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
            smoothed_vals = []
            epoch_idx = []
            for chunk_id, i in enumerate(range(0, len(vals), w), start=1):
                chunk_vals = vals[i:i + w]
                smoothed_vals.append(float(np.mean(chunk_vals)))
                epoch_idx.append(chunk_id)
            return np.array(epoch_idx, dtype=np.int64), np.array(smoothed_vals, dtype=np.float64)

        x_epoch, y_reward_s = _smooth_chunk(y_reward, window_size)
        _, y_loss_s = _smooth_chunk(y_loss, window_size)

        # Eval reward is already logged once per epoch-end.
        eval_epochs: List[int] = []
        eval_rewards: List[float] = []
        if os.path.exists(eval_file):
            with open(eval_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    iter_idx = int(row["iteration"])
                    eval_epochs.append(iter_idx // window_size + 1)
                    eval_rewards.append(float(row["eval_avg_reward"]))

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, constrained_layout=True, figsize=(14, 14))

        ax1.plot(
            x_epoch,
            y_reward_s,
            marker="o",
            linewidth=4,
            markersize=8,
            color="#4C72B0",
            label="Average Reward (epoch average)",
            zorder=5,
        )
        ax1.set_title("Average Reward Over Training", pad=20)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Average Reward")
        ax1.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)
        ax1.legend(frameon=False, loc="best")
        ax1.margins(x=0.02)

        ax2.plot(
            x_epoch,
            y_loss_s,
            marker="o",
            linewidth=4,
            markersize=8,
            color="#C44E52",
            label="Total Loss (epoch average)",
            zorder=5,
        )
        ax2.set_title("Total Loss Over Training", pad=20)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Total Loss")
        ax2.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)
        ax2.legend(frameon=False, loc="best")
        ax2.margins(x=0.02)

        if len(eval_epochs) > 0:
            ax3.plot(
                np.array(eval_epochs, dtype=np.int64),
                np.array(eval_rewards, dtype=np.float64),
                marker="o",
                linewidth=4,
                markersize=8,
                color="#55A868",
                label="Evaluation Reward",
                zorder=5,
            )
        ax3.set_title("Evaluation Reward Over Training", pad=20)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Eval Reward")
        ax3.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.5)
        ax3.legend(frameon=False, loc="best")
        ax3.margins(x=0.02)

        output_path = os.path.join(self.config.save_dir, "training_progress.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    
    def load_training_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        train_path = self.config.train_data_path
        eval_path = self.config.eval_data_path

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training split not found: {train_path}")
        if not os.path.exists(eval_path):
            raise FileNotFoundError(
                f"Evaluation split not found: {eval_path}. "
                "Run datasets/make_training_dataset.py to generate fixed split files."
            )

        with open(train_path, "r", encoding="utf-8") as f:
            training_data = [json.loads(line) for line in f if line.strip()]
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = [json.loads(line) for line in f if line.strip()]

        def resolve_prob_paths(samples: List[Dict[str, Any]], split_name: str):
            for idx, sample in enumerate(samples):
                prob_file = sample.get("target_prob_file")
                if not prob_file:
                    raise ValueError(
                        f"{split_name} sample #{idx} does not contain 'target_prob_file'. "
                        "Regenerate dataset with updated make_training_dataset.py."
                    )
                resolved_prob_file = prob_file
                if not os.path.isabs(resolved_prob_file):
                    resolved_prob_file = os.path.normpath(resolved_prob_file)
                    if not os.path.exists(resolved_prob_file):
                        candidate_train = os.path.normpath(os.path.join(os.path.dirname(train_path), prob_file))
                        candidate_eval = os.path.normpath(os.path.join(os.path.dirname(eval_path), prob_file))
                        if os.path.exists(candidate_train):
                            resolved_prob_file = candidate_train
                        elif os.path.exists(candidate_eval):
                            resolved_prob_file = candidate_eval
                sample["target_prob_file"] = resolved_prob_file
                if not os.path.exists(resolved_prob_file):
                    raise FileNotFoundError(f"Target probability file not found: {resolved_prob_file}")

        resolve_prob_paths(training_data, "train")
        resolve_prob_paths(eval_data, "eval")

        if self.config.eval_samples > 0 and len(eval_data) > self.config.eval_samples:
            rng = random.Random(self.config.seed)
            rng.shuffle(eval_data)
            eval_data = eval_data[: self.config.eval_samples]
            print(f"Subsampled eval split to {len(eval_data)} samples (eval_samples={self.config.eval_samples})")

        print(f"Loaded {len(training_data)} training samples from {train_path}")
        print(f"Loaded {len(eval_data)} evaluation samples from {eval_path}")
        return training_data, eval_data

    def _load_target_prob_data(self, path: str) -> Dict[str, torch.Tensor]:
        if path not in self._target_prob_cache:
            data = torch.load(path, map_location="cpu")
            required_keys = {"answer_token_ids", "teacher_topk_indices", "teacher_topk_probs"}
            missing = required_keys - set(data.keys())
            if missing:
                raise KeyError(f"Missing keys {missing} in target probability file: {path}")
            self._target_prob_cache[path] = data
        return self._target_prob_cache[path]
    
    def _split_data(self, all_data: List[Dict[str, Any]], eval_samples: int, seed: int) -> tuple:
        """
        Split data into training and evaluation sets with balanced task distribution.
        Uses fixed seed to ensure reproducibility.
        
        Args:
            all_data: All loaded data samples
            eval_samples: Number of samples to use for evaluation
            seed: Random seed for splitting
        
        Returns:
            tuple: (training_data, eval_data)
        """
        if len(all_data) <= eval_samples:
            raise ValueError(f"Not enough data: {len(all_data)} total samples, but {eval_samples} evaluation samples requested")
        
        def get_task_type(sample: Dict[str, Any]) -> str:
            """Use sample-provided task type directly."""
            task_type = sample.get("task_type")
            if not isinstance(task_type, str):
                return "unknown"
            task_type = task_type.strip()
            return task_type if task_type else "unknown"
        
        # Group data by task type
        from collections import defaultdict
        task_groups = defaultdict(list)
        for sample in all_data:
            task_type = get_task_type(sample)
            task_groups[task_type].append(sample)
        
        # Use fixed seed for reproducible split
        rng = random.Random(seed)
        
        # Shuffle each task group separately
        for task_type in task_groups:
            rng.shuffle(task_groups[task_type])
        
        # Calculate samples per task for evaluation
        num_tasks = len([g for g in task_groups.values() if len(g) > 0])
        if num_tasks == 0:
            raise ValueError("No valid task types found in data")
        
        eval_samples_per_task = eval_samples // num_tasks
        remaining_eval_samples = eval_samples % num_tasks
        
        # Collect evaluation samples from each task
        eval_data = []
        task_eval_counts = {}
        
        for task_type, samples in task_groups.items():
            if len(samples) == 0:
                continue
            
            # Calculate how many samples to take from this task
            samples_to_take = eval_samples_per_task
            if remaining_eval_samples > 0:
                samples_to_take += 1
                remaining_eval_samples -= 1
            
            # Take samples from this task (up to available samples)
            actual_take = min(samples_to_take, len(samples))
            task_eval = samples[:actual_take]
            eval_data.extend(task_eval)
            task_eval_counts[task_type] = actual_take
        
        # Collect training samples (remaining samples from each task)
        training_data = []
        for task_type, samples in task_groups.items():
            if len(samples) == 0:
                continue
            eval_count = task_eval_counts.get(task_type, 0)
            training_data.extend(samples[eval_count:])
        
        # Shuffle both sets
        rng.shuffle(eval_data)
        rng.shuffle(training_data)
        
        # Verify no overlap
        eval_set = {id(d) for d in eval_data}
        train_set = {id(d) for d in training_data}
        assert len(eval_set & train_set) == 0, "Training and evaluation data must be completely separate!"
        
        # Print task distribution
        print(f"\nTask distribution in evaluation set:")
        for task_type, count in sorted(task_eval_counts.items()):
            total_in_task = len(task_groups[task_type])
            print(f"  {task_type:20s}: {count:3d} / {total_in_task:5d} ({count/total_in_task*100:.1f}%)")
        print(f"Total evaluation samples: {len(eval_data)}")
        print(f"Total training samples: {len(training_data)}")
        
        return training_data, eval_data
    
    def _neural_ucb_update_batch(
        self,
        samples: List[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
        config,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, Any]:
        """
        Batch update: accumulate MAE losses over (state, action, reward) samples
        and run a single optimizer step. Also returns (gt, pred) pairs for logging.
        """
        if len(samples) == 0:
            return {
                "prediction_loss": 0.0,
                "l2_loss": 0.0,
                "total_loss": 0.0,
                "grad_norm": 0.0,
                "reward_pairs": [],
            }

        self.policy.train()
        self.env.context_encoder.eval()

        prediction_mse_losses = []
        reward_pairs: List[Tuple[float, float]] = []
        feature_action_pairs = []

        for state, action, reward in samples:
            if state.ndim == 1:
                state = state.unsqueeze(0)

            out = self.policy.forward(state)
            reward_pred = out["reward_pred"]
            feature_vector = out["feature_vector"]

            action_idx = self.policy._get_action_indices(action)
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
        for param in self.policy.parameters():
            if param.requires_grad:
                l2_loss += torch.sum(param ** 2)
        l2_loss = config.l2_coef * l2_loss

        total_loss = prediction_loss + l2_loss

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm_sq = 0.0
        for param in self.policy.parameters():
            if param.grad is not None:
                g = param.grad.detach()
                grad_norm_sq += float(torch.sum(g * g).item())
        grad_norm = grad_norm_sq ** 0.5
        optimizer.step()

        with torch.no_grad():
            for feature_vector, action_idx in feature_action_pairs:
                self.policy._update_inverse_covariances(feature_vector, action_idx)

        return {
            "prediction_loss": float(prediction_loss.item()),
            "l2_loss": float(l2_loss.item()),
            "total_loss": float(total_loss.item()),
            "grad_norm": float(grad_norm),
            "reward_pairs": reward_pairs,
        }
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create cosine learning rate scheduler"""
        return lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.scheduler_t_max,
            eta_min=self.config.lr * 0.01  # Minimum learning rate (1% of initial)
        )
    
    def _get_ucb_beta(self, iteration: int, total_iterations: int) -> float:
        """
        Linearly decay UCB beta from config.ucb_beta_max to config.ucb_beta_min.
        """
        if total_iterations <= 1:
            return float(self.config.ucb_beta_max)
        
        progress = iteration / float(total_iterations - 1)
        beta = self.config.ucb_beta_max + (self.config.ucb_beta_min - self.config.ucb_beta_max) * progress
        return float(beta)

    def train(self, num_epochs: int = 1) -> int:
        print(f"Starting training for {num_epochs} epochs")

        global_iteration = 0
        total_epochs = max(1, int(num_epochs))

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            # Epoch-based UCB beta scheduling: fixed within an epoch.
            current_beta = self._get_ucb_beta(epoch, total_epochs)

            for batch in self.train_loader:
                iteration_rewards = []
                iteration_actions_a = []
                iteration_actions_b = []
                iteration_reward_pairs = []
                update_samples = []

                start_time = time.time()

                # Process each episode in the batch
                for episode_data in batch:
                    prompt = episode_data["input_prompt"]
                    token_budget_candidates = TOKEN_BUDGET_CANDIDATES
                    target_prob_data = self._load_target_prob_data(episode_data["target_prob_file"])
                    for token_budget in token_budget_candidates:
                        state = self.env.encode_to_state(
                            prompt=prompt,
                            generation_length=episode_data["generation_length"],
                            target_prob_data=target_prob_data,
                            token_budget=token_budget,
                            dataset=episode_data["dataset"],
                            task_type=episode_data.get("task_type"),
                        )
                        state = state.to(self.device)

                        action, ucb_value = self.policy.act_with_ucb(state, beta=current_beta)
                        reward, info = self.env.step(action)
                        gt_reward_val = reward.item() if isinstance(reward, torch.Tensor) else float(reward)

                        update_samples.append((state, action, reward))

                        iteration_rewards.append(gt_reward_val)
                        a_val, b_val = action
                        iteration_actions_a.append(round(a_val.item(), 2) if isinstance(a_val, torch.Tensor) else a_val)
                        iteration_actions_b.append(int(b_val.item()) if isinstance(b_val, torch.Tensor) else int(b_val))

                end_time = time.time()
                print(f"Time taken for one iteration: {end_time - start_time} seconds")

                # Single update after evaluating all budgets for selected episodes.
                avg_loss_stats = self._neural_ucb_update_batch(
                    samples=update_samples,
                    config=self.config,
                    optimizer=self.optimizer,
                )
                iteration_reward_pairs = avg_loss_stats.get("reward_pairs", [])

                self._log_progress(
                    iteration=global_iteration,
                    loss_stats=avg_loss_stats,
                    iteration_rewards=iteration_rewards,
                    iteration_actions_a=iteration_actions_a,
                    iteration_actions_b=iteration_actions_b,
                    iteration_reward_pairs=iteration_reward_pairs,
                    current_beta=current_beta,
                )
                global_iteration += 1

            # Epoch-based cosine scheduler: step once per epoch.
            self.scheduler.step()

            # Run evaluation/checkpoint/plot once per epoch end.
            last_iter = max(0, global_iteration - 1)
            self._save_checkpoint(last_iter)
            self._evaluate(last_iter, current_beta=current_beta, total_iterations=total_epochs)
            self._plot_training_progress()

        return max(0, global_iteration - 1)
    
    def _evaluate(self, iteration: int, current_beta: Optional[float] = None, total_iterations: Optional[int] = None):
        print(f"Evaluating at iteration {iteration}")
        eval_beta = 0.0
        
        # Set to evaluation mode
        self.policy.eval()
        self.env.context_encoder.eval()
        
        eval_rewards = []
        eval_actions_a = []
        eval_actions_b = []
        eval_reward_pairs = []
        
        for batch in tqdm(self.eval_loader, desc="Evaluating"):
            # DataLoader with batch_size=1 returns a list with one element
            episode_data = batch[0]
            
            # Select random token budget from candidates that are less than prompt length
            prompt = episode_data["input_prompt"]
            tokenized_prompt = self.model_runner.tokenizer(prompt, truncation=False, return_tensors="pt")
            prompt_length = tokenized_prompt.input_ids.size(1)
            
            # Evaluate all budget candidates sequentially (no random sampling).
            token_budget_candidates = TOKEN_BUDGET_CANDIDATES
            budgets_to_eval = token_budget_candidates

            target_prob_data = self._load_target_prob_data(episode_data["target_prob_file"])
            for token_budget in budgets_to_eval:
                state = self.env.encode_to_state(
                    prompt=prompt,
                    generation_length=episode_data["generation_length"],
                    target_prob_data=target_prob_data,
                    token_budget=token_budget,
                    dataset=episode_data["dataset"],
                    task_type=episode_data.get("task_type"),
                )
                state = state.to(self.device)

                with torch.no_grad():
                    action, _ = self.policy.act(state)
                    pred_reward = self.policy.predict_reward(state, action)
                    pred_reward_val = float(pred_reward.view(-1)[0].item())

                reward, info = self.env.step(action)

                gt_reward_val = float(reward.item())
                eval_rewards.append(gt_reward_val)
                a_val, b_val = action
                eval_actions_a.append(round(a_val.item(), 2) if isinstance(a_val, torch.Tensor) else a_val)
                eval_actions_b.append(int(b_val.item()) if isinstance(b_val, torch.Tensor) else int(b_val))
                eval_reward_pairs.append((gt_reward_val, pred_reward_val))
        
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
            "eval_reward_pairs": self._format_reward_pairs(eval_reward_pairs, digits=4),
        }
        eval_data = self._serialize_with_fixed_precision(
            eval_data,
            digits_map={
                "eval_avg_reward": 4,
                "eval_actions_a": 4,
            },
        )
        
        with open(os.path.join(self.config.save_dir, "evaluation_progress.jsonl"), "a") as f:
            f.write(json.dumps(eval_data) + "\n")

    def _log_progress(
        self,
        iteration: int,
        loss_stats: Dict[str, float],
        iteration_rewards: List,
        iteration_actions_a: List,
        iteration_actions_b: List,
        iteration_reward_pairs: List[Tuple[float, float]],
        current_beta: float,
    ):
        avg_reward = sum(iteration_rewards) / len(iteration_rewards) if iteration_rewards else 0.0
        
        prediction_loss = round(loss_stats.get("prediction_loss", 0.0), 4)
        total_loss = round(loss_stats.get("total_loss", 0.0), 4)
        l2_loss = round(loss_stats.get("l2_loss", 0.0), 4)
        grad_norm = round(loss_stats.get("grad_norm", 0.0), 4)
        
        print(f"Iteration {iteration}:")
        print(f"  Avg Reward:        {avg_reward:.4f}")
        print(f"  Prediction Loss:   {prediction_loss:.4f}")
        print(f"  L2 Loss:           {l2_loss:.4f}")
        print(f"  Total Loss:        {total_loss:.4f}")
        print(f"  Grad Norm:         {grad_norm:.4f}")
        print(f"  UCB Beta:          {current_beta:.4f}")
        print()
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        progress_data = {
            "iteration": iteration,
            "avg_reward": avg_reward.item() if isinstance(avg_reward, torch.Tensor) else avg_reward,
            "prediction_loss": prediction_loss,
            "l2_loss": l2_loss,
            "total_loss": total_loss,
            "grad_norm": grad_norm,
            "learning_rate": current_lr,
            "ucb_beta": current_beta,
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
        
        with open(os.path.join(self.config.save_dir, "training_progress.jsonl"), "a") as f:
            f.write(json.dumps(progress_data) + "\n")

    def _save_checkpoint(self, iteration: int):
        checkpoint_path = os.path.join(self.config.save_dir, f"policy_{iteration}.pt")
        checkpoint_data = {
            "iteration": iteration,
            "policy_state_dict": self.policy.state_dict(),
            "attention_encoder_state_dict": {},  # Encoder is frozen, no state to save
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        checkpoint_data["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint_data, checkpoint_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Update config from checkpoint if available
        if "config" in checkpoint:
            checkpoint_config = checkpoint["config"]
            # Update important config values that affect model initialization
            if hasattr(checkpoint_config, "a_values"):
                self.config.a_values = checkpoint_config.a_values
            if hasattr(checkpoint_config, "b_values"):
                self.config.b_values = checkpoint_config.b_values
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        
        # Encoder is frozen and metadata-based, no need to load state
        if "attention_encoder_state_dict" in checkpoint and checkpoint["attention_encoder_state_dict"]:
            print("Note: Context encoder is metadata-based and frozen")
        
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
        return checkpoint["iteration"]