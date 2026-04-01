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

TOKEN_BUDGET_CANDIDATES = [128, 256, 512, 1024, 2048, 4096]


class A2SFTrainer:
    REAL_BEST_SCORE_EPS = 1e-9

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
        training_data_list = self.load_training_data()
        self.real_best_reference_avg = self._compute_real_best_reference_avg(training_data_list)
        print(f"RealBest Reference Avg (incl. 0.0): {self.real_best_reference_avg:.4f}")

        # Create datasets (precompute all encoder states before RL optimization loop).
        self.training_dataset = RLDataset(
            training_data_list,
            state_builder=self._build_state_for_sample,
            token_budgets=TOKEN_BUDGET_CANDIDATES,
        )

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

        for sample in tqdm(
            training_data,
            total=len(training_data),
            desc="[dataset] loading training samples",
            unit="sample",
        ):
            if "metric_type" not in sample:
                raise ValueError(
                    "training_data must include 'metric_type' (re-generate with RL/training/make_training_dataset.py)"
                )
            sample["metric_type"] = str(sample["metric_type"])
            if "answers" not in sample:
                sample["answers"] = []
            if "all_classes" not in sample:
                sample["all_classes"] = []
            tokenized_prompt = self.model_runner.tokenizer(
                sample["input_prompt"], truncation=False, return_tensors="pt"
            )
            sample["input_seq_len"] = int(tokenized_prompt.input_ids.size(1))
            outputs_by_budget = sample.get("action_outputs_by_budget")
            scores_by_budget = sample.get("action_scores_by_budget")
            if not isinstance(outputs_by_budget, dict) or not isinstance(scores_by_budget, dict):
                raise ValueError(
                    "training_data must include 'action_outputs_by_budget' and 'action_scores_by_budget' "
                    "(re-generate dataset with make_training_dataset.py)"
                )
            for token_budget in TOKEN_BUDGET_CANDIDATES:
                bkey = str(int(token_budget))
                action_outputs = outputs_by_budget.get(bkey)
                action_scores = scores_by_budget.get(bkey)
                if not isinstance(action_outputs, list):
                    raise ValueError(f"missing action_outputs_by_budget['{bkey}']")
                if not isinstance(action_scores, list) or len(action_scores) != len(action_outputs):
                    raise ValueError(
                        f"missing/invalid action_scores_by_budget['{bkey}'] matching action_outputs_by_budget['{bkey}']"
                    )

        print(f"Loaded {len(training_data)} training samples from {train_path}")
        return training_data

    def _build_state_for_sample(self, sample: Dict[str, Any], token_budget: int) -> torch.Tensor:
        return self.env.get_state(
            prompt=sample["input_prompt"],
            metric_type=str(sample["metric_type"]),
            token_budget=int(token_budget),
            answers=sample.get("answers", []),
            all_classes=sample.get("all_classes", []),
            generation_length=sample["generation_length"],
            dataset=sample.get("dataset"),
            task_type=sample.get("task_type"),
        )

    def _get_cached_state(self, sample: Dict[str, Any], token_budget: int) -> torch.Tensor:
        cached_states = sample.get("cached_states")
        key = str(int(token_budget))
        if isinstance(cached_states, dict) and key in cached_states:
            return cached_states[key].to(self.device, dtype=torch.float32)
        # Backward-compat fallback if cached states are missing.
        return self._build_state_for_sample(sample, token_budget).to(self.device, dtype=torch.float32)

    @staticmethod
    def _compute_real_best_reference_avg(samples: List[Dict[str, Any]]) -> float:
        if not samples:
            return 0.0
        values: List[float] = []
        for sample in samples:
            scores_by_budget = sample.get("action_scores_by_budget", {})
            if not isinstance(scores_by_budget, dict):
                continue
            for token_budget in TOKEN_BUDGET_CANDIDATES:
                scores = scores_by_budget.get(str(int(token_budget)), [])
                if not isinstance(scores, list) or len(scores) == 0:
                    continue
                best_idx = A2SFTrainer._best_action_index_from_scores(scores)
                if 0 <= best_idx < len(scores):
                    values.append(float(scores[best_idx]))
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _best_action_index_from_scores(scores: List[Any]) -> int:
        best_idx = 0
        best_reward = float("-inf")
        for idx, r in enumerate(scores):
            rv = float(r)
            if rv > best_reward:
                best_reward = rv
                best_idx = idx
        return int(best_idx)

    @staticmethod
    def _reward_for_action_index(scores: List[Any], action_idx: int) -> float:
        if not isinstance(scores, list) or not (0 <= action_idx < len(scores)):
            raise IndexError("action_idx out of range for action_scores")
        return float(scores[action_idx])

    def _compute_epoch_mrr(self) -> float:
        """모델 순위 대비 지상 최고 `action_scores` 집합(복수 정답은 점수에 이미 반영됨). 동점이면 최소 rank 사용."""
        reciprocal_ranks: List[float] = []
        score_eps = self.REAL_BEST_SCORE_EPS

        self.agent.eval()
        self.env.context_encoder.eval()

        with torch.no_grad():
            for sample in self.training_dataset.data:
                outputs_by_budget = sample.get("action_outputs_by_budget", {})
                scores_by_budget = sample.get("action_scores_by_budget", {})
                metric_type = str(sample["metric_type"])

                for token_budget in TOKEN_BUDGET_CANDIDATES:
                    budget_key = str(int(token_budget))
                    action_outputs = outputs_by_budget.get(budget_key)
                    if not isinstance(action_outputs, list) or len(action_outputs) == 0:
                        continue

                    action_scores = scores_by_budget.get(budget_key)
                    if not isinstance(action_scores, list) or len(action_scores) != len(action_outputs):
                        continue

                    scores_f = [float(s) for s in action_scores]
                    max_score = max(scores_f)
                    if max_score <= score_eps:
                        continue

                    optimal_indices = [
                        i for i, s in enumerate(scores_f) if s >= max_score - score_eps
                    ]

                    state = self._get_cached_state(sample, token_budget=int(token_budget))

                    reward_pred = self.agent.forward(state, metric_type=metric_type)["reward_pred"][0]
                    ranked_indices = torch.argsort(reward_pred, descending=True)
                    num_actions = len(scores_f)
                    rank_by_action = torch.empty(num_actions, dtype=torch.long, device=reward_pred.device)
                    for pos in range(num_actions):
                        rank_by_action[ranked_indices[pos]] = pos + 1

                    best_rank = min(int(rank_by_action[i].item()) for i in optimal_indices)
                    reciprocal_ranks.append(1.0 / float(best_rank))

        if not reciprocal_ranks:
            return 0.0
        return float(sum(reciprocal_ranks) / len(reciprocal_ranks))

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
                    input_seq_len = int(episode_data.get("input_seq_len", 0))
                    task_type_str = str(episode_data.get("task_type") or "unknown")
                    metric_type = str(episode_data["metric_type"])
                    outputs_by_budget = episode_data.get("action_outputs_by_budget", {})
                    scores_by_budget = episode_data.get("action_scores_by_budget", {})
                    if not isinstance(outputs_by_budget, dict) or not isinstance(scores_by_budget, dict):
                        raise ValueError("training_data must include budget-mapped action outputs/scores")

                    for token_budget in TOKEN_BUDGET_CANDIDATES:
                        budget_key = str(int(token_budget))
                        action_outputs = outputs_by_budget.get(budget_key)
                        action_scores = scores_by_budget.get(budget_key)
                        if not isinstance(action_outputs, list):
                            raise ValueError(f"missing action_outputs_by_budget['{budget_key}']")
                        if not isinstance(action_scores, list) or len(action_scores) != len(action_outputs):
                            raise ValueError(
                                f"missing/invalid action_scores_by_budget['{budget_key}'] matching outputs"
                            )
                        state = self._get_cached_state(episode_data, token_budget=token_budget)

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
                            selected_rewards.append(self._reward_for_action_index(action_scores, action_idx))
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

                        # Add real-best supervised term only when it has a positive signal.
                        max_action_score = max(float(s) for s in action_scores) if action_scores else 0.0
                        if max_action_score > self.REAL_BEST_SCORE_EPS:
                            real_best_idx = int(self._best_action_index_from_scores(action_scores))
                            real_best_a_idx = real_best_idx // self.agent.num_b_values
                            real_best_b_idx = real_best_idx % self.agent.num_b_values
                            real_best_action = (
                                self.agent.a_values[real_best_a_idx].to(self.device).view(1),
                                self.agent.b_values[real_best_b_idx].to(self.device).view(1),
                            )
                            real_best_reward = torch.tensor(
                                self._reward_for_action_index(action_scores, real_best_idx),
                                device=self.device,
                                dtype=torch.float32,
                            )
                            update_samples.append((state, real_best_action, real_best_reward, metric_type))
                            per_label_data.setdefault("real_best", {}).setdefault("used_count", 0)
                            per_label_data["real_best"]["used_count"] += 1
                            gt_real_best_reward = float(real_best_reward.item())
                            pred_real_best_reward = float(reward_pred_all[real_best_idx].item())
                            per_label_data.setdefault("real_best", {}).setdefault("reward_pairs", [])
                            per_label_data["real_best"]["reward_pairs"].append(
                                (gt_real_best_reward, pred_real_best_reward)
                            )
                        else:
                            per_label_data.setdefault("real_best", {}).setdefault("skipped_zero_count", 0)
                            per_label_data["real_best"]["skipped_zero_count"] += 1

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
                    real_best_used_count=int(
                        per_label_data.get("real_best", {}).get("used_count", 0)
                    ),
                    real_best_skipped_zero_count=int(
                        per_label_data.get("real_best", {}).get("skipped_zero_count", 0)
                    ),
                    real_best_reward_pairs=per_label_data.get("real_best", {}).get("reward_pairs", []),
                )
                self._log_real_best_status(
                    iteration=global_iteration,
                    used_count=int(per_label_data.get("real_best", {}).get("used_count", 0)),
                    skipped_zero_count=int(
                        per_label_data.get("real_best", {}).get("skipped_zero_count", 0)
                    ),
                    reward_pairs=per_label_data.get("real_best", {}).get("reward_pairs", []),
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
            epoch_mrr = self._compute_epoch_mrr()
            self._log_epoch_mrr(epoch=epoch + 1, epoch_mrr=epoch_mrr)
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
        real_best_used_count: int,
        real_best_skipped_zero_count: int,
        real_best_reward_pairs: List[Tuple[float, float]],
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
        real_best_avg_gt = (
            sum(gt for gt, _ in real_best_reward_pairs) / len(real_best_reward_pairs)
            if real_best_reward_pairs
            else 0.0
        )
        real_best_avg_pred = (
            sum(pred for _, pred in real_best_reward_pairs) / len(real_best_reward_pairs)
            if real_best_reward_pairs
            else 0.0
        )
        real_best_mae = (
            sum(abs(gt - pred) for gt, pred in real_best_reward_pairs) / len(real_best_reward_pairs)
            if real_best_reward_pairs
            else 0.0
        )

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
        print(f"  RealBest Used:     {real_best_used_count}")
        print(f"  RealBest Skipped0: {real_best_skipped_zero_count}")
        print(f"  RealBest Avg GT:   {real_best_avg_gt:.4f}")
        print(f"  RealBest Avg Pred: {real_best_avg_pred:.4f}")
        print(f"  RealBest MAE:      {real_best_mae:.4f}")
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
            "real_best_used_count": int(real_best_used_count),
            "real_best_skipped_zero_count": int(real_best_skipped_zero_count),
            "real_best_avg_gt_reward": real_best_avg_gt,
            "real_best_avg_pred_reward": real_best_avg_pred,
            "real_best_mae": real_best_mae,
            "real_best_reference_avg_reward": float(self.real_best_reference_avg),
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
                "real_best_avg_gt_reward": 4,
                "real_best_avg_pred_reward": 4,
                "real_best_mae": 4,
                "real_best_reference_avg_reward": 4,
            },
        )

        with open(
            os.path.join(self.training_config.save_dir, "training_progress.jsonl"), "a"
        ) as f:
            f.write(json.dumps(progress_data) + "\n")

    def _log_real_best_status(
        self,
        iteration: int,
        used_count: int,
        skipped_zero_count: int,
        reward_pairs: List[Tuple[float, float]],
    ) -> None:
        avg_gt = (
            sum(gt for gt, _ in reward_pairs) / len(reward_pairs) if reward_pairs else 0.0
        )
        avg_pred = (
            sum(pred for _, pred in reward_pairs) / len(reward_pairs) if reward_pairs else 0.0
        )
        mae = (
            sum(abs(gt - pred) for gt, pred in reward_pairs) / len(reward_pairs)
            if reward_pairs
            else 0.0
        )
        payload = {
            "iteration": int(iteration),
            "real_best_used_count": int(used_count),
            "real_best_skipped_zero_count": int(skipped_zero_count),
            "avg_gt_reward": self._format_fixed(avg_gt, 4),
            "avg_pred_reward": self._format_fixed(avg_pred, 4),
            "mae": self._format_fixed(mae, 4),
            "reward_pairs": self._format_reward_pairs(reward_pairs, digits=4),
        }
        with open(
            os.path.join(self.training_config.save_dir, "training_real_best_status.jsonl"), "a"
        ) as f:
            f.write(json.dumps(payload) + "\n")

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

    def _log_epoch_mrr(self, epoch: int, epoch_mrr: float) -> None:
        print(f"Epoch {epoch} MRR: {epoch_mrr:.4f}")
        epoch_data = {
            "epoch": int(epoch),
            "mrr": self._format_fixed(float(epoch_mrr), 4),
        }
        with open(
            os.path.join(self.training_config.save_dir, "training_epoch_metrics.jsonl"), "a"
        ) as f:
            f.write(json.dumps(epoch_data) + "\n")

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

