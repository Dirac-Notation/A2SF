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

        # Load pre-split data generated under `RL/training/data/`.
        training_data_list = self.load_training_data()

        # 모든 action의 reward가 0인 샘플 제외
        bkey = str(int(self.training_config.token_budget))
        before_count = len(training_data_list)
        training_data_list = [
            s for s in training_data_list
            if any(float(v) > 0 for v in s.get("action_scores_by_budget", {}).get(bkey, [0]))
        ]
        skipped = before_count - len(training_data_list)
        if skipped > 0:
            print(f"Filtered out {skipped}/{before_count} samples with all-zero rewards (budget={bkey})")

        state_dim = int(self.env.context_encoder.output_dim)
        num_heads = int(self.env.context_encoder.num_heads)
        num_metric_types = int(self.env.context_encoder.num_metric_types)

        # 실제 학습 데이터에 등장하는 metric만 head로 생성
        metric_heads = sorted({str(s["metric_type"]) for s in training_data_list if "metric_type" in s})
        if not metric_heads:
            metric_heads = ["qa_f1_score"]
        print(f"Metric heads: {metric_heads}")

        self.agent = NeuralUCBAgent(
            state_dim=state_dim,
            a_values=self.model_config.a_values,
            b_values=self.model_config.b_values,
            metric_heads=metric_heads,
            num_heads=num_heads,
            num_metric_types=num_metric_types,
        ).to(self.device)

        # Optimizer only includes agent parameters
        # Context encoder is metadata-based and frozen
        all_params = list(self.agent.parameters())
        self.optimizer = optim.SGD(all_params, lr=self.training_config.lr)

        self.real_best_reference_avg = self._compute_real_best_reference_avg(
            training_data_list, int(self.training_config.token_budget)
        )
        self.real_best_reference_avg_by_task = self._compute_real_best_reference_avg_by_task(
            training_data_list, int(self.training_config.token_budget)
        )
        print(f"RealBest Reference Avg (excl. all-zero): {self.real_best_reference_avg:.4f}")
        for t, v in sorted(self.real_best_reference_avg_by_task.items()):
            print(f"  {t}: {v:.4f}")

        # Create datasets (precompute all encoder states before RL optimization loop).
        self.training_dataset = RLDataset(
            training_data_list,
            state_builder=self._build_state_for_sample,
            token_budgets=[int(self.training_config.token_budget)],
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
            bkey = str(int(self.training_config.token_budget))
            action_outputs = outputs_by_budget.get(bkey)
            action_scores = scores_by_budget.get(bkey)
            if not isinstance(action_outputs, list):
                raise ValueError(
                    f"missing action_outputs_by_budget['{bkey}'] "
                    f"(train token_budget={self.training_config.token_budget})"
                )
            if not isinstance(action_scores, list) or len(action_scores) != len(action_outputs):
                raise ValueError(
                    f"missing/invalid action_scores_by_budget['{bkey}'] matching outputs "
                    f"(train token_budget={self.training_config.token_budget})"
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
    def _compute_real_best_reference_avg(samples: List[Dict[str, Any]], token_budget: int) -> float:
        if not samples:
            return 0.0
        bkey = str(int(token_budget))
        values: List[float] = []
        for sample in samples:
            scores_by_budget = sample.get("action_scores_by_budget", {})
            if not isinstance(scores_by_budget, dict):
                continue
            scores = scores_by_budget.get(bkey, [])
            if not isinstance(scores, list) or len(scores) == 0:
                continue
            best_idx = A2SFTrainer._best_action_index_from_scores(scores)
            if 0 <= best_idx < len(scores):
                values.append(float(scores[best_idx]))
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _compute_real_best_reference_avg_by_task(
        samples: List[Dict[str, Any]], token_budget: int
    ) -> Dict[str, float]:
        bkey = str(int(token_budget))
        per_task: Dict[str, List[float]] = {}
        for sample in samples:
            scores_by_budget = sample.get("action_scores_by_budget", {})
            if not isinstance(scores_by_budget, dict):
                continue
            scores = scores_by_budget.get(bkey, [])
            if not isinstance(scores, list) or len(scores) == 0:
                continue
            task = str(sample.get("task_type") or "unknown")
            best_idx = A2SFTrainer._best_action_index_from_scores(scores)
            if 0 <= best_idx < len(scores):
                per_task.setdefault(task, []).append(float(scores[best_idx]))
        return {t: float(sum(v) / len(v)) for t, v in per_task.items() if v}

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
        token_budget = int(self.training_config.token_budget)
        budget_key = str(token_budget)

        self.agent.eval()
        self.env.context_encoder.eval()

        with torch.no_grad():
            for sample in self.training_dataset.data:
                outputs_by_budget = sample.get("action_outputs_by_budget", {})
                scores_by_budget = sample.get("action_scores_by_budget", {})
                metric_type = str(sample["metric_type"])

                action_outputs = outputs_by_budget.get(budget_key)
                if not isinstance(action_outputs, list) or len(action_outputs) == 0:
                    continue

                action_scores = scores_by_budget.get(budget_key)
                if not isinstance(action_scores, list) or len(action_scores) != len(action_outputs):
                    continue

                scores_f = [float(s) for s in action_scores]
                max_score = max(scores_f)

                optimal_indices = [
                    i for i, s in enumerate(scores_f) if s >= max_score - score_eps
                ]

                state = self._get_cached_state(sample, token_budget=token_budget)

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
                "prediction_rmse": 0.0,
                "l2_loss": 0.0,
                "total_loss": 0.0,
                "grad_norm": 0.0,
                "reward_pairs": [],
            }

        self.agent.train()
        self.env.context_encoder.eval()

        prediction_mse_terms: List[torch.Tensor] = []
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
            prediction_mse_terms.append(F.mse_loss(selected_predict, actual_reward))
            reward_pairs.append(
                (
                    float(actual_reward.view(-1)[0].detach().item()),
                    float(selected_predict.view(-1)[0].detach().item()),
                )
            )
            feature_action_pairs.append((feature_vector.detach(), action_idx, str(metric_type)))

        prediction_rmse = torch.sqrt(torch.stack(prediction_mse_terms).mean())

        l2_loss = 0.0
        for param in self.agent.parameters():
            if param.requires_grad:
                l2_loss += torch.sum(param ** 2)
        l2_loss = config.l2_coef * l2_loss

        total_loss = prediction_rmse + l2_loss

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
            "prediction_rmse": float(prediction_rmse.item()),
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
        self.last_iteration = 0
        total_epochs = max(1, int(num_epochs))

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            current_beta = self._get_ucb_beta(epoch, total_epochs)

            for batch in self.train_loader:
                num_best = 4
                label_order = [f"best{i+1}" for i in range(num_best)]
                per_label_data: Dict[str, Dict[str, List[Any]]] = {
                    label: {
                        "rewards": [],
                        "actions_a": [],
                        "actions_b": [],
                        "reward_pairs": [],
                    }
                    for label in label_order
                }
                real_best_reward_pairs: List[Tuple[float, float]] = []
                common_input_seq_lengths: List[int] = []
                common_task_types: List[str] = []
                update_samples = []
                token_budget = int(self.training_config.token_budget)
                budget_key = str(token_budget)

                start_time = time.time()

                for episode_data in batch:
                    input_seq_len = int(episode_data.get("input_seq_len", 0))
                    task_type_str = str(episode_data.get("task_type") or "unknown")
                    metric_type = str(episode_data["metric_type"])
                    outputs_by_budget = episode_data.get("action_outputs_by_budget", {})
                    scores_by_budget = episode_data.get("action_scores_by_budget", {})
                    if not isinstance(outputs_by_budget, dict) or not isinstance(scores_by_budget, dict):
                        raise ValueError("training_data must include budget-mapped action outputs/scores")

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
                        actual_best = min(num_best, desc_idx.numel())
                        selected_indices = desc_idx[:actual_best]
                        if actual_best < num_best:
                            pad = selected_indices[-1:].expand(num_best - actual_best)
                            selected_indices = torch.cat([selected_indices, pad])

                        a_idx = selected_indices // self.agent.num_b_values
                        b_idx = selected_indices % self.agent.num_b_values
                        selected_a = self.agent.a_values[a_idx].to(self.device)
                        selected_b = self.agent.b_values[b_idx].to(self.device)

                    for local_idx, label in enumerate(label_order):
                        action_idx = int(selected_indices[local_idx].item())
                        reward_val = self._reward_for_action_index(action_scores, action_idx)
                        reward_t = torch.tensor(reward_val, device=self.device, dtype=torch.float32)

                        action = (selected_a[local_idx].view(1), selected_b[local_idx].view(1))
                        update_samples.append((state, action, reward_t, metric_type))

                        gt_reward_val = float(reward_val)
                        pred_reward_val = float(reward_pred_all[action_idx].item())
                        per_label_data[label]["rewards"].append(gt_reward_val)
                        per_label_data[label]["actions_a"].append(round(float(selected_a[local_idx].item()), 4))
                        per_label_data[label]["actions_b"].append(int(selected_b[local_idx].item()))
                        per_label_data[label]["reward_pairs"].append((gt_reward_val, pred_reward_val))

                    # Real best: 로깅 전용 (손실/역전파에는 넣지 않음).
                    real_best_idx = int(self._best_action_index_from_scores(action_scores))
                    gt_real_best_reward = float(
                        self._reward_for_action_index(action_scores, real_best_idx)
                    )
                    pred_real_best_reward = float(reward_pred_all[real_best_idx].item())
                    real_best_reward_pairs.append((gt_real_best_reward, pred_real_best_reward))

                    common_input_seq_lengths.append(input_seq_len)
                    common_task_types.append(task_type_str)

                end_time = time.time()
                print(f"Time taken for one iteration: {end_time - start_time} seconds")

                avg_loss_stats = self._neural_ucb_update_batch(
                    samples=update_samples,
                    config=self.training_config,
                    optimizer=self.optimizer,
                )
                self._log_progress_common(
                    iteration=global_iteration,
                    loss_stats=avg_loss_stats,
                    per_label_rewards={label: per_label_data[label]["rewards"] for label in label_order},
                    iteration_input_seq_lengths=common_input_seq_lengths,
                    iteration_task_types=common_task_types,
                    current_beta=current_beta,
                    real_best_reward_pairs=real_best_reward_pairs,
                )
                self._log_real_best_status(
                    iteration=global_iteration,
                    reward_pairs=real_best_reward_pairs,
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
                self.last_iteration = max(0, global_iteration - 1)

            self.scheduler.step()

            last_iter = self.last_iteration
            ce = int(getattr(self.training_config, "checkpoint_every_epochs", 100) or 0)
            if ce > 0 and (epoch + 1) % ce == 0:
                self._save_checkpoint(iteration=last_iter, epoch=epoch + 1)
            epoch_mrr = self._compute_epoch_mrr()
            self._log_epoch_mrr(epoch=epoch + 1, epoch_mrr=epoch_mrr)
            self._plot_training_progress()

        return self.last_iteration

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
        per_label_rewards: Dict[str, List],
        iteration_input_seq_lengths: List[int],
        iteration_task_types: List[str],
        current_beta: float,
        real_best_reward_pairs: List[Tuple[float, float]],
    ):
        avg_rewards = {}
        for label, rewards in per_label_rewards.items():
            avg_rewards[label] = sum(rewards) / len(rewards) if rewards else 0.0

        pred_rmse = loss_stats.get("prediction_rmse", loss_stats.get("prediction_loss", 0.0))
        prediction_rmse = round(float(pred_rmse), 4)
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

        print(f"Iteration {iteration}:")
        for label, avg in avg_rewards.items():
            print(f"  {label} Avg Reward: {avg:.4f}")
        print(f"  Prediction RMSE:   {prediction_rmse:.4f}")
        print(f"  L2 Loss:           {l2_loss:.4f}")
        print(f"  Total Loss:        {total_loss:.4f}")
        print(f"  Grad Norm:         {grad_norm:.4f}")
        print(f"  UCB Beta:          {current_beta:.4f}")
        print(f"  RealBest Avg GT:   {real_best_avg_gt:.4f}")
        print(f"  RealBest Avg Pred: {real_best_avg_pred:.4f}")
        print()

        current_lr = self.optimizer.param_groups[0]["lr"]

        progress_data: Dict[str, Any] = {"iteration": iteration}
        digits_map: Dict[str, int] = {}
        for label, avg in avg_rewards.items():
            key = f"{label}_avg_reward"
            progress_data[key] = avg.item() if isinstance(avg, torch.Tensor) else avg
            digits_map[key] = 4

        # per-sample rewards for each best label (aligned with task_types)
        for label, rewards in per_label_rewards.items():
            key = f"{label}_rewards"
            progress_data[key] = [round(float(r), 4) for r in rewards]

        progress_data.update({
            "prediction_rmse": prediction_rmse,
            "l2_loss": l2_loss,
            "total_loss": total_loss,
            "grad_norm": grad_norm,
            "learning_rate": current_lr,
            "ucb_beta": current_beta,
            "real_best_avg_gt_reward": real_best_avg_gt,
            "real_best_avg_pred_reward": real_best_avg_pred,
            "real_best_reference_avg_reward": float(self.real_best_reference_avg),
            "real_best_reference_avg_by_task": {
                t: round(float(v), 4) for t, v in self.real_best_reference_avg_by_task.items()
            },
            "input_seq_lengths": iteration_input_seq_lengths,
            "task_types": iteration_task_types,
        })
        digits_map.update({
            "prediction_rmse": 4,
            "l2_loss": 4,
            "total_loss": 4,
            "grad_norm": 4,
            "learning_rate": 4,
            "ucb_beta": 4,
            "real_best_avg_gt_reward": 4,
            "real_best_avg_pred_reward": 4,
            "real_best_reference_avg_reward": 4,
        })
        progress_data = self._serialize_with_fixed_precision(progress_data, digits_map=digits_map)

        with open(
            os.path.join(self.training_config.save_dir, "training_progress.jsonl"), "a"
        ) as f:
            f.write(json.dumps(progress_data) + "\n")

    def _log_real_best_status(
        self,
        iteration: int,
        reward_pairs: List[Tuple[float, float]],
    ) -> None:
        avg_gt = (
            sum(gt for gt, _ in reward_pairs) / len(reward_pairs) if reward_pairs else 0.0
        )
        avg_pred = (
            sum(pred for _, pred in reward_pairs) / len(reward_pairs) if reward_pairs else 0.0
        )
        payload = {
            "iteration": int(iteration),
            "avg_gt_reward": self._format_fixed(avg_gt, 4),
            "avg_pred_reward": self._format_fixed(avg_pred, 4),
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

    def _build_arch_config(self) -> Dict[str, Any]:
        """체크포인트에서 agent를 재구성할 때 필요한 모든 아키텍처 파라미터."""
        return {
            "state_dim": int(self.agent.state_dim),
            "num_heads": int(self.agent.num_heads),
            "num_metric_types": int(self.agent.num_metric_types),
            "feature_dim": int(self.agent.feature_dim),
            "topk": int(self.agent.topk),
            "num_actions": int(self.agent.num_actions),
            "num_a_values": int(self.agent.num_a_values),
            "num_b_values": int(self.agent.num_b_values),
            "metric_heads": list(self.agent.metric_heads),
            "a_values": self.agent.a_values.detach().cpu(),
            "b_values": self.agent.b_values.detach().cpu(),
        }

    def _agent_weights_only(self) -> Dict[str, torch.Tensor]:
        """추론에 필요한 weight만 추출 (inverse_lambdas, action_counts 등 제외)."""
        full = self.agent.state_dict()
        # Exclude large/training-only buffers.
        exclude_prefixes = ("inverse_lambdas", "action_counts")
        return {
            k: v for k, v in full.items()
            if not any(k.startswith(p) for p in exclude_prefixes)
        }

    def _save_checkpoint(self, iteration: int, epoch: int):
        checkpoint_path = os.path.join(self.training_config.save_dir, f"policy_epoch_{epoch}.pt")
        checkpoint_data = {
            "iteration": iteration,
            "epoch": epoch,
            "agent_state_dict": self.agent.state_dict(),
            "arch_config": self._build_arch_config(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model_config,
            "training_config": self.training_config,
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def save_final_checkpoint(self, iteration: int) -> str:
        """추론 전용 최소 체크포인트. Optimizer/Scheduler/대형 buffer 제외."""
        final_path = os.path.join(self.training_config.save_dir, "policy_final.pt")
        payload = {
            "iteration": iteration,
            "agent_state_dict": self._agent_weights_only(),
            "arch_config": self._build_arch_config(),
        }
        torch.save(payload, final_path)
        print(f"Saved final checkpoint (inference-only): {final_path}")
        return final_path

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
        # Final checkpoint은 buffer 일부가 빠져 있을 수 있어 strict=False.
        missing, unexpected = self.agent.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"[load_checkpoint] unexpected keys: {unexpected}")

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
        return checkpoint["iteration"]


__all__ = ["A2SFTrainer"]

