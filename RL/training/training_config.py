from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TrainingConfig:
    # ----- Optimization / RL hyperparameters -----
    lr: float = 1e-1
    ucb_beta_max: float = 1.0
    ucb_beta_min: float = 0.1
    l2_coef: float = 1e-6

    # ----- Training configuration -----
    epochs: int = 2000
    episodes_per_update: int = 32
    # 학습에 사용할 token budget (jsonl의 action_*_by_budget 키와 일치해야 함).
    token_budget: int = 128

    # ----- Reproducibility / IO -----
    seed: int = 42
    save_dir: str = "runs/a2sf_rl"
    resume: Optional[str] = None
    # 0이면 에폭 중간 체크포인트를 저장하지 않는다. 최종 저장은 run.py에서 policy_final.pt로 수행.
    checkpoint_every_epochs: int = 100

    # ----- Dataset paths (fixed splits) -----
    train_data_path: str = "RL/training/data/training_data.jsonl"

    @classmethod
    def from_args(cls, argv=None) -> "TrainingConfig":
        default_cfg = cls()

        parser = argparse.ArgumentParser(description="Train A2SF RL Agent (training config)")
        parser.add_argument("--save_dir", type=str, default=default_cfg.save_dir)
        parser.add_argument("--resume", type=str, default=default_cfg.resume)
        parser.add_argument("--train_data_path", type=str, default=default_cfg.train_data_path)
        parser.add_argument("--epochs", type=int, default=default_cfg.epochs)
        parser.add_argument("--episodes_per_update", type=int, default=default_cfg.episodes_per_update)
        parser.add_argument("--lr", type=float, default=default_cfg.lr)
        parser.add_argument("--ucb_beta_max", type=float, default=default_cfg.ucb_beta_max)
        parser.add_argument("--ucb_beta_min", type=float, default=default_cfg.ucb_beta_min)
        parser.add_argument("--l2_coef", type=float, default=default_cfg.l2_coef)
        parser.add_argument("--seed", type=int, default=default_cfg.seed)
        parser.add_argument(
            "--checkpoint_every_epochs",
            type=int,
            default=default_cfg.checkpoint_every_epochs,
            help="N 에폭마다 policy_epoch_{N}.pt 저장. 0이면 에폭 중 저장 안 함.",
        )

        args = parser.parse_args(argv)
        return cls(
            lr=args.lr,
            ucb_beta_max=args.ucb_beta_max,
            ucb_beta_min=args.ucb_beta_min,
            l2_coef=args.l2_coef,
            epochs=args.epochs,
            episodes_per_update=args.episodes_per_update,
            seed=args.seed,
            save_dir=args.save_dir,
            resume=args.resume,
            train_data_path=args.train_data_path,
            checkpoint_every_epochs=args.checkpoint_every_epochs,
        )

