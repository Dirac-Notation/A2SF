from __future__ import annotations

import argparse

from RL.a2sf_model import ModelConfig
from .training_config import TrainingConfig
from .trainer import A2SFTrainer


def build_model_config_from_args(model_name: str) -> ModelConfig:
    return ModelConfig(model=model_name)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train A2SF RL Agent")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["llama3-8b", "llama3-1b", "qwen2"],
        help="Base model type (used by utils.load_model())",
    )
    parser.add_argument("--epochs", type=int, default=None, help="(override) number of epochs")
    # TrainingConfig parses the rest.
    # We parse model separately so training_config doesn't need to know about it.
    args, remaining = parser.parse_known_args(argv)

    # TrainingConfig uses sys.argv by default; pass remaining explicitly.
    training_cfg = TrainingConfig.from_args(remaining)
    if args.epochs is not None:
        training_cfg.epochs = args.epochs

    model_cfg = build_model_config_from_args(args.model)

    trainer = A2SFTrainer(model_cfg, training_cfg)
    start_iteration = 0
    if training_cfg.resume:
        start_iteration = trainer.load_checkpoint(training_cfg.resume)
        print(f"Resuming training from iteration {start_iteration}")

    try:
        final_iteration = trainer.train(num_epochs=training_cfg.epochs)
        done_msg = "Training completed."
    except KeyboardInterrupt:
        final_iteration = int(getattr(trainer, "last_iteration", 0))
        done_msg = f"Training interrupted by user at iteration {final_iteration}."
        print(f"\n[run] {done_msg} Saving final checkpoint...")

    final_checkpoint_path = trainer.save_final_checkpoint(final_iteration)
    print(f"{done_msg} Final model saved to: {final_checkpoint_path}")


if __name__ == "__main__":
    main()

