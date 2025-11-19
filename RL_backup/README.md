# A2SF RL Training System

This directory contains the reinforcement learning system for training an A2SF model to dynamically adjust KV cache compression ratios based on context.

## Overview

The RL system consists of:

1. **RL Agent**: Neural network that takes recent context (64 tokens) as input and outputs a compression ratio (0.0 to 1.0)
2. **Environment**: A2SF model environment that executes actions and returns rewards
3. **Reward Function**: Based on model accuracy (ROUGE, F1 scores) and efficiency
4. **Training**: PPO algorithm for policy optimization

## Architecture

### Input
- Recent 64 tokens from LLM prompt, encoded using sentence-transformer
- Task type embedding
- History of compression ratios and rewards

### Output
- Single continuous action: compression ratio (0.0 to 1.0)

### Reward
- Primary: Model accuracy (ROUGE, F1 scores)
- Secondary: Efficiency (compression ratio)

## Files

- `config.py`: Configuration class for RL training
- `policy.py`: PPO policy network implementation
- `env.py`: RL environment for A2SF model
- `runner.py`: A2SF model runner with RL integration
- `trainer.py`: Main training loop
- `features.py`: Context encoding and state building
- `buffer.py`: Experience buffer for PPO
- `main.py`: Command-line interface
- `run_training.py`: Training script

## Usage

### Basic Training

```bash
python RL/run_training.py \
    --model llama2 \
    --gpus 0 \
    --iterations 1000 \
    --episodes_per_update 256 \
    --lr 3e-4 \
    --save_dir runs/a2sf_rl
```

### Advanced Training

```bash
python RL/run_training.py \
    --model llama2 \
    --gpus 0 1 2 3 \
    --iterations 2000 \
    --episodes_per_update 512 \
    --lr 1e-4 \
    --accuracy_weight 1.0 \
    --efficiency_weight 0.2 \
    --tasks "Code Complete" "Summarization" \
    --max_samples_per_task 200 \
    --eval_frequency 20 \
    --save_dir runs/a2sf_rl_advanced
```

### Resume Training

```bash
python RL/run_training.py \
    --resume runs/a2sf_rl/policy_500.pt \
    --iterations 1000
```

## Configuration

Key parameters in `config.py`:

- `model_name`: LLM model to use (llama, llama2, llama3, opt)
- `gpus`: List of GPU IDs
- `accuracy_weight`: Weight for accuracy-based reward
- `efficiency_weight`: Weight for efficiency-based reward
- `context_window`: Number of recent tokens to encode (default: 64)
- `episodes_per_update`: Number of episodes per PPO update
- `lr`: Learning rate for policy network

## Monitoring

Training progress is logged to:
- `progress.jsonl`: Training statistics
- `evaluation.jsonl`: Evaluation results
- `policy_*.pt`: Model checkpoints

## Requirements

See `../pip.txt` for required packages. RL training requires:
- torch (already included)
- transformers (already included)
- sentence-transformers>=2.2.0
- tqdm>=4.64.0
- nltk>=3.8.0

## Notes

- The system uses sentence-transformers to encode context tokens
- PPO algorithm with GAE for advantage estimation
- Continuous action space for compression ratio
- Multi-task training on LongBench datasets
- Automatic checkpointing and evaluation
