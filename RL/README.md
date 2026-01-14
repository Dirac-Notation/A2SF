# A2SF RL Training System

This directory contains the reinforcement learning system for training an A2SF model to dynamically adjust KV cache compression ratios based on context.

## Overview

The RL system consists of:

1. **RL Agent**: Neural network that takes context (encoded with m2-bert-80M-8k) as input and outputs compression parameters (a, b)
2. **Environment**: A2SF model environment that executes actions and returns rewards
3. **Reward Function**: Based on similarity between full cache and compressed cache generated texts
4. **Training**: NeuralUCB algorithm for policy optimization

## Architecture

### Input
- Full prompt text, encoded using m2-bert-80M-8k CLS token
- Fixed-length embedding vector (768 dimensions)

### Output
- Discrete action: tuple of (a, b) parameters for sigmoid cache compression

### Reward
- Cosine similarity between full cache and compressed cache generated texts

## Files

- `config.py`: Configuration class for RL training
- `policy.py`: NeuralUCB policy network implementation
- `env.py`: RL environment for A2SF model
- `runner.py`: A2SF model runner with RL integration
- `trainer.py`: Main training loop
- `features.py`: Context encoding and state building
- `buffer.py`: Experience buffer for NeuralUCB
- `main.py`: Command-line interface and training script

## Usage

### Basic Training

```bash
python -m RL.main \
    --model_name llama2 \
    --gpus 0 \
    --iterations 1000 \
    --episodes_per_update 256 \
    --lr 3e-4 \
    --save_dir runs/a2sf_rl
```

### Advanced Training

```bash
python -m RL.main \
    --model_name llama2 \
    --gpus 0 1 2 3 \
    --iterations 2000 \
    --episodes_per_update 512 \
    --lr 1e-4 \
    --ucb_beta 1.0 \
    --uncertainty_coef 0.1 \
    --eval_frequency 20 \
    --save_dir runs/a2sf_rl_advanced
```

### Resume Training

```bash
python -m RL.main \
    --resume runs/a2sf_rl/policy_500.pt \
    --iterations 1000
```

## Configuration

Key parameters in `config.py`:

- `model_name`: LLM model to use (llama, llama2, llama3, opt)
- `gpus`: List of GPU IDs
- `context_encoder_model`: Context encoder model (default: togethercomputer/m2-bert-80M-8k)
- `ucb_beta`: UCB exploration parameter (default: 1.0)
- `uncertainty_coef`: Uncertainty regularization coefficient (default: 0.1)
- `episodes_per_update`: Number of episodes per update
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

- The system uses m2-bert-80M-8k to encode full prompt text (CLS token)
- NeuralUCB algorithm for bandit-style learning
- Discrete action space for (a, b) compression parameters
- Multi-task training on LongBench datasets
- Automatic checkpointing and evaluation
