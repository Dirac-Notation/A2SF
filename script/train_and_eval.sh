#!/bin/bash

set -e

usage() {
    echo "Usage: bash script/train_and_eval.sh [OPTIONS]"
    echo ""
    echo "  학습(GPU 1개) → 추론(전체 GPU) → 평가를 한 번에 실행합니다."
    echo ""
    echo "Options:"
    echo "  -b, --budget              Token budget (default: 512)"
    echo "  -m, --model               Model name (default: llama3-1b)"
    echo "  -g, --train_gpu           Training GPU ID (default: 0)"
    echo "  -d, --train_data_path     Training data JSONL (default: RL/training/data/training_data.jsonl)"
    echo "  -e, --episodes_per_update Episodes per update (default: 16)"
    echo "  -p, --gpus_per_model      GPUs per model for inference (default: 1)"
    echo "  -s, --save_dir            Save directory (default: runs/<model>_<budget>)"
    echo "  -h, --help                Show this help message"
    exit 0
}

budget=512
model="llama3-1b"
train_gpu=0
train_data_path="RL/training/data/training_data.jsonl"
episodes_per_update=16
gpus_per_model=1
save_dir=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        -b|--budget) budget="$2"; shift 2 ;;
        -m|--model) model="$2"; shift 2 ;;
        -g|--train_gpu) train_gpu="$2"; shift 2 ;;
        -d|--train_data_path) train_data_path="$2"; shift 2 ;;
        -e|--episodes_per_update) episodes_per_update="$2"; shift 2 ;;
        -p|--gpus_per_model) gpus_per_model="$2"; shift 2 ;;
        -s|--save_dir) save_dir="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# save_dir 기본값: runs/<model>_<budget>
if [[ -z "$save_dir" ]]; then
    save_dir="runs/${model}_${budget}"
fi

rl_checkpoint="${save_dir}/policy_final.pt"

echo "============================================"
echo "  A2SF RL: Train → Predict → Evaluate"
echo "============================================"
echo "Model:              $model"
echo "Budget:             $budget"
echo "Train GPU:          $train_gpu"
echo "Train data:         $train_data_path"
echo "Episodes/update:    $episodes_per_update"
echo "Save dir:           $save_dir"
echo "Checkpoint:         $rl_checkpoint"
echo "Inference GPUs/model: $gpus_per_model"
echo "============================================"
echo ""

# ---- 1. Training (GPU 1개) ----
echo "[1/3] Training..."
CUDA_VISIBLE_DEVICES=$train_gpu python -m RL.training.run \
    --model "$model" \
    --save_dir "$save_dir" \
    --token_budget "$budget" \
    --train_data_path "$train_data_path" \
    --episodes_per_update "$episodes_per_update"

echo ""
echo "[1/3] Training completed. Checkpoint: $rl_checkpoint"
echo ""

# ---- 2. Prediction (전체 GPU) ----
output_dir="${save_dir}/${model}_sigmoid_${budget}_RL"
echo "[2/3] Running LongBench RL prediction (all GPUs)..."
echo "  Output dir: $output_dir"
python longbench_RL.py \
    --model "$model" \
    --budget "$budget" \
    --rl_checkpoint "$rl_checkpoint" \
    --gpus_per_model "$gpus_per_model" \
    --output_dir "$output_dir"

echo ""
echo "[2/3] Prediction completed."
echo ""

# ---- 3. Evaluation ----
echo "[3/3] Running evaluation on $output_dir ..."
python longbench_eval.py "$output_dir"

echo ""
echo "============================================"
echo "  All done!"
echo "  Results: $output_dir/result.json"
echo "============================================"
