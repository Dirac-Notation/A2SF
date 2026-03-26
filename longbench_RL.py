import os
import json
from tqdm import tqdm
import argparse
import torch
import sys

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed
from RL.a2sf_model import A2SFModel, ModelConfig
from longbench_eval import dataset2metric, evaluate_results

# ============================================================================
# Prediction Functions (from longbench_pred_RL.py)
# ============================================================================

TASK2DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "task2dataset.json")
with open(TASK2DATASET_PATH, "r", encoding="utf-8") as f:
    TASK_TO_DATASETS = json.load(f)
TASK_LIST = list(TASK_TO_DATASETS.keys())

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="LongBench end-to-end evaluation with RL-trained A2SF model")
    parser.add_argument('--model', type=str, required=True, choices=["llama", "llama2", "llama3", "opt"])
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--task', type=int, nargs='*', default=None, help="List of task numbers (0-5). If not specified, all tasks will be executed.")
    parser.add_argument('--datasets', type=str, nargs='*', default=None, help="List of specific dataset names to process. If specified, only these datasets will be processed (ignoring --task).")
    parser.add_argument('--rl_checkpoint', type=str, required=True, help="Path to RL model checkpoint (.pt file)")
    parser.add_argument('--skip_eval', action='store_true', help="Skip evaluation after prediction")
    return parser.parse_args(args)

def load_jsonl_file(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def resolve_selected_datasets(args):
    if args.datasets is not None:
        print(f"Processing specified datasets: {args.datasets}")
        return args.datasets

    if args.task is None:
        selected_tasks = TASK_LIST
    else:
        selected_tasks = []
        for task_num in args.task:
            if 0 <= task_num < len(TASK_LIST):
                selected_tasks.append(TASK_LIST[task_num])
            else:
                print(f"Warning: Task number {task_num} is out of range (0-{len(TASK_LIST)-1}), skipping")

    selected_datasets = []
    for task in selected_tasks:
        selected_datasets.extend(TASK_TO_DATASETS[task])
    return selected_datasets

def get_pred_rl(
    data,
    max_length,
    max_gen,
    dataset,
    out_path,
    model_name,
    a2sf_model: A2SFModel,
    budget: int,
):
    """Generate predictions using A2SFModel.generate() (RL action + KV compression)."""
    tokenizer = a2sf_model.model_runner.tokenizer

    for json_obj in tqdm(data):
        prompt = json_obj["input_prompt"]

        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                tokenized_prompt[-half:],
                skip_special_tokens=True,
            )

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            if "llama" in model_name:
                prompt = f"[INST]{prompt}[/INST]"

        out = a2sf_model.generate(
            prompt=prompt,
            task=dataset,
            max_new_tokens=max_gen,
            token_budget=budget,
            answers=json_obj.get("answers", []),
            all_classes=json_obj.get("all_classes", []),
            task_type=json_obj.get("task_type"),
        )

        a_val = out.info.get("a")
        b_val = out.info.get("b")
        # Historically we wrote `b` as an int.
        b_write = int(round(b_val)) if isinstance(b_val, (float, int)) else b_val

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": out.pred_text,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"],
                    "a": a_val,
                    "b": b_write,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

if __name__ == '__main__':
    set_seed(42)
    args = parse_args()
    
    model_name = args.model
    model_name = model_name.split("_")[0].lower()
    max_length = json.load(open("config/model2maxlen.json", "r"))[model_name]

    checkpoint = torch.load(args.rl_checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("agent_state_dict")
    if state_dict is None:
        # Backward compatibility for older checkpoints.
        state_dict = checkpoint.get("policy_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint missing 'agent_state_dict' (and legacy 'policy_state_dict').")

    # Model config (KV 모델 로딩 정보)는 `config`에서 가져오고,
    # agent 생성/weight 로딩은 checkpoint에서 추출합니다.
    model_cfg = ModelConfig(model=model_name)
    a2sf_model = A2SFModel(
        config=model_cfg,
        state_dict=state_dict,
    )

    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    if not os.path.exists("result_txt/pred"):
        os.makedirs("result_txt/pred")
    
    output_dir = None
    
    selected_datasets = resolve_selected_datasets(args)
    
    # Process each dataset
    for dataset in selected_datasets:
        print(f"\nProcessing dataset: {dataset}")
        
        jsonl_path = f"datasets/longbench/{dataset}.jsonl"
        if not os.path.exists(jsonl_path):
            print(f"Warning: {jsonl_path} not found, skipping {dataset}")
            continue
        data = load_jsonl_file(jsonl_path)
        
        # Create output directory with RL indicator
        output_dir = f"result_txt/pred/{args.model}_sigmoid_{args.budget}_RL"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_path = f"{output_dir}/{dataset}.jsonl"
        
        # Clear existing file
        if os.path.exists(out_path):
            os.remove(out_path)
        
        max_gen = dataset2maxlen[dataset]

        get_pred_rl(
            data,
            max_length,
            max_gen,
            dataset,
            out_path,
            model_name,
            a2sf_model,
            args.budget,
        )
        
        print(f"Completed {dataset} with RL agent")
    
    # Evaluate results if not skipped
    if not args.skip_eval and output_dir and os.path.exists(output_dir):
        evaluate_results(output_dir)
    
    print("\nRL LongBench evaluation completed!")

