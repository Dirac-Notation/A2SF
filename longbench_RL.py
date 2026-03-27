import os
import json
from tqdm import tqdm
import argparse
import torch

from utils import set_seed
from RL.a2sf_model import A2SFModel, ModelConfig
from longbench_eval import dataset2metric

# ============================================================================
# Prediction Functions (from longbench_pred_RL.py)
# ============================================================================

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="LongBench end-to-end evaluation with RL-trained A2SF model")
    parser.add_argument('--model', type=str, required=True, choices=["llama", "llama2", "llama3", "opt"])
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--datasets', type=str, nargs='*', default=None, help="List of specific dataset names to process. If not specified, all datasets will be processed.")
    parser.add_argument('--rl_checkpoint', type=str, required=True, help="Path to RL model checkpoint (.pt file)")
    return parser.parse_args(args)

def load_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def resolve_selected_datasets(args, dataset2maxlen):
    if args.datasets is not None:
        print(f"Processing specified datasets: {args.datasets}")
        return args.datasets

    return list(dataset2maxlen.keys())

def get_pred(data, max_length, max_gen, dataset, model, tokenizer, out_path, model_name, budget):
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

        input = tokenizer(prompt, truncation=False, return_tensors="pt")
        input_ids = input.input_ids
        context_length = input_ids.shape[-1]
        metric_fn = dataset2metric.get(dataset)
        metric_type = metric_fn.__name__ if metric_fn is not None else "qa_f1_score"

        if dataset == "samsum":
            out = model.generate(
                prompt=prompt,
                metric_type=metric_type,
                token_budget=budget,
                answers=json_obj.get("answers", []),
                all_classes=json_obj.get("all_classes", []),
                dataset=dataset,
                task_type=json_obj.get("task_type"),
                tokenizer=tokenizer,
                stop_strings="[/INST]",
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                min_length=context_length + 1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                num_logits_to_keep=1,
            )
        else:
            out = model.generate(
                prompt=prompt,
                metric_type=metric_type,
                token_budget=budget,
                answers=json_obj.get("answers", []),
                all_classes=json_obj.get("all_classes", []),
                dataset=dataset,
                task_type=json_obj.get("task_type"),
                tokenizer=tokenizer,
                stop_strings="[/INST]",
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                num_logits_to_keep=1,
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
    model = A2SFModel(
        config=model_cfg,
        state_dict=state_dict,
    )
    tokenizer = model.model_runner.tokenizer

    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    if not os.path.exists("result_txt/pred"):
        os.makedirs("result_txt/pred")
    
    output_dir = None
    
    selected_datasets = resolve_selected_datasets(args, dataset2maxlen)
    
    # Process each dataset
    for dataset in selected_datasets:
        print(f"\nProcessing dataset: {dataset}")
        
        jsonl_path = f"datasets/longbench/{dataset}.jsonl"
        if not os.path.exists(jsonl_path):
            print(f"Warning: {jsonl_path} not found, skipping {dataset}")
            continue
        data = load_jsonl_file(jsonl_path)
        
        output_dir = f"result_txt/pred/{args.model}_sigmoid_{args.budget}_RL"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_path = f"{output_dir}/{dataset}.jsonl"
        
        if os.path.exists(out_path):
            os.remove(out_path)
        
        max_gen = dataset2maxlen[dataset]

        get_pred(
            data,
            max_length,
            max_gen,
            dataset,
            model,
            tokenizer,
            out_path,
            model_name,
            args.budget,
        )
        
        print(f"Completed {dataset} with RL agent")
    
    print("\nRL LongBench evaluation completed!")

