import os
import json
from tqdm import tqdm
import argparse
import torch
import re
import string
from typing import List
from collections import Counter

try:
    import jieba
    from fuzzywuzzy import fuzz
    from rouge import Rouge
except ImportError:
    print("Warning: Some optional dependencies not found. Install with: pip install jieba fuzzywuzzy rouge-score")
    jieba = None
    fuzz = None
    Rouge = None

from utils import load_model, set_seed, CompressionConfig

# ============================================================================
# Metrics Functions (from longbench_metrics.py)
# ============================================================================

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""
    def white_space_fix(text):
        return "".join(text.split())
    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_punc(lower(s)))

def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def code_sim_score(prediction, ground_truth, **kwargs):
    if fuzz is None:
        return 0.0
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return (fuzz.ratio(prediction, ground_truth) / 100)

def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = (1.0 / len(em_match_list))
    else:
        score = 0.0
    return score

def rouge_score(prediction, ground_truth, **kwargs):
    if Rouge is None:
        return 0.0
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

def rouge_zh_score(prediction, ground_truth, **kwargs):
    if jieba is None or Rouge is None:
        return 0.0
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False))) 
    score = rouge_score(prediction, ground_truth)
    return score

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    if jieba is None:
        return 0.0
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)

# ============================================================================
# Evaluation Functions (from longbench_eval.py)
# ============================================================================

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

data_group = {
    "Code Complete": ["repobench-p", "lcc"],
    "Few Shot": ["trec", "triviaqa", "samsum", "lsht"],
    "Single-doc QA": ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"],
    "Multi-doc QA": ["hotpotqa", "2wikimqa", "musique", "dureader"],
    "Summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "Passage Retrieval": ["passage_retrieval_en", "passage_retrieval_zh", "passage_count"],
}

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

def calculate_group_averages(scores):
    group_scores = {}
    for group_name, datasets in data_group.items():
        group_scores[group_name] = {}
        group_datasets = []
        for dataset in datasets:
            if dataset in scores:
                group_datasets.append(scores[dataset])
        if group_datasets:
            group_scores[group_name] = round(sum(group_datasets) / len(group_datasets), 2)
    return group_scores

def evaluate_results(output_dir):
    """Evaluate predictions in the output directory"""
    print(f"\n{'='*60}")
    print("Evaluating results...")
    print(f"{'='*60}")
    
    scores = dict()
    all_files = os.listdir(output_dir)
    print("Evaluating on:", all_files)
    
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers = [], []
        dataset = filename.split('.')[0]
        with open(f"{output_dir}/{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
        
        score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    
    # Calculate group averages
    group_scores = calculate_group_averages(scores)
    
    # Calculate overall average
    all_scores = list(scores.values())
    overall_avg = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0
    
    # Combine individual scores and group scores
    final_results = {
        "individual_scores": scores,
        "group_averages": group_scores,
        "overall_average": overall_avg
    }
    
    # Save results to JSON
    out_path = f"{output_dir}/result.json"
    with open(out_path, "w") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    # Print summary
    print("\nResults Summary:")
    print("-" * 50)
    print("Individual Dataset Scores:")
    for dataset, score in scores.items():
        print(f"{dataset:20}: {score:.2f}")
    
    print("\nGroup Averages:")
    for group_name, avg_score in group_scores.items():
        print(f"{group_name:20}: {avg_score:.2f}")
    print(f"\n{'Overall Average':20}: {overall_avg:.2f}")
    
    print(f"\nResults have been saved to: {out_path}")
    return final_results

# ============================================================================
# Prediction Functions (from longbench_pred.py)
# ============================================================================

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="LongBench end-to-end evaluation")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help="List of GPU IDs (e.g., --gpus 0 1 2 3)")
    parser.add_argument('--model', type=str, required=True, choices=["llama", "llama2", "llama3", "opt"])
    parser.add_argument('--method', type=str, default="full")
    parser.add_argument('--window', type=int, default=16)
    parser.add_argument('--budget', type=int, default=128)
    parser.add_argument('--task', type=int, nargs='*', default=None, help="List of task numbers (0-5). If not specified, all tasks will be executed.")
    parser.add_argument('--skip_eval', action='store_true', help="Skip evaluation after prediction")
    return parser.parse_args(args)

def load_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def get_pred(data, max_length, max_gen, dataset, model, tokenizer, out_path, model_name, config):
    for json_obj in tqdm(data):
        prompt = json_obj["input_prompt"]
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            if "llama" in model_name:
                prompt = f"[INST]{prompt}[/INST]"
        
        input = tokenizer(prompt, truncation=False, return_tensors="pt")
        
        input_ids = input.input_ids.to(model.device)
        attention_mask = input.attention_mask.to(torch.bfloat16).to(model.device)
        
        context_length = input_ids.shape[-1]
        
        model.init_cache(config)
        
        with torch.inference_mode():
            if dataset == "samsum":
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                    pad_token_id=tokenizer.eos_token_id,
                )[0]
            else:
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

if __name__ == '__main__':
    set_seed(42)
    args = parse_args()
    
    gpus = args.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    
    model_name = args.model
    model_name = model_name.split("_")[0].lower()
    max_length = json.load(open("config/model2maxlen.json", "r"))[model_name]
    
    model, tokenizer = load_model(model_name, args.gpus)

    # Task 이름 리스트
    task_list = [
        "Code Complete",
        "Few Shot",
        "Single-doc QA",
        "Multi-doc QA",
        "Passage Retrieval",
        "Summarization",
    ]
    
    # Task 번호를 task 이름으로 변환
    if args.task is None:
        selected_tasks = task_list
    else:
        selected_tasks = []
        for task_num in args.task:
            if 0 <= task_num < len(task_list):
                selected_tasks.append(task_list[task_num])
            else:
                print(f"Warning: Task number {task_num} is out of range (0-{len(task_list)-1}), skipping")
    
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    if not os.path.exists("result_txt/pred"):
        os.makedirs("result_txt/pred")
    
    output_dir = None
    
    for task in selected_tasks:
        config = CompressionConfig()
        config["compression_method"] = args.method
        config["observation_window"] = args.window
        config["total_budget"] = args.budget
        config["a"] = 10
        config["b"] = args.window
        
        datasets = data_group[task]
        
        for dataset in datasets:
            print(f"\nProcessing dataset: {dataset}")
            
            jsonl_path = f"datasets/longbench/{dataset}.jsonl"
            if not os.path.exists(jsonl_path):
                print(f"Warning: {jsonl_path} not found, skipping {dataset}")
                continue
            data = load_jsonl_file(jsonl_path)
            output_dir = f"result_txt/pred/{args.model}_{args.method}_{args.window}_{args.budget}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            out_path = f"{output_dir}/{dataset}.jsonl"
            
            # Clear existing file
            if os.path.exists(out_path):
                os.remove(out_path)
            
            max_gen = dataset2maxlen[dataset]
            
            get_pred(data, max_length, max_gen, dataset, model, tokenizer, out_path, model_name, config)
    
    # Evaluate results if not skipped
    if not args.skip_eval and output_dir and os.path.exists(output_dir):
        evaluate_results(output_dir)
    
    print("\nLongBench evaluation completed!")

