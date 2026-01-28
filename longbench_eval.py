import os
import json
import re
import string
import argparse
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

# ============================================================================
# Metrics Functions
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
# Evaluation Functions
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
# Main function for standalone execution
# ============================================================================

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Evaluate LongBench predictions")
    parser.add_argument('output_dir', type=str, help="Path to directory containing prediction JSONL files (e.g., result_txt/backup/llama3_sigmoid_128_RL)")
    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    
    # Check if output directory exists
    if not os.path.exists(args.output_dir):
        print(f"Error: Directory '{args.output_dir}' does not exist")
        exit(1)
    
    # Evaluate results
    evaluate_results(args.output_dir)

