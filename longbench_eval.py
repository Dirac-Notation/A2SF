import os
import json
import argparse
import numpy as np

from longbench_metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

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

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--method', type=str, nargs='+', default=["a2sf"])
    parser.add_argument('--budget', type=int, default=100)
    return parser.parse_args(args)

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
        
        # Calculate average for the group
        if group_datasets:
            group_scores[group_name] = round(sum(group_datasets) / len(group_datasets), 2)
    
    return group_scores

if __name__ == '__main__':
    args = parse_args()
    
    for method in args.method:
        print(f"\nEvaluating method: {method}")
        scores = dict()
        path = f"result_txt/pred/{args.model}_{method}_{args.budget}/"
        
        all_files = os.listdir(path)
        print("Evaluating on:", all_files)
        
        for filename in all_files:
            if not filename.endswith("jsonl"):
                continue
            predictions, answers = [], []
            dataset = filename.split('.')[0]
            with open(f"{path}{filename}", "r", encoding="utf-8") as f:
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
        out_path = f"{path}result.json"
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
            if group_name == "overall average":
                print("-" * 50)
            print(f"{group_name:20}: {avg_score:.2f}")
        
        print(f"\nResults have been saved to: {out_path}")
