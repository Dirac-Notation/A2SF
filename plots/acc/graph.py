import json
import matplotlib.pyplot as plt

data_group = {
    "Code Complete": ["repobench-p", "lcc"],
    "Few Shot": ["trec", "triviaqa", "samsum"],
    "Single-doc QA": ["narrativeqa", "qasper", "multifieldqa_en"],
    "Multi-doc QA": ["hotpotqa", "2wikimqa", "musique"],
    "Summarization": ["gov_report", "qmsum", "multi_news"],
    "Passage Retrieval": ["passage_retrieval_en", "passage_count"],
}

X = [1, 16, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
method = ["snap1", "snap16", "snap512", "snap1024", "snap2048", "snap3072", "snap4096", "snap5120", "snap6144", "snap7168", "h2o"]

scores = {}

for m in method:
    with open(f"result_txt/pred/llama3_{m}_128/result.json", "r") as f:
        data = json.load(f)
    
    if scores == {}:
        for dataset, score in data["individual_scores"].items():
            scores[dataset] = [score]
    else:
        for dataset, score in data["individual_scores"].items():
            scores[dataset].append(score)

for task, datasets in data_group.items():
    plt.figure(figsize=(12, 7))
    for dataset in datasets:
        plt.plot(range(len(X)), scores[dataset], label=dataset)
    plt.xticks(range(len(X)), X)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"plots/acc/graph_{task}.png")
    plt.close()