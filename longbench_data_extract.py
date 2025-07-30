import os
from datasets import load_dataset
import json
from tqdm import tqdm
import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

class DummyModel:
    """Dummy model that returns predefined responses for testing"""
    def __init__(self):
        self.responses = {
            "narrativeqa": "This is a dummy answer for narrative question answering.",
            "qasper": "This is a dummy answer for scientific article question answering.",
            "multifieldqa_en": "This is a dummy answer for English multi-field question answering.",
            "multifieldqa_zh": "这是中文多领域问答的虚拟答案。",
            "hotpotqa": "This is a dummy answer for hotpot question answering.",
            "2wikimqa": "This is a dummy answer for 2wiki multi-hop question answering.",
            "musique": "This is a dummy answer for musique question answering.",
            "dureader": "这是中文阅读理解任务的虚拟答案。",
            "gov_report": "This is a dummy summary for government report summarization.",
            "qmsum": "This is a dummy answer for meeting summarization.",
            "multi_news": "This is a dummy summary for multi-news summarization.",
            "vcsum": "这是会议总结的虚拟答案。",
            "trec": "This is a dummy classification for TREC question classification.",
            "triviaqa": "This is a dummy answer for trivia question answering.",
            "samsum": "This is a dummy summary for dialogue summarization.",
            "lsht": "这是新闻分类的虚拟答案。",
            "passage_count": "5",
            "passage_retrieval_en": "Paragraph 3",
            "passage_retrieval_zh": "段落3",
            "lcc": "def dummy_function():\n    return 'dummy'",
            "repobench-p": "def dummy_function():\n    return 'dummy'"
        }
    
    def generate(self, prompt, dataset_name):
        """Generate dummy response based on dataset type"""
        return self.responses.get(dataset_name, "This is a dummy response.")

def build_chat(prompt, model_name="dummy"):
    """Build chat format prompt if needed"""
    if "llama" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    return prompt

def extract_data(data, prompt_format, dataset_name, dummy_model, out_path):
    """Extract input prompts and generate dummy outputs"""
    for json_obj in tqdm(data, desc=f"Processing {dataset_name}"):
        # Format the prompt using the dataset's prompt format
        prompt = prompt_format.format(**json_obj)
        
        # Build chat format if needed (for most datasets except specific ones)
        if dataset_name not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(prompt, "dummy")
        
        # Generate dummy output
        dummy_output = dummy_model.generate(prompt, dataset_name)
        
        # Save to file
        result = {
            "input_prompt": prompt,
            "output": dummy_output,
            "answers": json_obj.get("answers", []),
            "all_classes": json_obj.get("all_classes", []),
            "length": json_obj.get("length", 0),
            "dataset": dataset_name
        }
        
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

def main():
    # Set random seed for reproducibility
    seed_everything(42)
    
    # Initialize dummy model
    dummy_model = DummyModel()
    
    # Define datasets (same as in original code)
    datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    
    # Load configurations
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    
    # Create output directory
    output_dir = "result_txt/longbench"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each dataset
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        
        try:
            # Load dataset
            data = load_dataset('THUDM/LongBench', dataset, split='test', trust_remote_code=True)
            
            # Set output path
            out_path = f"{output_dir}/{dataset}.jsonl"
            
            # Get prompt format and max length for this dataset
            prompt_format = dataset2prompt[dataset]
            max_gen = dataset2maxlen[dataset]
            
            print(f"Dataset size: {len(data)}")
            print(f"Max generation length: {max_gen}")
            
            # Extract data using dummy model
            extract_data(data, prompt_format, dataset, dummy_model, out_path)
            
            print(f"Results saved to: {out_path}")
            
        except Exception as e:
            print(f"Error processing dataset {dataset}: {e}")
            continue

if __name__ == '__main__':
    main()