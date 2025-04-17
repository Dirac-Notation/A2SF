import os
import torch
import json
import argparse
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from tqdm import tqdm

from utils import load_datasets
from utils_real_drop.kv_llama import LlamaForCausalLM

def main(args):
    # budget 리스트
    sb_list = args.select_budget
    rb_list = args.recent_budget
    randb_list = args.random_budget
    # 새로 추가된 하이퍼파라미터 리스트
    ff_list = args.forgetting_factor
    rm_list = args.random_method

    # budget 리스트 길이 검사 및 확장
    lengths = [len(sb_list), len(rb_list), len(randb_list)]
    max_len = max(lengths)
    for l in lengths:
        if l != 1 and l != max_len:
            raise ValueError("모든 budget 리스트의 길이는 같거나, 하나의 원소만 존재해야 합니다.")
    if len(sb_list) == 1:
        sb_list *= max_len
    if len(rb_list) == 1:
        rb_list *= max_len
    if len(randb_list) == 1:
        randb_list *= max_len

    device = f"cuda:{args.gpu}"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = LlamaForCausalLM.from_pretrained(args.model_name).to(torch.bfloat16).to(device)
    prompts, answers, output_indices = load_datasets(dataset_path=args.datasets, tokenizer=tokenizer)
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    eos_token_id = tokenizer.eos_token_id

    dataset_name = os.path.splitext(os.path.basename(args.datasets))[0]
    output_dir = "output_text"
    os.makedirs(output_dir, exist_ok=True)

    # forgetting_factor, random_method → budgets 순서로 중첩 루프
    for ff in ff_list:
        for rm in rm_list:
            for config_idx in range(max_len):
                cur_sb = sb_list[config_idx]
                cur_rb = rb_list[config_idx]
                cur_randb = randb_list[config_idx]
                predictions = []
                throughput_samples = []

                for idx, prompt in enumerate(tqdm(prompts,
                                                  desc=f"FF={ff}, RM={rm} Config {config_idx+1}/{max_len}")):
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end   = torch.cuda.Event(enable_timing=True)
                    start.record()

                    model.init_cache(
                        use_compression=True,
                        select_budget=cur_sb,
                        recent_budget=cur_rb,
                        random_budget=cur_randb,
                        random_method=rm,
                        forgetting_factor=ff
                    )
                    input_ids = prompt.to(device)
                    gen_ids = model.generate(
                        input_ids,
                        max_new_tokens=output_indices[idx].numel(),
                        eos_token_id=eos_token_id,
                        do_sample=False
                    )

                    end.record()
                    torch.cuda.synchronize()
                    elapsed = start.elapsed_time(end) / 1000.0
                    toks = gen_ids.shape[1] - input_ids.shape[1]
                    throughput_samples.append(toks/elapsed if elapsed>0 else 0)

                    pred = tokenizer.decode(
                        gen_ids[0, input_ids.shape[1]:],
                        skip_special_tokens=True
                    )
                    predictions.append(pred)

                # 결과 저장
                fname = (
                    f"{dataset_name}"
                    f"_select{cur_sb}"
                    f"_recent{cur_rb}"
                    f"_random{cur_randb}"
                    f"_ff{ff}"
                    f"_rm{rm}.jsonl"
                )
                path = os.path.join(output_dir, fname)
                with open(path, 'w', encoding='utf-8') as fout:
                    for p in predictions:
                        fout.write(json.dumps({"prediction": p}, ensure_ascii=False) + "\n")

                # 메트릭 계산
                avg_tp = sum(throughput_samples) / len(throughput_samples)
                scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
                for ref, pred in zip(answers, predictions):
                    s = scorer.score(ref, pred)
                    scores['rouge1'].append(s['rouge1'].fmeasure)
                    scores['rouge2'].append(s['rouge2'].fmeasure)
                    scores['rougeL'].append(s['rougeL'].fmeasure)

                print(f"Model: {args.model_name}, Dataset: {dataset_name}, FF: {ff}, RM: {rm}")
                print(f" Config {config_idx+1}/{max_len} – select:{cur_sb}, recent:{cur_rb}, random:{cur_randb}")
                print(f"  ROUGE-1: {sum(scores['rouge1'])/len(scores['rouge1']):.4f}")
                print(f"  ROUGE-2: {sum(scores['rouge2'])/len(scores['rouge2']):.4f}")
                print(f"  ROUGE-L: {sum(scores['rougeL'])/len(scores['rougeL']):.4f}")
                print(f"  Throughput: {avg_tp:.2f} toks/s\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Llama model predictions using ROUGE scores.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--datasets", type=str, default="fewshot_data/cnn_dailymail-3shot.jsonl")
    parser.add_argument("--select_budget", type=int,   nargs='+', default=[100])
    parser.add_argument("--recent_budget", type=int,   nargs='+', default=[100])
    parser.add_argument("--random_budget", type=int,   nargs='+', default=[0])
    parser.add_argument("--forgetting_factor", type=float, nargs='+', default=[1.0])
    parser.add_argument("--random_method",     type=str,   nargs='+', default=["att"])
    args = parser.parse_args()
    main(args)
