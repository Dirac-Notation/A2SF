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
    # -- budget 및 하이퍼파라미터 리스트 초기화
    sb_list     = args.select_budget       # select budget
    rb_list     = args.recent_budget       # recent budget
    randb_list  = args.random_budget       # random budget
    sbud_list   = args.streaming_budget    # streaming budget
    ff_list     = args.forgetting_factor   # forgetting factor
    rm_list     = args.random_method       # random method

    # -- 모든 리스트 길이 검사 및 1개짜리 확장
    lengths = [len(sb_list), len(rb_list), len(randb_list), len(sbud_list)]
    max_len = max(lengths)
    for l in lengths:
        if l != 1 and l != max_len:
            raise ValueError("모든 budget 리스트(streaming 포함)의 길이는 같거나, 하나의 원소만 존재해야 합니다.")
    if len(sb_list)    == 1: sb_list    *= max_len
    if len(rb_list)    == 1: rb_list    *= max_len
    if len(randb_list) == 1: randb_list *= max_len
    if len(sbud_list)  == 1: sbud_list  *= max_len

    # -- 디바이스, 모델, 토크나이저 준비
    device    = f"cuda:{args.gpu}"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model     = (LlamaForCausalLM.from_pretrained(args.model_name)
                   .to(torch.bfloat16).to(device))

    # -- 데이터셋 로드
    prompts, answers, output_indices = load_datasets(
        dataset_path=args.datasets,
        tokenizer=tokenizer
    )

    # -- ROUGE 스코어러
    scorer       = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    eos_token_id = tokenizer.eos_token_id

    # -- 결과 저장용 폴더/이름
    dataset_name = os.path.splitext(os.path.basename(args.datasets))[0]
    output_dir   = "output_text"
    os.makedirs(output_dir, exist_ok=True)

    # -- 중첩 루프: forgetting_factor → random_method → 각 budget config
    for ff in ff_list:
        for rm in rm_list:
            for idx in range(max_len):
                cur_sb   = sb_list[idx]
                cur_rb   = rb_list[idx]
                cur_rand = randb_list[idx]
                cur_sbud = sbud_list[idx]

                predictions        = []
                throughput_samples = []

                # === Warm-up (캐시 초기화 연습) ===
                for p in prompts[:10]:
                    model.init_cache(
                        use_compression=True,
                        select_budget=cur_sb,
                        recent_budget=cur_rb,
                        random_budget=cur_rand,
                        streaming_budget=cur_sbud,
                        random_method=rm,
                        forgetting_factor=ff
                    )
                    _ = model.generate(
                        p.to(device),
                        max_new_tokens=output_indices[0].numel(),
                        eos_token_id=eos_token_id,
                        do_sample=False
                    )

                # === Main Test ===
                for i, p in enumerate(tqdm(prompts, desc=(f"sel={cur_sb},rec={cur_rb},ran={cur_rand},str={cur_sbud},ff={ff},rm={rm},cfg{idx+1}/{max_len}"))):
                    torch.cuda.synchronize()
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt   = torch.cuda.Event(enable_timing=True)
                    start_evt.record()

                    model.init_cache(
                        use_compression=True,
                        select_budget=cur_sb,
                        recent_budget=cur_rb,
                        random_budget=cur_rand,
                        streaming_budget=cur_sbud,
                        random_method=rm,
                        forgetting_factor=ff
                    )
                    gen = model.generate(
                        p.to(device),
                        max_new_tokens=output_indices[i].numel(),
                        eos_token_id=eos_token_id,
                        do_sample=False
                    )

                    end_evt.record()
                    torch.cuda.synchronize()
                    elapsed = start_evt.elapsed_time(end_evt) / 1000.0
                    toks    = gen.shape[1] - p.shape[1]
                    throughput_samples.append(toks/elapsed if elapsed>0 else 0)

                    txt = tokenizer.decode(gen[0, p.shape[1]:], skip_special_tokens=True)
                    predictions.append(txt)

                    # 디버깅: 각 레이어의 토큰 인덱스 일부 출력
                    # for i in range(32):
                    #     print(model.model.layers[i].self_attn.past_key_value.token_indices[:, :, cur_randb:].tolist())

                # -- JSONL로 저장
                fname = (
                    f"{dataset_name}"
                    f"_sel{cur_sb}_rec{cur_rb}_ran{cur_rand}"
                    f"_str{cur_sbud}_ff{ff}_rm{rm}.jsonl"
                )
                path = os.path.join(output_dir, fname)
                with open(path, 'w', encoding='utf-8') as fout:
                    for pred in predictions:
                        fout.write(json.dumps({"prediction": pred}, ensure_ascii=False)+"\n")

                # -- 메트릭 계산·출력
                avg_tp = sum(throughput_samples) / len(throughput_samples)
                scores = {'rouge1':[], 'rouge2':[], 'rougeL':[]}
                for ref, pred in zip(answers, predictions):
                    sc = scorer.score(ref, pred)
                    scores['rouge1'].append(sc['rouge1'].fmeasure)
                    scores['rouge2'].append(sc['rouge2'].fmeasure)
                    scores['rougeL'].append(sc['rougeL'].fmeasure)

                avg_r1 = sum(scores['rouge1']) / len(scores['rouge1'])
                avg_r2 = sum(scores['rouge2']) / len(scores['rouge2'])
                avg_rL = sum(scores['rougeL']) / len(scores['rougeL'])

                # -- 결과 출력 (모든 인수 포함)
                print(
                    f"Config {idx+1}/{max_len} | "
                    f"select={cur_sb}, recent={cur_rb}, random={cur_rand}, "
                    f"streaming={cur_sbud}, ff={ff}, rm={rm}"
                )
                print(
                    f"  ROUGE-1: {avg_r1:.4f}, "
                    f"ROUGE-2: {avg_r2:.4f}, "
                    f"ROUGE-L: {avg_rL:.4f}"
                )
                print(f"  Throughput: {avg_tp:.2f} toks/s\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Llama predictions with various budgets.")
    parser.add_argument("--model_name",        type=str,   default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--gpu",               type=int,   default=0)
    parser.add_argument("--datasets",          type=str,   default="fewshot_data/cnn_dailymail-3shot.jsonl")
    parser.add_argument("--select_budget",     type=int,   nargs='+', default=[100])
    parser.add_argument("--recent_budget",     type=int,   nargs='+', default=[100])
    parser.add_argument("--random_budget",     type=int,   nargs='+', default=[0])
    parser.add_argument("--streaming_budget",  type=int,   nargs='+', default=[0])
    parser.add_argument("--forgetting_factor", type=float, nargs='+', default=[1.0])
    parser.add_argument("--random_method",     type=str,   nargs='+', default=["att"])
    args = parser.parse_args()
    main(args)
