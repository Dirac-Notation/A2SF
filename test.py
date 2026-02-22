"""
세 가지 캐시 방식의 결과 비교 테스트:
1. A2SF Cache (forgetting_factor=1.0)
2. Snap Cache (observation_window=prompt_length)
3. Sigmoid Cache (a=10.0, b=prompt_length)

동일한 프롬프트에 대해 각 캐시의 생성 결과를 비교합니다.
"""

import torch
import json
import os

from utils import load_model, set_seed, CompressionConfig

PROMPT = (
    "Answer the following question based on the given context.\n\n"
    "Context: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
    "It is named after the engineer Gustave Eiffel, whose company designed and built the tower. "
    "Locally nicknamed 'La dame de fer', it was constructed from 1887 to 1889 as the centerpiece "
    "of the 1889 World's Fair. The tower is 330 metres tall, about the same height as an 81-storey building. "
    "The tower has three levels for visitors, with restaurants on the first and second levels. "
    "The top level's upper platform is 276 m above the ground. Tickets can be purchased to ascend by stairs "
    "or lift to the first and second levels. The climb from ground level to the first level is over 300 steps, "
    "as is the climb from the first level to the second. Although there is a staircase to the top level, "
    "it is usually accessible only by lift.\n\n"
    "Question: How tall is the Eiffel Tower?\n\nAnswer:"
)

MODEL_NAME = "llama3"
TOTAL_BUDGET = 128
MAX_NEW_TOKENS = 64
SEED = 42


def run_generation(model, tokenizer, input_ids, attention_mask, config, max_new_tokens, seed):
    """주어진 config로 모델 생성을 수행하고 결과를 반환"""
    model.init_cache(config)
    set_seed(seed)
    
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids.clone(),
            attention_mask=attention_mask.clone(),
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )[0]
    
    return output


def main():
    set_seed(SEED)
    
    # =========================================================================
    # 1. 모델 및 토크나이저 로드
    # =========================================================================
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load_model(MODEL_NAME)
    num_layers = model.config.num_hidden_layers
    print(f"  - Number of layers: {num_layers}")
    
    # =========================================================================
    # 2. 프롬프트 준비
    # =========================================================================
    formatted_prompt = f"[INST]{PROMPT}[/INST]"
    
    input_enc = tokenizer(formatted_prompt, truncation=False, return_tensors="pt")
    input_ids = input_enc.input_ids.to(model.device)
    attention_mask = input_enc.attention_mask.to(torch.bfloat16).to(model.device)
    
    prompt_length = input_ids.shape[1]
    print(f"  - Prompt length (tokens): {prompt_length}")
    print(f"  - Total budget: {TOTAL_BUDGET}")
    print(f"  - Max new tokens: {MAX_NEW_TOKENS}")
    
    # =========================================================================
    # 3. 각 캐시 방식으로 생성 수행
    # =========================================================================
    results = {}
    output_ids = {}
    
    # ----- (1) A2SF Cache: forgetting_factor=0.0 -----
    print("\n" + "=" * 70)
    print("[1/3] A2SF Cache (forgetting_factor=1.0)")
    print("=" * 70)
    
    config_a2sf = CompressionConfig()
    config_a2sf["compression_method"] = "a2sf"
    config_a2sf["total_budget"] = TOTAL_BUDGET
    config_a2sf["forgetting_factor"] = 0.75
    config_a2sf["layerwise_ratios"] = [1.0] * num_layers
    config_a2sf["local_ratios"] = 0.125
    
    output = run_generation(model, tokenizer, input_ids, attention_mask, config_a2sf, MAX_NEW_TOKENS, SEED)
    pred_a2sf = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
    results["a2sf (ff=1.0)"] = pred_a2sf
    output_ids["a2sf (ff=1.0)"] = output[prompt_length:].tolist()
    print(f"  Generated: {pred_a2sf[:200]}{'...' if len(pred_a2sf) > 200 else ''}")
    
    # ----- (2) Snap Cache: observation_window=prompt_length -----
    print("\n" + "=" * 70)
    print(f"[2/3] Snap Cache (observation_window={prompt_length})")
    print("=" * 70)
    
    config_snap = CompressionConfig()
    config_snap["compression_method"] = "snap"
    config_snap["total_budget"] = TOTAL_BUDGET
    config_snap["observation_window"] = 16
    
    output = run_generation(model, tokenizer, input_ids, attention_mask, config_snap, MAX_NEW_TOKENS, SEED)
    pred_snap = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
    results["snap (ow=prompt_len)"] = pred_snap
    output_ids["snap (ow=prompt_len)"] = output[prompt_length:].tolist()
    print(f"  Generated: {pred_snap[:200]}{'...' if len(pred_snap) > 200 else ''}")
    
    # ----- (3) Sigmoid Cache: a=10.0, b=prompt_length -----
    print("\n" + "=" * 70)
    print(f"[3/3] Sigmoid Cache (a=10.0, b={prompt_length})")
    print("=" * 70)
    
    config_sigmoid = CompressionConfig()
    config_sigmoid["compression_method"] = "sigmoid"
    config_sigmoid["total_budget"] = TOTAL_BUDGET
    config_sigmoid["a"] = 10.0
    config_sigmoid["b"] = 16
    
    output = run_generation(model, tokenizer, input_ids, attention_mask, config_sigmoid, MAX_NEW_TOKENS, SEED)
    pred_sigmoid = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
    results["sigmoid (a=10, b=prompt_len)"] = pred_sigmoid
    output_ids["sigmoid (a=10, b=prompt_len)"] = output[prompt_length:].tolist()
    print(f"  Generated: {pred_sigmoid[:200]}{'...' if len(pred_sigmoid) > 200 else ''}")
    
    # =========================================================================
    # 4. 결과 비교
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    method_names = list(results.keys())
    
    # 전체 텍스트 출력
    for name in method_names:
        print(f"\n--- {name} ---")
        print(results[name])
    
    # 토큰 레벨 비교
    print("\n" + "-" * 70)
    print("TOKEN-LEVEL COMPARISON")
    print("-" * 70)
    
    # 각 쌍 비교
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            name_i = method_names[i]
            name_j = method_names[j]
            ids_i = output_ids[name_i]
            ids_j = output_ids[name_j]
            
            # 토큰 일치율 계산
            min_len = min(len(ids_i), len(ids_j))
            if min_len > 0:
                matches = sum(1 for a, b in zip(ids_i[:min_len], ids_j[:min_len]) if a == b)
                match_rate = matches / min_len * 100
            else:
                match_rate = 0.0
            
            # 첫 번째 불일치 토큰 위치
            first_diff = -1
            for k in range(min_len):
                if ids_i[k] != ids_j[k]:
                    first_diff = k
                    break
            
            print(f"\n  [{name_i}] vs [{name_j}]")
            print(f"    Token match rate: {match_rate:.1f}% ({matches}/{min_len})")
            print(f"    Length: {len(ids_i)} vs {len(ids_j)} tokens")
            if first_diff >= 0:
                token_i = tokenizer.decode([ids_i[first_diff]])
                token_j = tokenizer.decode([ids_j[first_diff]])
                print(f"    First difference at token {first_diff}: '{token_i}' vs '{token_j}'")
            elif min_len > 0:
                print(f"    Outputs are identical up to min length ({min_len} tokens)")
    
    # 세 방법 모두 동일한지 확인
    print("\n" + "-" * 70)
    all_same = (output_ids[method_names[0]] == output_ids[method_names[1]] == output_ids[method_names[2]])
    if all_same:
        print("ALL THREE METHODS PRODUCED IDENTICAL OUTPUT")
    else:
        print("METHODS PRODUCED DIFFERENT OUTPUTS")
    print("-" * 70)
    
    # 결과를 JSON으로 저장
    save_path = "test_cache_comparison.json"
    save_data = {
        "model": MODEL_NAME,
        "prompt_length": prompt_length,
        "total_budget": TOTAL_BUDGET,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "configs": {
            "a2sf": {"forgetting_factor": 1.0, "layerwise_ratios": "all 1.0", "local_ratios": 0.125},
            "snap": {"observation_window": prompt_length},
            "sigmoid": {"a": 10.0, "b": prompt_length},
        },
        "results": results,
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
