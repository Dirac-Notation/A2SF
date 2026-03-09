"""
선택한 캐시 기법 1개에 대해 두 구현을 비교:
1) model.generate(...) 기반 생성
2) model.forward(...) + past_key_values 기반 수동 디코딩

동일한 프롬프트/시드 조건에서 두 경로의 결과가 일치하는지 확인합니다.
"""

import torch
import json
import os

from utils import load_model, set_seed, CompressionConfig

PROMPTS = [
    (
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
    ),
    (
        "Answer the following question based on the given context.\n\n"
        "Context: The Great Wall of China is a series of fortifications that were built across the historical "
        "northern borders of ancient Chinese states and Imperial China as protection against various nomadic groups. "
        "Several walls were built from as early as the 7th century BC, with selective stretches later joined by "
        "Qin Shi Huang (220-206 BC), the first emperor of China. Little of the Qin wall remains. "
        "Later on, many successive dynasties built and maintained multiple stretches of border walls. "
        "The best-known sections of the wall were built by the Ming dynasty (1368-1644). "
        "The total length of all sections ever built is over 21,196 kilometres. "
        "The wall's width ranges from 4.5 to 9.1 metres at the base and tapers to 3.7 metres at the top.\n\n"
        "Question: What is the total length of the Great Wall of China?\n\nAnswer:"
    ),
    (
        "Answer the following question based on the given context.\n\n"
        "Context: Photosynthesis is a process used by plants and other organisms to convert light energy into "
        "chemical energy that, through cellular respiration, can later be released to fuel the organism's activities. "
        "Some of this chemical energy is stored in carbohydrate molecules, such as sugars and starches, which are "
        "synthesized from carbon dioxide and water. In most cases, oxygen is also released as a waste product that "
        "sustains nearly all life on Earth. Most plants, algae, and cyanobacteria perform photosynthesis; such "
        "organisms are called photoautotrophs. Photosynthesis is largely responsible for producing and maintaining "
        "the oxygen content of the Earth's atmosphere, and supplies most of the biological energy necessary for "
        "complex life on Earth.\n\n"
        "Question: What is the waste product released during photosynthesis?\n\nAnswer:"
    ),
    (
        "Answer the following question based on the given context.\n\n"
        "Context: The Amazon River is the largest river by discharge volume of water in the world, and the "
        "disputed longest river in the world compared to the Nile. The headwaters of the Apurimac River on "
        "Nevado Mismi had been considered for nearly a century as the Amazon's most distant source. The river "
        "has a series of major tributaries in Colombia, Ecuador, and Peru, some of which flow into the Maranon "
        "and Ucayali, and others directly into the Amazon itself. The Amazon basin is the largest drainage basin "
        "in the world, with an area of approximately 7,050,000 square kilometres. It covers roughly 40 percent "
        "of the South American continent. The Amazon enters Brazil with only one-fifth of the flow it finally "
        "discharges into the Atlantic Ocean.\n\n"
        "Question: What percentage of the South American continent does the Amazon basin cover?\n\nAnswer:"
    ),
    (
        "Answer the following question based on the given context.\n\n"
        "Context: The human brain is the central organ of the human nervous system, and with the spinal cord "
        "makes up the central nervous system. The brain consists of the cerebrum, the brainstem and the cerebellum. "
        "It controls most of the activities of the body, processing, integrating, and coordinating the information "
        "it receives from the sense organs, and making decisions as to the instructions sent to the rest of the body. "
        "The brain is contained in, and protected by, the skull bones of the head. The cerebrum is the largest part "
        "of the human brain. It is divided into two cerebral hemispheres. The cerebral cortex is an outer layer of "
        "grey matter, covering the core of white matter. The adult human brain weighs on average about 1.2 to "
        "1.4 kilograms, which is about 2% of the total body weight.\n\n"
        "Question: What is the average weight of an adult human brain?\n\nAnswer:"
    ),
    (
        "Answer the following question based on the given context.\n\n"
        "Context: The theory of general relativity was published by Albert Einstein in 1915. It is the geometric "
        "theory of gravitation and the current description of gravitation in modern physics. General relativity "
        "generalizes special relativity and refines Newton's law of universal gravitation, providing a unified "
        "description of gravity as a geometric property of space and time, or four-dimensional spacetime. In "
        "particular, the curvature of spacetime is directly related to the energy and momentum of whatever matter "
        "and radiation are present. Some predictions of general relativity differ significantly from those of "
        "classical physics, especially concerning the passage of time, the geometry of space, the motion of bodies "
        "in free fall, and the propagation of light.\n\n"
        "Question: When was the theory of general relativity published?\n\nAnswer:"
    ),
    (
        "Answer the following question based on the given context.\n\n"
        "Context: The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions. It extends from "
        "the Arctic Ocean in the north to the Southern Ocean in the south, and is bounded by the continents of Asia "
        "and Australia in the west and the Americas in the east. At 165,250,000 square kilometres in area, this "
        "largest division of the World Ocean covers about 46% of Earth's water surface and about 32% of its total "
        "surface area, making it larger than all of Earth's land area combined. The Mariana Trench in the western "
        "North Pacific is the deepest point in the world, reaching a depth of 10,994 metres. The Pacific Ocean was "
        "sighted by Europeans early in the 16th century, first by Spanish explorer Vasco Nunez de Balboa in 1513.\n\n"
        "Question: What is the deepest point in the Pacific Ocean and how deep is it?\n\nAnswer:"
    ),
    (
        "Answer the following question based on the given context.\n\n"
        "Context: DNA, or deoxyribonucleic acid, is a molecule composed of two polynucleotide chains that coil "
        "around each other to form a double helix. The molecule carries genetic instructions for the development, "
        "functioning, growth, and reproduction of all known organisms and many viruses. DNA and ribonucleic acid "
        "(RNA) are nucleic acids. Alongside proteins, lipids and complex carbohydrates, nucleic acids are one of "
        "the four major types of macromolecules that are essential for all known forms of life. The two DNA strands "
        "are known as polynucleotides as they are composed of simpler monomeric units called nucleotides. Each "
        "nucleotide is composed of one of four nitrogen-containing nucleobases (cytosine, guanine, adenine, or "
        "thymine), a sugar called deoxyribose, and a phosphate group.\n\n"
        "Question: What are the four nucleobases found in DNA?\n\nAnswer:"
    ),
    (
        "Answer the following question based on the given context.\n\n"
        "Context: The Industrial Revolution, which took place from the 18th to 19th centuries, was a period during "
        "which predominantly agrarian, rural societies in Europe and America became industrial and urban. Prior to "
        "the Industrial Revolution, manufacturing was often done in people's homes, using hand tools or basic machines. "
        "Industrialization marked a shift to powered, special-purpose machinery, factories and mass production. "
        "The iron and textile industries, along with the development of the steam engine, played central roles in "
        "the Industrial Revolution. It began in Britain in the late 1700s and from there spread to other parts of "
        "the world, including the United States, by the early 1800s. The revolution improved standards of living "
        "for some, but resulted in poor working and living conditions for the working class.\n\n"
        "Question: Where did the Industrial Revolution begin?\n\nAnswer:"
    ),
    (
        "Answer the following question based on the given context.\n\n"
        "Context: The International Space Station (ISS) is a modular space station in low Earth orbit. It is a "
        "multinational collaborative project involving five participating space agencies: NASA, Roscosmos, JAXA, "
        "ESA, and CSA. The station serves as a microgravity and space environment research laboratory in which "
        "scientific research is conducted in astrobiology, astronomy, meteorology, physics, and other fields. "
        "The ISS maintains an orbit with an average altitude of 408 kilometres by means of reboost manoeuvres "
        "using the engines of the Zvezda Service Module or visiting spacecraft. The ISS circles the Earth in "
        "roughly 92 minutes, completing 15.5 orbits per day. The station has been continuously occupied since "
        "November 2000, making it the longest continuous human presence in low Earth orbit.\n\n"
        "Question: How many orbits per day does the ISS complete?\n\nAnswer:"
    ),
]

MODEL_NAME = "llama3"
TOTAL_BUDGET = 128
MAX_NEW_TOKENS = 64
SEED = 42
TARGET_METHOD = "sigmoid"  # "a2sf", "snap", "sigmoid", "h2o", "full"


def build_compression_config(method, total_budget, num_layers):
    config = CompressionConfig()
    config["compression_method"] = method

    if method == "a2sf":
        config["total_budget"] = total_budget
        config["forgetting_factor"] = 0.75
        config["layerwise_ratios"] = [1.0] * num_layers
        config["local_ratios"] = 0.125
    elif method == "snap":
        config["total_budget"] = total_budget
        config["observation_window"] = 16
    elif method == "sigmoid":
        config["total_budget"] = total_budget
        config["a"] = 10.0
        config["b"] = 16
    elif method == "h2o":
        config["total_budget"] = total_budget
        config["heavy_budget_ratio"] = 0.5
        config["recent_budget_ratio"] = 0.5
    elif method == "full":
        pass
    else:
        raise ValueError(f"Unsupported method: {method}")

    return config


def run_generate_path(model, tokenizer, input_ids, attention_mask, config, max_new_tokens, seed):
    """generate 경로 실행"""
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


def run_forward_path(model, tokenizer, input_ids, attention_mask, config, max_new_tokens, seed):
    """forward + cache 경로 실행 (greedy)"""
    model.init_cache(config)
    set_seed(seed)

    generated_ids = input_ids.clone()
    cur_attention_mask = attention_mask.clone()
    past_key_values = None

    with torch.inference_mode():
        outputs = model(
            input_ids=generated_ids,
            attention_mask=cur_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        for _ in range(max_new_tokens):
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            new_mask = torch.ones(
                (cur_attention_mask.shape[0], 1),
                dtype=cur_attention_mask.dtype,
                device=cur_attention_mask.device,
            )
            cur_attention_mask = torch.cat([cur_attention_mask, new_mask], dim=-1)

            if tokenizer.eos_token_id is not None and torch.all(next_token == tokenizer.eos_token_id):
                break

            outputs = model(
                input_ids=next_token,
                attention_mask=cur_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

    return generated_ids[0]


def main():
    set_seed(SEED)
    
    # =========================================================================
    # 1. 모델 및 토크나이저 로드
    # =========================================================================
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load_model(MODEL_NAME)
    num_layers = model.config.num_hidden_layers
    print(f"  - Number of layers: {num_layers}")
    print(f"  - Target method: {TARGET_METHOD}")
    print(f"  - Total budget: {TOTAL_BUDGET}")
    print(f"  - Max new tokens: {MAX_NEW_TOKENS}")
    print(f"  - Number of prompts: {len(PROMPTS)}")
    
    all_results = []
    
    for p_idx, prompt in enumerate(PROMPTS):
        print("\n" + "#" * 70)
        print(f"  PROMPT {p_idx + 1}/{len(PROMPTS)}")
        print("#" * 70)
        
        # =====================================================================
        # 2. 프롬프트 준비
        # =====================================================================
        formatted_prompt = f"[INST]{prompt}[/INST]"
        
        input_enc = tokenizer(formatted_prompt, truncation=False, return_tensors="pt")
        input_ids = input_enc.input_ids.to(model.device)
        attention_mask = input_enc.attention_mask.to(torch.bfloat16).to(model.device)
        
        prompt_length = input_ids.shape[1]
        print(f"  Prompt length (tokens): {prompt_length}")
        
        # =====================================================================
        # 3. 선택한 기법으로 generate/forward 각각 수행
        # =====================================================================
        config = build_compression_config(TARGET_METHOD, TOTAL_BUDGET, num_layers)

        print(f"\n  [1/2] generate path ({TARGET_METHOD})")
        output_generate = run_generate_path(
            model,
            tokenizer,
            input_ids,
            attention_mask,
            config,
            MAX_NEW_TOKENS,
            SEED,
        )
        gen_ids = output_generate[prompt_length:].tolist()
        pred_generate = tokenizer.decode(output_generate[prompt_length:], skip_special_tokens=True)
        print(f"    Generated: {pred_generate[:200]}{'...' if len(pred_generate) > 200 else ''}")

        print(f"  [2/2] forward path ({TARGET_METHOD})")
        output_forward = run_forward_path(
            model,
            tokenizer,
            input_ids,
            attention_mask,
            config,
            MAX_NEW_TOKENS,
            SEED,
        )
        fwd_ids = output_forward[prompt_length:].tolist()
        pred_forward = tokenizer.decode(output_forward[prompt_length:], skip_special_tokens=True)
        print(f"    Generated: {pred_forward[:200]}{'...' if len(pred_forward) > 200 else ''}")

        # =====================================================================
        # 4. 결과 비교
        # =====================================================================
        print(f"\n  --- COMPARISON (Prompt {p_idx + 1}) ---")
        print(f"\n    [generate] {pred_generate[:120]}{'...' if len(pred_generate) > 120 else ''}")
        print(f"    [forward ] {pred_forward[:120]}{'...' if len(pred_forward) > 120 else ''}")

        min_len = min(len(gen_ids), len(fwd_ids))
        if min_len > 0:
            matches = sum(1 for a, b in zip(gen_ids[:min_len], fwd_ids[:min_len]) if a == b)
            match_rate = matches / min_len * 100
        else:
            match_rate = 0.0
            matches = 0

        first_diff = -1
        for k in range(min_len):
            if gen_ids[k] != fwd_ids[k]:
                first_diff = k
                break

        print(f"\n  TOKEN-LEVEL:")
        print(f"    [generate] vs [forward]: {match_rate:.1f}% ({matches}/{min_len})", end="")
        if first_diff >= 0:
            token_gen = tokenizer.decode([gen_ids[first_diff]])
            token_fwd = tokenizer.decode([fwd_ids[first_diff]])
            print(f", first diff at {first_diff}: '{token_gen}' vs '{token_fwd}'")
        else:
            print(", identical")

        identical = gen_ids == fwd_ids
        print(f"  => {'IDENTICAL' if identical else 'DIFFERENT'}")

        all_results.append({
            "prompt_idx": p_idx,
            "prompt_length": prompt_length,
            "method": TARGET_METHOD,
            "generate_result": pred_generate,
            "forward_result": pred_forward,
            "token_match_rate": f"{match_rate:.1f}%",
            "identical": identical,
        })
    
    # =========================================================================
    # 5. 전체 요약 및 저장
    # =========================================================================
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    identical_count = sum(1 for r in all_results if r["identical"])
    print(f"  Identical outputs: {identical_count}/{len(all_results)}")
    for r in all_results:
        status = "IDENTICAL" if r["identical"] else "DIFFERENT"
        print(f"    Prompt {r['prompt_idx'] + 1} (len={r['prompt_length']}): {status}")
    
    save_path = "test_cache_comparison.json"
    save_data = {
        "model": MODEL_NAME,
        "target_method": TARGET_METHOD,
        "total_budget": TOTAL_BUDGET,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "num_prompts": len(PROMPTS),
        "prompt_results": all_results,
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
