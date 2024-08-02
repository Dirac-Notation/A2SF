import copy
import torch
import argparse

from lm_eval import tasks, evaluator
from lm_eval.models import huggingface
from lm_eval.tasks import initialize_tasks

from utils_lm_eval.lm_model import lm_model

initialize_tasks()

if __name__ == "__main__":
    task_list = ["openbookqa", "piqa", "arc_challenge", "arc_easy", "mathqa", "winogrande"]
    model_list = ["meta-llama/Llama-2-7b-hf"]
    fewshot_list = [5]
    ratio_list = [0.1, 0.2]

    for num_fewshot in fewshot_list:
        print(f"fewshot: {num_fewshot}")
        
        for model_name in model_list:
            
            lm = huggingface.HFLM(model_name, device="cpu", batch_size=16)
            check_point = copy.deepcopy(lm.model.state_dict())
            lm._device = "cuda"
            lm.model.cuda()

            print(f"model: {model_name}")

            for ratio in ratio_list:
                config = {}
                for i in range(24,32): config[f"ASDF_{i}"] = (0.0, ratio, 0.0, 0.2, i)

                for method, (streaming, selecting, recent, factor, tmp) in config.items():
                    
                    if method != "Full":
                        lm_model(
                            model_name=model_name,
                            lm=lm,
                            check_point=check_point,
                            device="cuda",
                            streaming_ratio=streaming,
                            selecting_ratio=selecting,
                            recent_ratio=recent,
                            forgetting_factor=factor,
                            tmp=tmp,
                        )
                    
                    print(f"================={method} streaming : {streaming:.2f} / selecting : {selecting:.2f} / recent : {recent:.2f} / factor : {factor:.2f}=================")
                    
                    results = evaluator.evaluate(lm, tasks.get_task_dict(task_list, num_fewshot=num_fewshot, fewshot_split="validation"))

                    print(evaluator.make_table(results))
                    if "groups" in results:
                        print(evaluator.make_table(results, "groups"))
                    print("=====================================================================================================================================")
                            
            lm.model.cpu()