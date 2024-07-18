import copy
import torch
import argparse

from lm_eval import tasks, evaluator
from lm_eval.models import huggingface
from lm_eval.tasks import initialize_tasks

from utils_lm_eval.lm_model import lm_model

initialize_tasks()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate language models on various tasks.")
    
    parser.add_argument("--task_list", nargs="+", help="List of tasks to evaluate")
    parser.add_argument("--model_list", nargs="+", help="List of model names")
    parser.add_argument("--fewshot_list", nargs="+", type=int, help="List of fewshot values")
    parser.add_argument("--ratio_list", nargs="+", type=float, help="List of ratio values")

    args = parser.parse_args()

    task_list = args.task_list
    model_list = args.model_list
    fewshot_list = args.fewshot_list
    ratio_list = args.ratio_list

    for num_fewshot in fewshot_list:
        print(f"fewshot: {num_fewshot}")
        
        for model_name in model_list:
            
            lm = huggingface.HFLM(model_name, device="cpu", batch_size=16)
            check_point = copy.deepcopy(lm.model.state_dict())
            lm.model.cuda()

            print(f"model: {model_name}")
            
            # print("Full")
            # lm_test(lm, task_list, num_fewshot)

            for ratio in ratio_list:
                print(f"================={ratio}=================")

                config = {
                    # "IDEAL": (ratio, 0.00, 1.0, True),
                    "H2O": (ratio/2, ratio/2, 1.0, False),
                    # "A2SF_0": (ratio, 0.00, 0.00, False),
                    # "A2SF_5": (ratio, 0.00, 0.05, False),
                    # "A2SF_10": (ratio, 0.00, 0.10, False),
                    # "A2SF_15": (ratio, 0.00, 0.15, False),
                    # "A2SF_20": (ratio, 0.00, 0.20, False),
                    # "A2SF_25": (ratio, 0.00, 0.25, False),
                    # "A2SF_30": (ratio, 0.00, 0.30, False),
                    # "A2SF_35": (ratio, 0.00, 0.35, False),
                    # "A2SF_40": (ratio, 0.00, 0.40, False),
                    # "A2SF_45": (ratio, 0.00, 0.45, False),
                    # "A2SF_50": (ratio, 0.00, 0.50, False),
                    # "A2SF_55": (ratio, 0.00, 0.55, False),
                    # "A2SF_60": (ratio, 0.00, 0.60, False),
                    # "A2SF_65": (ratio, 0.00, 0.65, False),
                    # "A2SF_70": (ratio, 0.00, 0.70, False),
                    # "A2SF_75": (ratio, 0.00, 0.75, False),
                    # "A2SF_80": (ratio, 0.00, 0.80, False),
                    # "A2SF_85": (ratio, 0.00, 0.85, False),
                    # "A2SF_90": (ratio, 0.00, 0.90, False),
                    # "A2SF_95": (ratio, 0.00, 0.95, False),
                    # "A2SF": (ratio, 0.00, 0.2, False),
                }

                for method, (select, local, penalty, ideal) in config.items():
                    lm_model(
                        model_name=model_name,
                        lm=lm,
                        check_point=check_point,
                        device="cuda",
                        heavy_ratio=select,
                        recent_ratio=local,
                        penalty=penalty,
                        ideal=ideal
                    )
                    results = evaluator.evaluate(lm_model, tasks.get_task_dict(task_list, num_fewshot=num_fewshot, fewshot_split="validation"))

                    print(method)
                    print(evaluator.make_table(results))
                    if "groups" in results:
                        print(evaluator.make_table(results, "groups"))
                    print("==============================================")
                            
            lm.model.cpu()