import copy
import torch
import os
import numpy as np

from utils_lm_eval.modify_llama import convert_kvcache_llama_heavy_recent
from utils_lm_eval.ideal_llama import convert_kvcache_llama_heavy_recent_ideal
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_name = "meta-llama/Llama-2-7b-hf"

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name).half().eval()

check_point = None

root_path = os.path.dirname(__file__)

prompts = {
    "winogrande": "Dharma wanted to bake some cookies and cakes for the bake sale. She ended up only baking the cakes because she didn't have a cookie sheet.\n\nI always wonder how people prefer reading in a library instead of at the house because the lack of people at the library would make it easier to concentrate.",
    "piqa": "Question: To make pumpkin spice granola\nAnswer: With a wooden spoon, mix together 2 1/2 tsp pumpkin pie spice, 1/4 tsp salt, 1/4 cup light brown sugar, 1/3 cup canned pumpkin puree (not pumpkin pie filling) , 2 Tbsp unsweetened applesauce, 2 Tbsp honey, 1/2 tsp vanilla extract, 1/4 cup raisins, 1/4 cup craisins in a medium bowl. Mix into 3 cups rolled oats until evenly coated. Line a cookie sheet with parchment paper. Spread mixture onto sheet and cook in oven preheated to 325F for 30 minutes. Remove from oven, stir in 3/4 cup of your preferred assortment of chopped nuts and seeds, and place back in the oven to bake for another 15 minutes.\n\nQuestion: how do you cheese something?\nAnswer: sprinkle cheese all over it.",
    "openbookqa": "A man is filling up his tank at the gas station, then goes inside and pays. When he comes back out, his truck is being stolen! He chases the truck as the thief pulls away in it and begins to speed off. Though the truck was huge, right in front of him a moment ago, as the truck accelerates and the man struggles to keep up, the truck looks smaller\n\nA positive effect of burning biofuel is shortage of crops for the food supply",
    "arc_e": "Question: Homes that are built to be environmentally friendly because they use energy more efficiently than other homes are called \"green\" homes. \"Green\" homes often have reflective roofs and walls made of recycled materials. The windows in these energy-saving homes are double-paned, meaning each window has two pieces of glass. Double-paned windows have a layer of air between the window panes. This layer is a barrier against extreme temperatures and saves energy. A solar panel on a \"green\" home uses\nAnswer: a renewable energy source\n\nQuestion: Which is the function of the gallbladder?\nAnswer: store bile",
    "mathqa": "Question: at company x , senior sales representatives visit the home office once every 15 days , and junior sales representatives visit the home office once every 10 days . the number of visits that a junior sales representative makes in a 2 - year period is approximately what percent greater than the number of visits that a senior representative makes in the same period ?\nAnswer: 50 %\n\nQuestion: a circle graph shows how the budget of a certain company was spent : 61 percent for salaries , 10 percent for research and development , 6 percent for utilities , 5 percent for equipment , 3 percent for supplies , and the remainder for transportation . if the area of each sector of the graph is proportional to the percent of the budget it represents , how many degrees of the circle are used to represent transportation ?\nAnswer:"
}

ratio = 0.2

methods = {
    # "NO_PRUNING": (0.0, 1.0, 1.0, False),
    # "IDEAL": (ratio, 0.0, 1.0, True),
    # "H2O": (ratio/2, ratio/2, 1.0, False),
    "A2SF_ZERO": (ratio, 0.00, 0.1, False),
    # "LOW_DIMENSION": (ratio, 0.0, 0.1, False),
}

for name, (i, j, k, ideal) in tqdm(methods.items()):
    config.heavy_ratio = i
    config.recent_ratio = j
    config.penalty = k

    if (i + j < 1.0):
        if check_point is None:
            model.cpu()
            check_point = copy.deepcopy(model.state_dict())
        
        if ideal:
            convert_kvcache_llama_heavy_recent_ideal(model, config)
        else:
            convert_kvcache_llama_heavy_recent(model, config)
        model.load_state_dict(check_point)
        torch.cuda.empty_cache()
        model.half().eval().cuda()
    else:
        if check_point is None:
            model.cuda()
        else:
            config = AutoConfig.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_config(config)
            model.half().eval().cuda()
            
    for dataset, prompt in prompts.items():
        input_ids = tokenizer(prompt, add_special_tokens=True, return_tensors='pt').input_ids.cuda()

        with torch.no_grad():
            result = model(input_ids, output_attentions=True)
            
        folder_path = os.path.join(root_path, "analysis", "mask", "npy", dataset, name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for i in range(len(result.attentions)):
            np.save(os.path.join(folder_path, f"{i}.npy"), result.attentions[i].cpu().detach().numpy())