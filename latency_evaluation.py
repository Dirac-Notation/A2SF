import torch
import math

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

from utils_real_drop.modify_llama import H2OLlamaForCausalLM, H2OLlamaAttention

name = "meta-llama/Llama-2-7b-hf"
input_length = 512
batch_size = 8

config = AutoConfig.from_pretrained(name)

# config.hh_size = math.ceil(input_length*0.2)
# config.recent_size = math.ceil(input_length*0.2)
# config.scoring_policy = "h2o"

config.hh_size = math.ceil(input_length*0.2)*2
config.recent_size = 0
config.scoring_policy = "a2sf"

config.forgetting_factor = 0.2

tokenizer = AutoTokenizer.from_pretrained(name)
model = H2OLlamaForCausalLM.from_pretrained(name, config=config)
# model = AutoModelForCausalLM.from_pretrained(name)

model.eval().half().cuda()

vocab_size = tokenizer.vocab_size

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

result_list = []

for _ in tqdm(range(10)):
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, input_length)).to(model.device)

    starter.record()
    model.generate(
        input_ids=input_ids,
        max_length=input_length*2
    )
    ender.record()
    torch.cuda.synchronize()
    
    for name, m in model.named_modules():
        if isinstance(m, H2OLlamaAttention):
            m._clean_cache()

result_list.append(starter.elapsed_time(ender))
averaged_result = sum(result_list)/len(result_list)

print(f"{config.scoring_policy} average latency({2*input_length} tokens) : {averaged_result/1000:.2f}s")