import argparse
import logging

import numpy as np
import torch
import json
import tqdm 
import copy

from rouge import Rouge

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from utils_real_drop.modify_llama import H2OLlamaAttention

def prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is not None:
        
        input_ids = input_ids[:, -1:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default='huggyllama/llama-7b')
    parser.add_argument("--cache_budget", type=int, default=20)
    parser.add_argument("--forgetting_factor", type=float, default=0.1)
    parser.add_argument("--length", type=int, default=64)
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.warning(f"device: {args.device}")

    rouge = Rouge()

    # Change to your custom prompt text
    prompt_text = '###\nArticle: Almost one million people visited the city during the six-week festival period over Christmas and Hogmanay. Organisers said almost 890,000 people visited the Edinburgh\'s Christmas events in 2014/15, contributing \u00a3199.5m to the local economy. The three-day Hogmanay celebrations attracted more than 150,000 people, creating an economic impact of \u00a341.8m. Charlie Wood, Edinburgh\'s Christmas festival director, said: \"This is great news for Edinburgh. The revenue generated does not go to the events themselves, the event organisers or to Edinburgh city council. \"This is money, which is going to the businesses of Edinburgh, be it retail, accommodation, food, drink, shopping and entertainment.\"\n\nSummarize the above article in 1 sentence.\nEdinburgh\'s winter festivals generated more than \u00a3241m for the city, according to organisers.\n\n###\nArticle: The 25-year-old, from North Ormesby, was shaping metal when a part from the press fell on his foot on 17 March. Teesside Magistrates\' Court heard that SM Thompson Limited, of Middlesbrough, had allowed dangerous lifting practices to go unchecked over 10 years. The firm admitted a Health and Safety Executive (HSE) breach and was fined \u00c2\u00a37,500. It must also pay \u00c2\u00a31,120 costs. The hearing heard how the worker had to have the big toe on his left foot amputated and two other toes removed. He was in hospital for seven days but has since returned to work, the hearing heard. HSE inspector Paul Wilson said: \"This worker\'s injuries need not have happened. \"The failure of SM Thompson to look properly at the risks involved and then organise the lifting operation properly put staff at needless risk. \"This sadly led to the painful and life-changing injuries suffered by this young man.\"\n\nSummarize the above article in 1 sentence.\nA Teesside steel firm has been fined after a worker was crushed by a press and had to have three toes amputated.\n\n###\nArticle: The colourful phenomenon was visible in Scotland and Northern Ireland, but was also spotted as far south as Anglesey in Wales and Staffordshire in England. Aurora Borealis occurs when electrically-charged particles from the sun enter the earth\'s atmosphere. Many people took to social media to share photographs of the dramatic show. Forecasters had predicted a solar storm and good conditions for Aurora Borealis, and sightings of green, pink, purple, red and yellow lights were reported for several hours from about 20:00 GMT. Gavin Chambers, an RSPB warden, tweeted pictures of vivid green in the sky over Lake Vyrnwy in Powys, Wales, saying: \"Well worth getting back out of bed for!!\" Donna Butcher tweeted: \"Just been watching an amazing display of Aurora from Staveley, Cumbria. Shafts of light streaming directly towards Polaris.\" You can email your pictures and video to yourpics@bbc.co.uk, and find out more about the Northern Lights here.\n\nSummarize the above article in 1 sentence.\nThere have been spectacular displays of the Aurora Borealis - better known as the Northern Lights - across parts of the UK overnight.\n\n###\nArticle: Prison Link Cymru had 1,099 referrals in 2015-16 and said some ex-offenders were living rough for up to a year before finding suitable accommodation. Workers at the charity claim investment in housing would be cheaper than jailing homeless repeat offenders. The Welsh Government said more people than ever were getting help to address housing problems. Changes to the Housing Act in Wales, introduced in 2015, removed the right for prison leavers to be given priority for accommodation. Prison Link Cymru, which helps people find accommodation after their release, said things were generally good for women because issues such as children or domestic violence were now considered. However, the same could not be said for men, the charity said, because issues which often affect them, such as post traumatic stress disorder or drug dependency, were often viewed as less of a priority. Andrew Stevens, who works in Welsh prisons trying to secure housing for prison leavers, said the need for accommodation was \"chronic\". \"There\'s a desperate need for it, finding suitable accommodation for those leaving prison there is just a lack of it everywhere,\" he said. \"It could take six months to a year, without a lot of help they could be on the streets for six months. \"When you think of the consequences of either being on the street, especially with the cold weather at the moment or you may have a roof over your head, sometimes there is only one choice.\" Mr Stevens believes building more one-bedroom flats could help ease the problem. \"The average price is a hundred pounds a week to keep someone in a rented flat, prison is a lot more than that so I would imagine it would save the public purse quite a few pounds,\" he said. Official figures show 830 one-bedroom properties were built in the year to March 2016, of an overall total of 6,900 new properties in Wales. Marc, 50, who has been in and out of prison for the past 20 years for burglary offences, said he struggled to find accommodation each time he was released. He said he would ask himself: \"Where am I going to stay? Where am I going to live? Have I got somewhere where I can see my daughter.\" \"You\'re put out among the same sort of people doing the same sort of thing, and it\'s difficult, it\'s difficult to get away from it. It\'s like every man for himself, there\'s nothing.\" Marc has now found stable accommodation with homeless charity Emmaus and said it had been life changing. \"You feel safe, you got hot food, you\'ve got company of people in similar situations to yourself but all dealing with different issues. It\'s a constructive, helpful atmosphere,\" he said. Tom Clarke, chief executive of Emmaus South Wales, agreed there was not enough support available. \"We do still see [people] homeless on the streets, so clearly they haven\'t got accommodation and haven\'t got provision,\" he said. \"I think the key is connecting people with the services they need. I don\'t delude myself that Emmaus can offer a one size fits all for everyone, we can\'t. \"But there must be other opportunities and given suitable encouragement I believe that can and should happen.\" A Welsh Government spokesman said the national pathway for homeless services to children, young people and adults in the secure estate had prevented many people from losing their home whilst serving their prison sentence. It added there were already significant demands for one-bedroom flats across the public and private sector and it was providing 20,000 new affordable homes in the next five years.\n\nSummarize the above article in 1 sentence.\n'

    model_name = args.model_name
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).half().eval()
    
    check_point = copy.deepcopy(model.state_dict())
    model.to(args.device)

    input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.to(args.device)
    print(f"prompt length: {input_ids.shape}")

    ######## Generate with Full Cache
    
    generate_ids = model.generate(input_ids, max_new_tokens=args.length, do_sample=False, temperature=1.0, top_p=1.0)
    result = tokenizer.batch_decode(generate_ids)[0]
    result = result.replace(prompt_text, "")
    result = result[:result.find("###")]
    print("################## Generated Context with Full Cache ###################")
    print(result)
    print()

    ######### Enable H2O

    model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(model, type(model))
    
    config.scoring_policy = "h2o"
    config.hh_size = int(args.cache_budget/2)
    config.recent_size = int(args.cache_budget/2)
    config.forgetting_factor = 1.0
    for layer_idx in range(config.num_hidden_layers):
        model.model.layers[layer_idx].self_attn = H2OLlamaAttention(config)
    model.load_state_dict(check_point)
    model.half().eval().to(args.device)

    generate_ids = model.generate(input_ids, max_new_tokens=args.length, do_sample=False, temperature=1.0, top_p=1.0)
    result_hh = tokenizer.batch_decode(generate_ids)[0]
    result_hh = result_hh.replace(prompt_text, "")
    result_hh = result_hh[:result_hh.find("###")]
    print("################## Generated Context with H2O ###################")
    print(result_hh)
    print()

    score = rouge.get_scores(result_hh, result, avg=True)

    print(score)
    print()

    ######### Enable A2SF
    
    config.scoring_policy = "a2sf"
    config.hh_size = int(args.cache_budget/2)
    config.recent_size = int(args.cache_budget/2)
    # config.hh_size = int(args.cache_budget)
    # config.recent_size = 0
    config.forgetting_factor = args.forgetting_factor
    for layer_idx in range(config.num_hidden_layers):
        model.model.layers[layer_idx].self_attn = H2OLlamaAttention(config)
    model.load_state_dict(check_point)
    model.half().eval().to(args.device)

    generate_ids = model.generate(input_ids, max_new_tokens=args.length, do_sample=False, temperature=1.0, top_p=1.0)
    result_hh = tokenizer.batch_decode(generate_ids)[0]
    result_hh = result_hh.replace(prompt_text, "")
    result_hh = result_hh[:result_hh.find("###")]
    print("################## Generated Context with A2SF ###################")
    print(result_hh)
    print()

    score = rouge.get_scores(result_hh, result, avg=True)

    print(score)
    print()