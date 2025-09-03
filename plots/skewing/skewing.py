import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

model_path = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path, skip_special_tokens=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

queries = {}

keies = {}

values = {}

hook_handles = []

def make_hook(layer_idx, attn_module, op_type):
    head_dim  = getattr(attn_module, "head_dim", None)
    rotary    = getattr(attn_module, "rotary_emb", None)

    def hook_fn(module, inputs, output):
        tensors = output
        
        bsz, seqlen, _ = tensors.shape

        tensors = tensors.view(bsz, seqlen, -1, head_dim).transpose(1, 2)

        if op_type == "v":
            values[layer_idx] = tensors.detach().to("cpu")
            return
            
        position_ids = torch.arange(seqlen, device=tensors.device)[None,:].to(tensors.device)

        cos, sin = rotary(tensors, seqlen)
        tensors_rope, _ = apply_rotary_pos_emb(tensors, tensors, cos, sin, position_ids)

        if op_type == "q":
            queries[layer_idx] = tensors_rope.detach().to("cpu")
        elif op_type == "k":
            keies[layer_idx] = tensors_rope.detach().to("cpu")

    return hook_fn

def get_skew_matrix(matrix):
    _, _, V = torch.svd(matrix)
    return V

def plot_matrix(matrix, save_path):
    # matrix = matrix.T.abs()
    matrix = matrix.abs()
    
    fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection="3d")

    # x = torch.arange(matrix.shape[0])
    # y = torch.arange(matrix.shape[1])
    # x, y = torch.meshgrid(x, y, indexing="ij")

    # ax.plot_surface(x, y, matrix, cmap="viridis")

    # ax.set_xlabel("Channel")
    # ax.set_ylabel("Token ID")
    # ax.set_zlabel("Absolute Value")
    plt.imshow(matrix, cmap="Blues", interpolation="none")
    plt.xlabel("Query")
    plt.ylabel("Key")
    # ax.set_zlabel("Absolute Diff Value")

    plt.savefig(save_path)
    plt.close()

for layer_idx in range(model.config.num_hidden_layers):
    attn = model.model.layers[layer_idx].self_attn
        
    handle_q = attn.q_proj.register_forward_hook(make_hook(layer_idx, attn, "q"))
    handle_k = attn.k_proj.register_forward_hook(make_hook(layer_idx, attn, "k"))
    handle_v = attn.v_proj.register_forward_hook(make_hook(layer_idx, attn, "v"))
    
    hook_handles.append(handle_q)
    hook_handles.append(handle_k)
    hook_handles.append(handle_v)

text = "You are given several news passages. Write a one-page summary of all news. \n\nNews:\nPassage 1:\nMembers of Occupy Philadelphia remain on site at City Hall into the evening of Nov. 28. (David M Warren / Staff Photographer) NEWLINE_CHAR NEWLINE_CHAR About 100 Occupy Philly protesters sat down on the cold concrete of Dilworth Plaza at 5 p.m. Sunday and waited to be rousted for violating a deadline Mayor Nutter had set for the group to leave. NEWLINE_CHAR NEWLINE_CHAR But the expected police eviction had not happened by late Sunday evening, and city officials continued to avoid saying when, or whether, they would throw the Occupiers and their tents off City Hall's so-called front lawn. NEWLINE_CHAR NEWLINE_CHAR \"We are expecting people to pack up and leave,\" said Mark McDonald, spokesman for Mayor Nutter. \"I'm not going to speculate about what the city might do at any time down the road from now.\" NEWLINE_CHAR NEWLINE_CHAR For about 90 minutes, protesters chanted slogans such as, \"What does democracy look like? This is what democracy looks like.\" NEWLINE_CHAR NEWLINE_CHAR They served squash soup in paper cups, handed out bottles of water, and railed against what they believe is excessive corporate greed and power. NEWLINE_CHAR NEWLINE_CHAR All the while, police looked on calmly. NEWLINE_CHAR NEWLINE_CHAR Police had no plans to evict anyone, Chief Inspector Joseph Sullivan said about 6:30 p.m. NEWLINE_CHAR NEWLINE_CHAR \"We look forward to working with Occupy Philadelphia and a resolution of the problem. Confrontation is never good. Anyone who is being fair would have to say that there is a big difference between the police reaction to Occupy Philadelphia than in other cities,\" he said. NEWLINE_CHAR NEWLINE_CHAR \"I definitely, definitely want to really stress that the vast majority of people participating in this movement have been cooperative, nonviolent, and very respectful,\" he said. NEWLINE_CHAR NEWLINE_CHAR Sullivan cautioned, however, that protesters would be prevented from setting up another camp elsewhere in the city unless they got a permit. NEWLINE_CHAR NEWLINE_CHAR The city has said it needs to erect fencing this week for work at Dilworth Plaza, including renovation of the SEPTA tunnels and the addition of grass, a cafe, stage, and winter ice rink. NEWLINE_CHAR NEWLINE_CHAR Over the last few days, Occupy Philly participants, many homeless, had begun to take down their tents. By 7:15 p.m. Sunday, roughly a third of what originally was about 300 tents were gone. NEWLINE_CHAR NEWLINE_CHAR Darrin Annussek, an unemployed career counselor, and William Tuttle, a student, were folding up Annussek's tent Sunday. Someone had stolen Tuttle's tent, and the men, who got married at Occupy Philly last month, said they probably would move to the Occupy site in Washington. NEWLINE_CHAR NEWLINE_CHAR \"I wanted to protest the fact that corporations have too much control and government hasn't done much to stop it,\" Annussek said. NEWLINE_CHAR NEWLINE_CHAR He said he believed Nutter had exaggerated the health and safety concerns the mayor cited in explaining why he could not issue protesters a permit for a new location once the $50 million renovation project began. NEWLINE_CHAR NEWLINE_CHAR \"I really thought this would become one of the longer occupations,\" Annussek said. NEWLINE_CHAR NEWLINE_CHAR Ellen Rogovinhart of Elkins Park said she thought the city could have done more to help the Occupiers find a new location, possibly in a church. NEWLINE_CHAR NEWLINE_CHAR \"I know the city has work to do,\" Rogovinhart said. \"But I think this is a very important movement.\" NEWLINE_CHAR NEWLINE_CHAR She was hoping Philadelphia police would not use force, as police in Oakland and other cities have, to get Occupiers to move. NEWLINE_CHAR NEWLINE_CHAR Rogovinhart held a sign that read, \"The eyes of the world are watching to see if we are the city of brotherly and sisterly love today.\" NEWLINE_CHAR NEWLINE_CHAR At times it seemed the whole world were watching. On Sunday night, hip-hop mogul Russell Simmons, using the screen name @Uncle Rush, went on Twitter to ask Nutter to \"remember this is a nonviolent movement - please show restraint tonight.\" NEWLINE_CHAR NEWLINE_CHAR Nutter tweeted back that he agreed. NEWLINE_CHAR NEWLINE_CHAR On the west side of City Hall, several hundred people gathered to observe what would happen to those risking arrest. The crowd used its \"human mic\" system of amplification, in which others repeat what each person says so all could hear. NEWLINE_CHAR NEWLINE_CHAR \"I went to college. I graduated with honors,\" Marcel Williams Foster yelled, and the crowd repeated. \"I work three part-time jobs with no benefits and I have $50,000 in student loans.\" NEWLINE_CHAR NEWLINE_CHAR Lauren Keiser, 26, a student from Audubon, said she was willing to get arrested because she believed homeless people, who are a constant presence on the plaza, deserve more help. She thinks the money that will transform Dilworth could be better spent on housing, addiction programs, and other services.\nPassage 2:\nThe LAPD early Monday declared an unlawful assembly on the streets surrounding City Hall and ordered Occupy L.A. protesters to immediately disperse or face arrest. NEWLINE_CHAR NEWLINE_CHAR Police officers carrying batons, plastic handcuffs and non-lethal weapons lined up on 1st Street directly outside LAPD headquarters in anticipation of possible arrests of protesters who were standing in the streets. NEWLINE_CHAR NEWLINE_CHAR \"It is not our intent to clear the park at this time,\" an officer said over a loudspeaker. \"It is only our intent to clear the street. Thank you in advance for your cooperation.\" NEWLINE_CHAR NEWLINE_CHAR PHOTOS: Occupy L.A. NEWLINE_CHAR NEWLINE_CHAR The dispersal order came nearly four hours after the midnight Sunday deadline for protesters to clear the lawn at City Hall, where Occupy L.A. protesters have camped out for two months. NEWLINE_CHAR NEWLINE_CHAR Police were on scene since before midnight, urging protesters to stay off the streets and warning them that arrests would be imminent. They first gave a 4 a.m. deadline to arrest protesters who were in the street, then pushed the time to 4:30 a.m. NEWLINE_CHAR NEWLINE_CHAR No arrests have yet been made. NEWLINE_CHAR NEWLINE_CHAR \"Right now, we have no plans to go into the encampment,\" LAPD Cmdr. Andy Smith said. NEWLINE_CHAR NEWLINE_CHAR RELATED: NEWLINE_CHAR NEWLINE_CHAR FULL COVERAGE: Occupy protests around the nation NEWLINE_CHAR NEWLINE_CHAR City councilman urges Occupy L.A. to move indoors, into politics NEWLINE_CHAR NEWLINE_CHAR Occupy L.A. campers play, pray as city’s midnight deadline looms NEWLINE_CHAR NEWLINE_CHAR -- Nicole Santa Cruz and Rick Rojas at City Hall NEWLINE_CHAR NEWLINE_CHAR Photo: With a line of LAPD officers behind them, Occupy L.A. demonstrators sit in the middle of the street near Los Angeles City Hall early Monday morning. Credit: Rick Loomis / Los Angeles Times\nPassage 3:\n1 of 6. The Occupy Los Angeles encampment at City Hall Park is seen before the midnight deadline for eviction from City Hall Park passes in Los Angeles, November 27, 2011. NEWLINE_CHAR NEWLINE_CHAR LOS ANGELES (Reuters) - Police in riot gear closed in before dawn on Monday on anti-Wall Street activists in Los Angeles who defied a midnight deadline to vacate a camp outside City Hall, but stopped short of clearing the encampment. NEWLINE_CHAR NEWLINE_CHAR Police managed to reopen blocked streets for morning rush-hour commuters after a tense standoff with protesters who had taken over a downtown intersection, but remnants of a crowd that had swelled to 2,000 overnight remained at City Hall. NEWLINE_CHAR NEWLINE_CHAR Four demonstrators were arrested during the brief confrontation, accused of being present at an unlawful assembly, before police ultimately pulled back from City Hall park. NEWLINE_CHAR NEWLINE_CHAR Later, attorneys for Occupy LA asked a federal judge for an injunction barring police from evicting the camp, arguing that Mayor Antonio Villaraigosa and police chief Charlie Beck had violated their civil rights by ordering it dismantled. NEWLINE_CHAR NEWLINE_CHAR The Los Angeles encampment, which officials had tolerated for weeks even as other cities moved in to clear out similar camps, is among the largest on the West Coast aligned with a 2-month-old national Occupy Wall Street movement protesting economic inequality and excesses of the U.S. financial system. NEWLINE_CHAR NEWLINE_CHAR Villaraigosa eventually gave protesters until just after midnight to remove their tents and leave or face a forcible removal, setting the stage for the latest showdown between leaders of a major U.S. city and the Occupy movement. NEWLINE_CHAR NEWLINE_CHAR But about two hours after the eviction deadline had passed, police commanders said they would permit the Occupy LA encampment to stay until at least daybreak. Police Commander Andrew Smith later said he thought it was \"highly unlikely\" that the camp would be forced to shut down on Monday. NEWLINE_CHAR NEWLINE_CHAR Elsewhere in the country, a 5 p.m. Sunday deadline set by Philadelphia officials for Occupy protesters there to move from a similar encampment came and went without incident. NEWLINE_CHAR NEWLINE_CHAR Dozens of people heeded the order but many tents and other structures stayed put. Police sources said authorities were hoping the rest of the protesters would relocate voluntarily and that no major actions were expected before Tuesday. NEWLINE_CHAR NEWLINE_CHAR 'WHOSE STREET? OUR STREET!' NEWLINE_CHAR NEWLINE_CHAR Staking its place since October 1 on the grounds surrounding City Hall, the Los Angeles camp had grown to roughly 400 tents and 700 to 800 people, organizers and municipal officials said. At least a third of campers were believed to be homeless. NEWLINE_CHAR NEWLINE_CHAR By Sunday night the size of the crowd outside City Hall swelled further as supporters from organized labor, clergy, civil rights and other groups streamed into the area, answering a call for an 11th-hour show of support. NEWLINE_CHAR NEWLINE_CHAR The overall number of protesters, some wearing gas masks, had grown to at least 2,000 by late Sunday, police estimated. NEWLINE_CHAR NEWLINE_CHAR After keeping out of sight throughout the day on Sunday, police began to make their presence known as the mayor's eviction deadline passed, and the protesters' mood turned from calm and festive to rowdy. NEWLINE_CHAR NEWLINE_CHAR Demonstrators and police confronted each other overnight but except for some debris thrown by protesters at one point, there was no violence. One skirmish involved an intersection occupied by protesters. NEWLINE_CHAR NEWLINE_CHAR Minutes after ordering protesters in the street to disperse, dozens of helmeted police carrying night sticks and special shotguns for firing \"bean-bag\" projectiles enclosed the intersection and forced their way into the crowd. NEWLINE_CHAR NEWLINE_CHAR Most in the crowd quickly retreated into the park, as onlookers chanted \"Whose street? Our Street\" at police and shouted at those defying police to \"Get off the street!\" NEWLINE_CHAR NEWLINE_CHAR Someone hurled what appeared to be pieces of a bamboo pole and a bottle at police, and Smith said four people were arrested. NEWLINE_CHAR NEWLINE_CHAR Los Angeles has been relatively accommodating to its Occupy group compared to other major cities, with Villaraigosa at one point providing ponchos to campers when it rained. NEWLINE_CHAR NEWLINE_CHAR But after the collapse of negotiations aimed at persuading protesters to relocate voluntarily, the mayor said last week the encampment would have to go. NEWLINE_CHAR NEWLINE_CHAR The mayor complimented the protesters on Sunday for staying peaceful. But he added in a statement: \"It is time for Occupy LA to move from focusing their efforts to hold a particular patch of park land to spreading the message of economic justice and restoration of balance to American society.\" NEWLINE_CHAR NEWLINE_CHAR He said he hoped to avoid the sporadic violence that erupted in other cities when police used force against Occupy protesters. NEWLINE_CHAR NEWLINE_CHAR A number of protesters early on Monday credited the police with showing restraint, including Clark Davis, an Occupy LA organizer, who said to Smith and a group of other officers standing by, \"You guys have been fantastic.\" NEWLINE_CHAR NEWLINE_CHAR (Writing by Steve Gorman and Dan Whitcomb; Additional reporting by Lucy Nicholson and Dave Warner in Philadelphia; Editing by Greg McCune and Cynthia Johnston)\n\n\nNow, write a one-page summary of all the news.\n\nSummary:"

input_ids = tokenizer(text, return_tensors="pt").input_ids
input_ids = torch.cat([input_ids[:,:128], input_ids[:,-128:]], dim=1)
input_ids = input_ids.to("cuda")

with torch.no_grad():
    outputs = model(input_ids)

for h in hook_handles:
    h.remove()

num_hidden_layers = model.config.num_hidden_layers
num_attention_heads = model.config.num_attention_heads
num_key_value_heads = model.config.num_key_value_heads
num_key_value_groups = num_attention_heads // num_key_value_heads
head_dim = model.config.hidden_size // num_attention_heads

att_mask = torch.triu(torch.full((input_ids.size(1), input_ids.size(1)), float("-inf")), diagonal=1)[None,:,:]

for layer_idx in tqdm(range(num_hidden_layers)):
    query = queries[layer_idx][0].float()
    key = keies[layer_idx][0].float()

    k_skew_matrix = get_skew_matrix(key)

    skewed_key = key @ k_skew_matrix
    skewed_dim = 96
    reconstructed_key = skewed_key[:,:,:skewed_dim] @ k_skew_matrix[:,:,:skewed_dim].transpose(1,2)

    key = key[:,None,:,:].expand(-1,num_key_value_groups,-1,-1).reshape(num_attention_heads,query.size(1),head_dim)
    reconstructed_key = reconstructed_key[:,None,:,:].expand(-1,num_key_value_groups,-1,-1).reshape(num_attention_heads,query.size(1),head_dim)

    original_attn = torch.softmax(torch.matmul(query,key.transpose(1,2)) + att_mask, dim=-1).pow(0.25)
    reconstructed_attn = torch.softmax(torch.matmul(query,reconstructed_key.transpose(1,2)) + att_mask, dim=-1).pow(0.25)
    
    # save_dir = os.path.join("models_skew", "skew_matrix", "llama3", f"layer{layer_idx}")
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # torch.save(k_skew_matrix, os.path.join(save_dir, "k_skew_matrix.pt"))
    
    for head_idx in range(num_key_value_heads):
        # save_dir = os.path.join(DIR_PATH, "pre_skewing", f"layer{layer_idx}", f"head{head_idx}")
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)

        # for head_idx_inner in range(num_key_value_groups):
        #     # plot_matrix(query[head_idx_inner], os.path.join(save_dir, f"query_{head_idx_inner}.png"))
        #     plot_matrix(key[head_idx_inner], os.path.join(save_dir, f"key_{head_idx_inner}.png"))

        save_dir = os.path.join(DIR_PATH, "post_skewing", f"layer{layer_idx}", f"head{head_idx}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plot_matrix(key[head_idx], os.path.join(save_dir, f"key_{head_idx}.png"))
        plot_matrix(skewed_key[head_idx], os.path.join(save_dir, f"skewed_key_{head_idx}.png"))
        plot_matrix(reconstructed_key[head_idx], os.path.join(save_dir, f"reconstructed_key_{head_idx}.png"))
        
        for head_idx_inner in range(num_key_value_groups):
            plot_matrix(original_attn[head_idx_inner], os.path.join(save_dir, f"original_attn_{head_idx_inner}.png"))
            plot_matrix(reconstructed_attn[head_idx_inner], os.path.join(save_dir, f"reconstructed_attn_{head_idx_inner}.png"))