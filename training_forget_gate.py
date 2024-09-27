import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AdamW
from torch.utils.data import DataLoader
import json

model_name = "llama-2-7b"
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

train_data = []
with open('data/data.jsonl', 'r') as f:
    for line in f:
        train_data.append(json.loads(line))

train_encodings = tokenizer([item['text'] for item in train_data], truncation=True, padding=True)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = CustomDataset(train_encodings)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

for name, param in model.named_parameters():
    if 'query' not in name:
        param.requires_grad = False

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

model.train()
for epoch in range(10):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['input_ids'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1} completed with loss: {loss.item()}")

print("Training completed!")