import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import shutil

# --- 실행되는 스크립트가 위치한 폴더 경로 가져오기 ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# --- 설정 및 폴더 준비 ---
model_id = "meta-llama/Llama-3.2-1B-Instruct" 

# plots 폴더를 현재 스크립트가 위치한 폴더 내부에 생성
BASE_PLOT_DIR = os.path.join(script_dir, "plots")
if os.path.exists(BASE_PLOT_DIR):
    shutil.rmtree(BASE_PLOT_DIR)  # 기존 폴더 삭제 후 새로 생성
os.makedirs(BASE_PLOT_DIR, exist_ok=True)

def setup_folders(layer_idx, head_idx):
    layer_dir = os.path.join(BASE_PLOT_DIR, f"layer_{layer_idx}")
    os.makedirs(layer_dir, exist_ok=True)
    head_dir = os.path.join(layer_dir, f"head_{head_idx}")
    os.makedirs(head_dir, exist_ok=True)
    return head_dir

def get_save_path(dir_path, filename):
    return os.path.join(dir_path, filename)

# --- 모델 및 토크나이저 로드 ---
print(f"Loading model: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, output_attentions=True)
model.eval() # 평가 모드
print("Model loaded.")

# --- 입력 텍스트 처리 및 어텐션 맵 획득 ---
text = "I want to go to the park with my dog and play ball."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

attentions = outputs.attentions
num_layers = len(attentions)
num_heads = attentions[0].shape[1]
seq_len = attentions[0].shape[2]

# --- 이미지 저장 함수 정의 ---
def save_attention_map(attn_map, dir_path, filename, box_start=None, box_height=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(attn_map, cmap='Blues', vmin=0, vmax=attn_map.max() if attn_map.max() > 0 else 1.0)
    
    if box_start is not None and box_height is not None:
        rect = patches.Rectangle(
            (-0.5, box_start - 0.5), seq_len, box_height, 
            linewidth=2, edgecolor='green', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_ylabel("Query")
    ax.xaxis.set_label_position('top')
    ax.set_xlabel("Key")
    
    plt.savefig(get_save_path(dir_path, filename), bbox_inches='tight')
    plt.close()

def save_vector(vec, dir_path, filename, cmap='Blues', vmin=None, vmax=None):
    is_horizontal = vec.shape[0] == 1
    figsize = (6, 0.5) if is_horizontal else (0.5, 6)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(vec, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.savefig(get_save_path(dir_path, filename), bbox_inches='tight')
    plt.close()

# --- 모든 레이어와 헤드에 대해 반복하여 저장 ---
print(f"Generating plots for {num_layers} layers and {num_heads} heads each ({num_layers * num_heads} total)...")

for l_idx in range(num_layers):
    for h_idx in range(num_heads):
        head_dir = setup_folders(l_idx, h_idx)
        
        attention_map = attentions[l_idx][0, h_idx].numpy()
        N = attention_map.shape[0]

        # 3. full.png 저장
        save_attention_map(attention_map, head_dir, 'full.png')

        # 4. TOVA
        tova_attn = np.zeros_like(attention_map)
        tova_attn[-1, :] = attention_map[-1, :]
        save_attention_map(tova_attn, head_dir, 'tova.png', box_start=N-1, box_height=1)

        tova_vector = tova_attn.sum(axis=0, keepdims=True)
        save_vector(tova_vector, head_dir, 'tova_vector.png', cmap='Blues')

        tova_weight = np.zeros((N, 1))
        tova_weight[-1, 0] = 1
        save_vector(tova_weight, head_dir, 'tova_weight.png', cmap='Reds', vmin=0, vmax=1)

        # 5. SnapKV
        window = min(4, N)
        snapkv_attn = np.zeros_like(attention_map)
        snapkv_attn[-window:, :] = attention_map[-window:, :]
        save_attention_map(snapkv_attn, head_dir, 'snapkv.png', box_start=N-window, box_height=window)

        snapkv_vector = snapkv_attn.sum(axis=0, keepdims=True)
        save_vector(snapkv_vector, head_dir, 'snapkv_vector.png', cmap='Blues')

        snapkv_weight = np.zeros((N, 1))
        snapkv_weight[-window:, 0] = 1
        save_vector(snapkv_weight, head_dir, 'snapkv_weight.png', cmap='Reds', vmin=0, vmax=1)

        # 6. H2O
        h2o_attn = attention_map.copy()
        save_attention_map(h2o_attn, head_dir, 'h2o.png', box_start=0, box_height=N)

        h2o_vector = h2o_attn.sum(axis=0, keepdims=True)
        save_vector(h2o_vector, head_dir, 'h2o_vector.png', cmap='Blues')

        h2o_weight = np.ones((N, 1))
        save_vector(h2o_weight, head_dir, 'h2o_weight.png', cmap='Reds', vmin=0, vmax=1)

        # 7. WAITS
        K = 0.8
        waits_attn = np.zeros_like(attention_map)
        waits_weight = np.zeros((N, 1))

        for i in range(N):
            power = (N - 1) - i
            weight_val = K ** power
            waits_attn[i, :] = attention_map[i, :] * weight_val
            waits_weight[i, 0] = weight_val

        save_attention_map(waits_attn, head_dir, 'waits.png', box_start=0, box_height=N)

        waits_vector = waits_attn.sum(axis=0, keepdims=True)
        save_vector(waits_vector, head_dir, 'waits_vector.png', cmap='Blues')

        save_vector(waits_weight, head_dir, 'waits_weight.png', cmap='Reds', vmin=0, vmax=1)

print(f"All images saved in the '{BASE_PLOT_DIR}' directory with layer and head subdirectories.")