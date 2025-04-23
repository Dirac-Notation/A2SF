import ast
import torch
import os
import matplotlib.pyplot as plt

# 파일에서 리스트 형태로 저장된 토큰 인덱스 읽어오기
with open("att10token.txt", "r") as f:
    att_lines = f.readlines()
with open("normaltoken.txt", "r") as f:
    rand_lines = f.readlines()

# (100,32,32,100) 형태의 텐서로 변환
att = torch.tensor([ast.literal_eval(line.strip()) for line in att_lines]) \
           .view(100, 32, 32, 100)
rand = torch.tensor([ast.literal_eval(line.strip()) for line in rand_lines]) \
           .view(100, 32, 32, 100)

def sim(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """
    두 1D 텐서의 top-k 인덱스 집합을 Jaccard 유사도로 비교
    """
    set_a = set(vec_a.tolist())
    set_b = set(vec_b.tolist())
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

# 배치, 레이어, 헤드, 시퀀스 길이
batch_size, n_layers, n_heads, seq_len = att.shape

# 레이어×헤드 유사도 저장
sim_matrix = torch.zeros(n_layers, n_heads)

# 각 (layer, head) 쌍마다 배치 평균 Jaccard 계산
for layer in range(n_layers):
    for head in range(n_heads):
        sims = [
            sim(att[b, layer, head, :50], rand[b, layer, head, :50])
            for b in range(batch_size)
        ]
        sim_matrix[layer, head] = sum(sims) / batch_size

# CPU로 옮겨 NumPy 배열로 변환
heatmap = sim_matrix.cpu().numpy()

# 히트맵 그리기
plt.figure(figsize=(12, 8))
plt.imshow(heatmap, aspect='auto', interpolation='nearest', cmap='viridis')
plt.colorbar(label='Jaccard Similarity')
plt.xlabel('Head', fontsize=12)
plt.ylabel('Layer', fontsize=12)
plt.title('Layer-Head Jaccard Similarity (Batch-averaged)', fontsize=14)
plt.xticks(range(n_heads))
plt.yticks(range(n_layers))
plt.tight_layout()

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/layer_head_similarity_heatmap.png", dpi=300)
plt.show()