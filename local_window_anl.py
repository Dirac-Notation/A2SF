import re
import pandas as pd
import matplotlib.pyplot as plt

# 로그 파일 열기
with open("local_window_test.txt", "r") as f:
    log = f.read()

# 정규표현식으로 필요한 정보 추출
pattern = re.compile(
    r"layer=(\d+), local_window_size=([\d.]+).*?ROUGE-1:\s*([\d.]+)",
    re.DOTALL
)
matches = pattern.findall(log)

# 정리
data = [(int(layer), float(window), float(rouge1)) for layer, window, rouge1 in matches]
df = pd.DataFrame(data, columns=["Layer", "Local_Window_Size", "ROUGE-1"])

# 피벗 테이블 (행: Layer, 열: Window Size)
pivot_df = df.pivot(index="Layer", columns="Local_Window_Size", values="ROUGE-1")
pivot_df = pivot_df.sort_index().sort_index(axis=1)

# 서브플롯 생성 (32개 = 8x4)
fig, axes = plt.subplots(8, 4, figsize=(20, 16))
fig.suptitle("ROUGE-1 by Local Window Size per Layer (highlight: max point)", fontsize=16)

# 서브플롯에 각 Layer별 선그래프 그리기
for idx, (layer, row) in enumerate(pivot_df.iterrows()):
    ax = axes[idx // 4][idx % 4]
    x = row.index
    y = row.values

    # 기본 선그래프
    ax.plot(x, y, marker='o', label='ROUGE-1')

    # 최고점 하이라이트
    max_idx = y.argmax()
    ax.plot(x[max_idx], y[max_idx], marker='o', color='red', markersize=8, label='Max ROUGE-1')

    # 그래프 설정
    ax.set_title(f"Layer {layer}", fontsize=10)
    ax.set_xlabel("Local Window Size")
    ax.set_ylabel("ROUGE-1")
    ax.grid(True)

# 여백 조정 및 저장
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("rouge1_lineplots_highlighted_by_layer.png", dpi=300)
plt.close()


# import re
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # 로그 파일 열기
# with open("local_window_test.txt", "r") as f:
#     log = f.read()

# # 정규표현식으로 필요한 정보 추출
# pattern = re.compile(
#     r"layer=(\d+), local_window_size=([\d.]+).*?ROUGE-1:\s*([\d.]+)",
#     re.DOTALL
# )
# matches = pattern.findall(log)

# # 정리
# data = [(int(layer), float(window), float(rouge1)) for layer, window, rouge1 in matches]
# df = pd.DataFrame(data, columns=["Layer", "Local_Window_Size", "ROUGE-1"])

# # 피벗 테이블 (행: Layer, 열: Window Size)
# pivot_df = df.pivot(index="Layer", columns="Local_Window_Size", values="ROUGE-1")
# pivot_df = pivot_df.sort_index().sort_index(axis=1)

# # 데이터 준비
# data_matrix = pivot_df.values
# layers = pivot_df.index.to_list()
# windows = pivot_df.columns.to_list()

# fig, ax = plt.subplots(figsize=(14, 10))
# cax = ax.imshow(data_matrix, cmap='Blues_r', aspect='auto')

# # 컬러바 추가
# cbar = fig.colorbar(cax, ax=ax)
# cbar.set_label('ROUGE-1')

# # 축 라벨 설정
# ax.set_xticks(np.arange(len(windows)))
# ax.set_yticks(np.arange(len(layers)))
# ax.set_xticklabels(windows)
# ax.set_yticklabels(layers)
# ax.set_xlabel('Local Window Size')
# ax.set_ylabel('Layer')
# ax.set_title('ROUGE-1 by Local Window Size per Layer (highlight: best per layer)')

# # 셀마다 값 출력 및 최고점 하이라이트
# for i in range(len(layers)):
#     row = data_matrix[i]
#     max_j = np.argmax(row)
#     for j in range(len(windows)):
#         value = row[j]
#         ax.text(j, i, f'{value:.3f}', ha='center', va='center', color='black')
#         if j == max_j:
#             # 최고값 셀 테두리 하이라이트
#             rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2)
#             ax.add_patch(rect)

# plt.tight_layout()
# plt.savefig("rouge1_heatmap_by_layer.png", dpi=300)
# plt.close()
