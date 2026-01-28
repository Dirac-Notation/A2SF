import torch
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import gc
import sys

from matplotlib import rcParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_model, CompressionConfig
from rouge_score import rouge_scorer

workpath = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------
# 1. Global Style Settings
# ---------------------------------------------------------
rcParams.update({
    "font.family": "serif",
    "figure.figsize": (15, 8),
    "figure.dpi": 150,
    "font.size": 22,
    "axes.labelsize": 26,
    "axes.titlesize": 28,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
    "axes.linewidth": 1.5,
})

# ---------------------------------------------------------
# 2. 데이터 준비 및 프롬프트 구성
# ---------------------------------------------------------
model_name = "llama3"
model2path = json.load(open("config/model2path.json", "r"))
model_path = model2path[model_name]

# 토크나이저는 계속 사용하므로 미리 로드
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\nSection 420 of the Robert T. Stafford Disaster Relief and Emergency Assistance Act ( P.L. 93-288 , hereinafter the Stafford Act) authorizes the President to \"declare\" a Fire Management Assistance Grant (FMAG). The current FMAG system was established by regulation in October of 2001. These grants provide federal assistance for fire suppression activities. This authority has been delegated to the Federal Emergency Management Agency's (FEMA's) Regional Administrators. Once issued, the FMAG declaration authorizes various forms of federal assistance such as the provision of equipment, personnel, and grants to state, local, and tribal governments for the control, management, and mitigation of any fire on certain public or private forest land or grassland that might become a major disaster. This federal assistance requires a cost-sharing component such that state, local, and tribal governments are responsible for 25% of the expenses. This report discusses the most frequently asked questions received by the Congressional Research Service on FMAGs. It addresses questions regarding how FMAGs are requested, how requests are evaluated using thresholds, and the types of assistance provided under an FMAG declaration. FMAGs can be requested by a state when the governor determines that a fire is burning out of control and threatens to become a major disaster. At that point, a request for assistance can be submitted to FEMA. Typically, requests are submitted to the FEMA Regional Administrator. Requests can be submitted any time—day or night—and can be submitted by telephone to expedite the process. Telephone requests must be followed by written confirmation within 14 days of the phone request. Under the Sandy Recovery Improvement Act of 2013 (SRIA, Division B of P.L. 113-2 ), tribes are equivalent to states in their ability to request a major disaster declaration, an emergency declaration, or a request for an FMAG declaration. Note that some tribal land holdings are administered by the federal government and, therefore, receive fire suppression support through the National Interagency Fire Center (NIFC). The NIFC supports interagency \"wildland\" firefighting efforts on federal lands by the U.S. Forest Service, National Weather Service, National Park Service, Bureau of Indian Affairs (BIA), U.S. Fish and Wildlife Service and FEMA's U.S. Fire Administration. Unlike FMAGs, such support generally does not require tribes to reimburse firefighting costs (FMAGs require the state to pay a 25% cost-share). In addition, tribes with their own fire suppression resources may receive reimbursement from BIA for their costs related to fire suppression on tribal lands. The FMAG request should include cost estimates to support the request as well as information about the fire including the size of the fire(s) in acres or square miles, the population of the community (or communities) threatened, the number of persons evacuated (if applicable), weather conditions, and the degree to which state and local resources are committed to this fire and other fires in federal, state, and/or local jurisdictions. The verbal request must be followed up with a completed \"Request for Fire Management Assistance Declaration\" (FEMA form 078-0-1) and the \"Principal Advisor's Report\" (FEMA form 078-0-2). The following criteria are used to evaluate wildfires and make a determination whether to issue an FMAG: the threat to lives and property including critical facilities, infrastructures, and watershed areas; the availability of state and local fire resources; high fire danger conditions based on nationally accepted indices such as the National Fire Danger Ratings System; and the potential economic impacts of the fire. In addition, FEMA has developed fire cost thresholds that are typically updated on an annual basis. There are two types of fire cost thresholds used to help determine if a state or tribal nation is eligible for fire assistance: (1) individual thresholds for a single fire, and (2) cumulative thresholds for multiple fires. Cumulative thresholds are applied to multiple fires burning simultaneously, or the accumulation of multiple fires in a single fire season. Threshold amounts vary by state (see Table 1 ). Taking Pennsylvania as an example, generally, a single fire would need to meet or exceed $927,274 in damages for Pennsylvania to be eligible for an FMAG declaration. In contrast, the formula for the cumulative fire threshold for a given state is one of two amounts—$500,000 or the amount of that state's individual fire threshold multiplied by three, whichever is greater. Returning to the Pennsylvania example, the sum of three individual fire thresholds equals $2,781,822. Since that amount is larger than $500,000, cumulative fire damages in Pennsylvania must meet or exceed $2,781,822 to be eligible for assistance. In contrast, the individual fire threshold for Alaska is $100,000, but the cumulative threshold is $500,000, not the sum of three individual fire thresholds ($300,000). If FEMA denies the request for assistance, the state has one opportunity to appeal the denial. The appeal must be submitted in writing to the Regional Administrator no later than 30 days from the date of the denial letter. The appeal should contain any additional information that strengthens the original request for assistance. The Regional Administrator will review the appeal, prepare a recommendation, and forward the appeal package to the FEMA Headquarters Office. The FEMA Headquarters Office will notify the state of its determination in writing within 90 days of receipt of the appeal (or receipt of additional requested information). The state may request a time extension to submit the appeal. The request for an extension must be submitted in writing to the Regional Administrator no later than 30 days from the date of the denial letter. The request for an extension must include a justification for the need for an extension. The FEMA Headquarters Office will notify the state in writing whether the extension request is granted or denied. No, an emergency or major disaster can be declared after an FMAG declaration has been issued. However, the emergency or major disaster declaration must be initiated by a separate request for assistance by the state or tribal government. FMAGs are funded through FEMA's Disaster Relief Fund (DRF), the main account FEMA uses to provide disaster assistance. The DRF is a no-year account—unused funds from the previous fiscal year are carried over to the next fiscal year. Funds in the DRF fall into two categories. The first category is for disaster relief costs associated with major disasters under the Stafford Act. This category reflects the impact of the Budget Control Act ( P.L. 112-25 , BCA), which allows appropriations to cover the costs incurred as a result of major disasters to be paid through an \"allowable adjustment\" to the discretionary spending limits. The second category is colloquially known as \"base funding.\" Base funding includes activities not tied to major disasters under the Stafford Act. Base funding is scored as discretionary spending that counts against the discretionary spending limits, whereas FMAGs are funded through the DRF's base funding category. The decision to issue a FMAG declaration is not contingent on the DRF balance. Similarly, FMAGs do not reduce the amount of funding available for major disasters. When the DRF balance was low in the past, FEMA used its \"immediate needs funding\" (INF) policy until supplemental appropriations were passed to replenish the DRF. Under INF, long-term projects (such as mitigation work) are put on hold and only activities deemed urgent are funded. FMAGs would most likely fall into the category of events with an \"urgent\" need. Under the INF policy, FEMA also delays interagency reimbursements, and recovers funds from previous years in order to stretch its available funds. As with many other Stafford Act disaster assistance grant programs (Public Assistance, Hazard Mitigation Grant assistance, Other Needs Assistance) the cost-share for FMAGs is based on a federal share of 75% of eligible expenses. The grantee (the state) and subgrantees (local communities) assume the remaining 25% of eligible costs. Under the FMAG process, FEMA reimburses grantees for eligible activities they have undertaken. The state application for specific grant funds must be submitted within 90 days after the FMAG is granted. That time frame permits the state to gather all information and supporting data on potentially eligible spending to include in their grant application package. The package must also stipulate that the fire cost threshold was met. Following submission of the grant application FEMA has 45 days to approve or deny the application. FMAG assistance is similar in some basic respects to other FEMA assistance. For example, FMAGs will not replicate or displace the work of other federal agencies, nor will FEMA pay straight-time salaries for public safety forces, though it will reimburse overtime expenses for the event. Other eligible expenses can include costs for equipment and supplies (less insurance proceeds); mobilization and demobilization; emergency work (evacuations and sheltering, police barricading and traffic control, arson investigation); prepositioning federal, out-of-state, and international resources for up to 21 days when approved by the FEMA Regional Administrator; personal comfort and safety items for firefighter health and safety; field camps and meals in lieu of per diem; and/or the mitigation, management, and control of declared fires burning on comingled federal land, when such costs are not reimbursable by another federal agency. Until recently, only major disaster declarations made statewide hazard mitigation grants available. Division D of P.L. 115-254 (Disaster Recovery Reform Act, hereinafter DRRA) amended the Stafford Act to make hazard mitigation available for FMAG declarations as well. Under Section 404 of the Stafford Act as amended by DRRA, mitigation grants from the Hazard Mitigation Grant Program (HMGP) are provided to states and tribes on a sliding scale based on the percentage of funds spent for FMAG assistance. For states and federally recognized tribes with a FEMA-approved Standard State or Tribal Mitigation Plan, the formula provides for up to 15% of the first $2 billion of estimated aggregate amounts of disaster assistance, up to 10% for amounts between $2 billion and $10 billion, and 7.5% for amounts between $10 billion and $35.333 billion. FEMA assistance through FMAGs is a direct relationship with the states to assist the state in fighting the fire on state lands. FMAGs are employed so a disaster declaration may not be necessary. The Forest Service and other federal agencies do provide other types of assistance related to wildfire management, such as postfire recovery assistance, or assistance planning and mitigating the potential risk from future wildfires. Most of these programs provide financial and technical assistance to state partners. In addition, other USDA agencies administer various other programs to provide disaster recovery assistance to nonfederal forest landowners, including the Emergency Forest Restoration Program and the Emergency Watershed Program. This depends on the type of assistance being provided by the Forest Service. FMAG assistance is not generally available in conjunction with emergency suppression assistance from the Forest Service, or any other federal agency engaged in suppression operations. FMAGs provide assistance for suppression operations on nonfederal lands, whereas suppression operations on federal lands are the responsibility of the federal agency with jurisdiction. Limited exceptions may occur for declared fires on lands in which the ownership is comingled federal and nonfederal, and the costs incurred by the eligible entity are not entitled to any other type of federal reimbursement. However, FMAGs may be provided in conjunction with other Forest Service assistance programs, such as any technical and financial assistance provided through the agency's state and volunteer fire assistance programs or state and private forestry office. FMAG and other federal assistance may potentially occur in conjunction when there is a cooperative agreement between federal, state, and other governmental or tribal partners to coordinate emergency wildfire protection and response activities. The cooperative agreement often delineates different geographic areas where the state government is responsible for initial suppression operations, regardless of land ownership, and vice versa, where the federal government may be responsible for providing suppression operations in lands under nonfederal ownership. The cooperative agreements (sometimes referred to as \"fire compacts\") specify how costs are to be apportioned among the partners, including provisions allowing for reimbursement, in accordance with applicable federal and state statutes. In the circumstance where a state (or other eligible entity) conducted suppression operations on federal land and the costs were not reimbursable, an FMAG may potentially be applied for and used to cover eligible costs. No, most fires that begin on federal land are the responsibility of the federal agency that owns or manages the land, and are not eligible to receive FMAG assistance. There are some exceptions, however. For example, FMAGs may be available to assist with declared fires that occur in areas with a mix of federal and nonfederal land, if the state has a responsibility for suppression activities under a cooperative agreement with the applicable federal agency, and those costs are not reimbursable under another federal statute.\n\nNow, write a one-page summary of the report.\n\nSummary:"
prompt_with_format = f"[INST]{prompt}[/INST]"
print(prompt_with_format)
# 토크나이징 및 위치 매핑
input_enc = tokenizer(prompt_with_format, return_tensors="pt", return_offsets_mapping=True)
offset_mapping = input_enc.offset_mapping[0]

# --- 텍스트 위치 찾기 ---
# Gemini가 뽑은 주요 문장 5개 (프롬프트의 실제 문장과 매칭)
# 실제 프롬프트에서 찾을 핵심 부분들
key_sentence_patterns = [
    # 1. 프로그램 정의: "Section 420... FMAG" + "These grants provide federal assistance for fire suppression activities."
    ("Section 420 of the Robert T. Stafford Disaster Relief and Emergency Assistance Act", "These grants provide federal assistance for fire suppression activities."),
    # 2. 신청 조건: "FMAGs can be requested by a state when the governor determines..."
    ("FMAGs can be requested by a state when the governor determines", "threatens to become a major disaster."),
    # 3. 비용 분담: "This federal assistance requires a cost-sharing component such that state, local, and tribal governments are responsible for 25% of the expenses."
    ("This federal assistance requires a cost-sharing component such that state, local, and tribal governments are responsible for 25% of the expenses", "."),
    # 4. 자격 평가: "There are two types of fire cost thresholds used to help determine if a state or tribal nation is eligible for fire assistance: (1) individual thresholds for a single fire, and (2) cumulative thresholds for multiple fires."
    ("There are two types of fire cost thresholds used to help determine if a state or tribal nation is eligible for fire assistance", "cumulative thresholds for multiple fires."),
    # 5. 지원 확대: "Division D of P.L. 115-254 (Disaster Recovery Reform Act, hereinafter DRRA) amended the Stafford Act to make hazard mitigation available for FMAG declarations as well."
    ("Division D of P.L. 115-254 (Disaster Recovery Reform Act, hereinafter DRRA) amended the Stafford Act to make hazard mitigation available for FMAG declarations as well", "based on the percentage of funds spent for FMAG assistance.")
]

def find_sentence_position(prompt, start_pattern, end_pattern):
    """시작 패턴과 끝 패턴으로 문장 위치 찾기"""
    start_pos = prompt.find(start_pattern)
    if start_pos == -1:
        return -1, -1
    
    # 시작 위치부터 끝 패턴 찾기
    search_start = start_pos
    end_pos = prompt.find(end_pattern, search_start)
    if end_pos == -1:
        # 끝 패턴을 찾지 못하면 다음 문장 끝(.)까지
        end_pos = prompt.find(".", search_start + len(start_pattern))
        if end_pos != -1:
            return start_pos, end_pos + 1
        return start_pos, start_pos + len(start_pattern)
    
    return start_pos, end_pos + len(end_pattern)

# 각 주요 문장의 문자 위치 찾기
key_sentence_ranges = []
for start_pattern, end_pattern in key_sentence_patterns:
    start_char, end_char = find_sentence_position(prompt, start_pattern, end_pattern)
    key_sentence_ranges.append((start_char, end_char))
    if start_char == -1:
        print(f"Warning: Could not find sentence starting with: {start_pattern[:50]}...")

# --- 토큰 인덱스 매핑 ---
key_sentence_token_ranges = []
for start_char, end_char in key_sentence_ranges:
    if start_char == -1:
        key_sentence_token_ranges.append([0, 0])
        continue
    
    token_range = [0, 0]
    found_start = False
    for i, (start, end) in enumerate(offset_mapping):
        if not found_start and start >= start_char:
            token_range[0] = i
            found_start = True
        if start < end_char:
            token_range[1] = i
    key_sentence_token_ranges.append(token_range)

# ---------------------------------------------------------
# Phase 1: Attention Map 분석 및 시각화
# ---------------------------------------------------------
print(">>> Phase 1: Attention Map 분석 시작")

# 분석용 일반 모델 로드
attention_model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
).eval()

input_ids = input_enc.input_ids.to(attention_model.device)
attention_mask = input_enc.attention_mask.to(attention_model.device)

with torch.no_grad():
    output = attention_model(input_ids, attention_mask=attention_mask, output_attentions=True)

attentions = [attn.cpu() for attn in output.attentions]
del output
torch.cuda.empty_cache()
attention_maps = torch.stack(attentions, dim=0).squeeze(1).to(attention_model.device)
seq_len = attention_maps.size(2)

# --- Multi-Window Observation (SNAP Logic) ---
LOCAL_WINDOW = 16
factors = [0.50, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0]
heatmap_data = []

exponents = torch.arange(seq_len-1,-1,-1, device=attention_model.device, dtype=attention_model.dtype)

for w in factors:
    
    forgetting = w ** exponents
    
    forgotten_attention_maps = forgetting.view(1,1,seq_len,1) * attention_maps
    
    accumulated_score = forgotten_attention_maps.sum(dim=2)
    accumulated_score[:,:,-LOCAL_WINDOW:] = accumulated_score.max()
    
    k = 128
    selected_indices = accumulated_score.topk(k=k, dim=-1).indices
    
    count = torch.zeros(seq_len, device=attention_model.device)
    flat_indices = selected_indices.flatten()
    ones = torch.ones_like(flat_indices, dtype=torch.float)
    count.scatter_add_(0, flat_indices, ones)
    
    ratio = count / count.sum()
    heatmap_data.append(ratio.cpu().numpy())

heatmap_matrix = np.array(heatmap_data)

# --- forgetting 곡선 및 히트맵을 factor별 개별 파일로 시각화 ---
# 각 factor마다:
#   - 위: 해당 factor의 forgetting 곡선 (지수형 가중치)
#   - 아래: 현재와 동일한 selection ratio 히트맵 (1 x seq_len)
#
# 파일 이름 예시:
#   snap_w0.35.png, snap_w0.7.png, ...

# 박스 그리기 헬퍼 함수 (각 factor의 히트맵 한 줄에만 그리기)
def add_rect(ax, start, end, label, color, style='-', zorder=5):
    if start == 0 and end == 0:
        return  # 유효하지 않은 범위는 건너뛰기
    rect = patches.Rectangle(
        (start, -0.5), max(end - start, 1), 1,
        linewidth=2, edgecolor=color, facecolor='none',
        linestyle=style, label=label, zorder=zorder, clip_on=False
    )
    ax.add_patch(rect)

# 주요 문장 색/라벨
key_sentence_colors = ['red', 'blue', 'green', 'magenta', 'cyan']
key_sentence_labels = [
    'Key Sentence 1',
    'Key Sentence 2',
    'Key Sentence 3',
    'Key Sentence 4',
    'Key Sentence 5'
]

# forgetting 곡선 (GPU -> CPU numpy) - factor별로 다시 계산
exponents_cpu = torch.arange(seq_len - 1, -1, -1, dtype=torch.float32)
x = np.arange(seq_len)

last_im_for_legend = None

for idx, w in enumerate(factors):
    fig, (top_ax, bottom_ax) = plt.subplots(
        2, 1, figsize=(18, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2]}
    )

    # --- 위: forgetting 곡선 ---
    forgetting = (w ** exponents_cpu).numpy()
    top_ax.plot(x, forgetting, color='black')
    top_ax.set_ylabel(f"w={w}")
    top_ax.set_xlim(0, seq_len - 1)
    top_ax.grid(True, alpha=0.3)

    # --- 아래: selection ratio 히트맵 (해당 factor 한 줄만) ---
    row_heatmap = heatmap_matrix[idx:idx + 1, :]  # shape: (1, seq_len)
    im = bottom_ax.imshow(
        row_heatmap,
        cmap="Blues",
        aspect='auto',
        interpolation='none',
        vmin=0,
        vmax=0.01
    )
    last_im_for_legend = im

    for spine in bottom_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    bottom_ax.set_yticks([])  # y축 눈금 제거 (한 줄 히트맵)
    bottom_ax.set_xlabel("Token Index")
    bottom_ax.set_ylabel("Selection\nRatio")

    # 주요 문장 박스 (현재 factor의 히트맵 위에 표시)
    for (token_range, color, label) in zip(key_sentence_token_ranges, key_sentence_colors, key_sentence_labels):
        add_rect(bottom_ax, token_range[0], token_range[1], label, color, zorder=6)

    # The number of accumulated queries (Green Dashed) - 각 factor별로 해당 window만 표시
    if w == "ALL":
        w_size = seq_len
        start_idx = 0
    else:
        # w가 0~1 사이의 비율 값이므로, 전체 길이에 비례하는 window로 해석
        w_size = int(max(1, min(seq_len, round(w * seq_len))))
        start_idx = seq_len - w_size

    rect_window = patches.Rectangle(
        (start_idx, -0.5), w_size, 1,
        linewidth=2, edgecolor='#2ca02c', facecolor='none', linestyle='--',
        clip_on=False, zorder=10
    )
    bottom_ax.add_patch(rect_window)

    plt.tight_layout()

    # factor별 파일 이름
    safe_w = str(w).replace(".", "_")
    out_path = os.path.join(workpath, f"snap_w{safe_w}.png")
    plt.savefig(out_path)
    plt.close(fig)

# --- 범례를 별도의 파일로 저장 ---
legend_elements = [
    patches.Patch(facecolor='none', edgecolor='#2ca02c', linewidth=2, linestyle='--', label='The number of accumulated queries')
]
for color, label in zip(key_sentence_colors, key_sentence_labels):
    legend_elements.append(
        patches.Patch(facecolor='none', edgecolor=color, linewidth=2, label=label)
    )

fig_legend = plt.figure(figsize=(12, 3))
ax_legend = fig_legend.add_subplot(111)
ax_legend.axis('off')
ax_legend.legend(
    handles=legend_elements,
    loc='center',
    framealpha=0.95,
    fontsize=16,
    ncol=3
)
plt.tight_layout()
plt.savefig(os.path.join(workpath, "snap_legend.png"), bbox_inches='tight', pad_inches=0.2)
plt.close(fig_legend)