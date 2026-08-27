# iclr/ — Per-head/Per-sample (a,b) feasibility (ICLR 에포크)

원점 테제: 스코어링은 sigmoid 망각 곡선 w[q]=σ(a(q−(N−b−0.5)))로 하고,
남은 문제는 (a,b)를 어떻게 설정하는가. KVP(Apple)의 Input(k,v,pos)/Reward
(미래-attention 전-budget AUC, label-free)를 차용해 문장×헤드 선택의 가치를 측정.

## 파일

| 파일 | 역할 |
|---|---|
| `trace_dump.py` | 태스크-free trace 수집 (K/V + 미래-attention u + 17후보 스코어). 커스텀 attention 캡처 — **인과 마스크 자체 처리 필수** (v5 계약, 2026-08-27 사고 참조) |
| `build_rewards.py` | trace → (doc, action, layer, head) AUC-ratio 보상 행렬. 동점 랜덤 파기 |
| `eval_lb.py` | LB 사다리 2-pass 드라이버. pass1 = full-cache 미래-attention 관측(스토어 full pred teacher-forcing, 단일 forward) → 샘플×헤드 오라클 배정 + 보상 저장. pass2 = 배정표로 압축 추론 |
| `run_ladder.sh` | 16셋 × 행 러너. **실행 직전 빈 GPU 동적 스캔** (타 사용자 경합 방지) |
| `analyze_ladder.py` | 사다리 결과 분석 (행 비교, 헤드 선택 구조) |
| `_parked/train_agent.py` | 헤드별 (k,v,pos)→액션 에이전트 (배치화 grouped-MLP). **보류** — 보상 profile이 평평해 학습 무의미 판정. oracle 결과에 따라 재개 |
| `runs/status.json` | 유효 수치·판정 스냅샷 |

## 규칙 (사고에서 나온 것)

1. **13-grid 균일 곡선 × LB는 스토어 조회** (`result_txt/backup/fast_store/`) — 재실행 금지. full-cache pred도 스토어에 있음
2. 커스텀 attention 함수는 mask=None일 때 인과 처리를 직접 — compress.py 패턴 복사. 스모크는 긴 컨텍스트 + 생성문 육안 확인 포함
3. GPU는 실행 직전 동적 스캔 — 고정 리스트 금지 (GPU0 seulkee, GPU2 경합 사고)
4. 사다리 중단 시 run_ladder **서브셸(부모)부터 kill** — 자식만 죽이면 다음 데이터셋 재발사됨

## 현 상태 (2026-08-27)

- 유효: feas1b_fixed 25.73 / feas8b_fixed 35.86 (스토어 정합 |Δ|=0.35 검증)
- wikitext 보상 profile: 상위 13개 액션 등가(5위 regret 1.2~1.6%), 꼬리만 불량 — 1B/8B 동일
- 진행 대기: **8B oracle 행** (pass1 캐시 /data2/smp9898/iclr_traces/lb_pass1/, 92/800 축적)
- 재개: `ROW=oracle N=50 MODEL=llama3-8b TAG=feas8b bash iclr/run_ladder.sh`

데이터: /data2/smp9898/iclr_traces/{llama3-1b,llama3-8b}(trace+rewards), lb_pass1/(LB 보상·배정 캐시)
그림: result_txt/analysis/perhead_profile/ · 계획판/미팅 자료 URL은 memory(project_iclr_perhead_sigmoid) 참조
