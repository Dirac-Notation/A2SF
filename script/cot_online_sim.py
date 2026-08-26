"""CoT 온라인 적응 시뮬레이션 (스트림 전체 정확도).

sid 0-199를 순서대로 추론하되, 앞 N개는 UCB(beta>0)로 탐색하며 선택기를 갱신하고 이후는
beta=0 greedy 액션 하나로 고정한다. 보고 값은 탐색 손실을 포함한 200개 전체 정확도다.
탐색 구간의 각 문제는 그 시점에 시도 중인 arm 하나로만 푼다(실제 온라인 상황과 동일).

동점은 무작위로 깬다. 미탐색 arm의 UCB가 전부 정확히 sqrt(2)로 같아서, argmax를 쓰면
액션 그리드 나열 순서가 탐색 순서를 결정해 버리기 때문이다(a 오름차순이라 CoT에 유리한
완만한 곡선이 앞에 온다). seed마다 순서가 달라지므로 여러 seed의 평균과 표준편차를 낸다.

보상 행렬 조회는 그 자리에서 추론하는 것과 결과가 동일하다(사전계산).

  python script/cot_online_sim.py --seeds 500
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from RL.model import RoutingNeuralUCB  # noqa: E402
from RL.action_grid import SIGMOID_A_VALUES, SIGMOID_B_VALUES  # noqa: E402

ACTIONS = [(float(a), int(b)) for a, b in zip(SIGMOID_A_VALUES, SIGMOID_B_VALUES)]
DIR = "result_txt/analysis/long_decoding"
POOL = f"{DIR}/cot_train_qwen3-1.7b_ev512.jsonl"
STREAM = 200
TASK, METRIC = "Few Shot", "qa_f1_score"
SNAPKV = ACTIONS.index((10.0, 16))


def load_matrix():
    """sid -> {action_idx: reward}. 13열 전수 파일과 부분 채점 파일을 합친다."""
    M = {}
    for f in [POOL] + sorted(glob.glob(f"{DIR}/cot_train_qwen3-1.7b_pv*.jsonl")
                             + glob.glob(f"{DIR}/cot_train_qwen3-1.7b_q*.jsonl")):
        if not os.path.exists(f):
            continue
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            d = M.setdefault(r["sample_id"], {})
            if len(r.get("action_scores_gt", [])) == len(ACTIONS):
                for i, v in enumerate(r["action_scores_gt"]):
                    d[i] = float(v)
            if "action_scores_partial" in r:
                for (a, b), v in zip(r["actions"], r["action_scores_partial"]):
                    d[ACTIONS.index((float(a), int(b)))] = float(v)
    return M


def load_full():
    """무압축(Full) 정확도. 압축과 무관하므로 기존 스트림 결과를 재사용."""
    acc = {}
    for f in glob.glob(f"{DIR}/gsm8k_stream_qwen3-1.7b_rf1024*.jsonl"):
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            if "full" in r.get("preds", {}):
                acc[r["sid"]] = float(r["preds"]["full"]["correct"])
    return acc


def run(M, n, seed, beta):
    """앞 n개 탐색 -> greedy 고정. (전체 정확도, 탐색 정확도, greedy 액션) 반환."""
    rng = np.random.default_rng(seed)
    ag = RoutingNeuralUCB(beta=beta, seed=seed, actions=ACTIONS)
    p = ag.phi(TASK, METRIC)
    hits = 0.0
    for t in range(n):
        u = np.array([ag.theta[k] @ p + ag.beta * np.sqrt(max(p @ ag.Ainv[k] @ p, 0.0))
                      for k in range(len(ACTIONS))])
        a = int(rng.choice(np.flatnonzero(u >= u.max() - 1e-12)))   # 동점 랜덤 파기
        hits += M[t][a]
        ag.update(p, a, float(M[t][a]))
    th = np.array([ag.theta[k] @ p for k in range(len(ACTIONS))])
    g = int(rng.choice(np.flatnonzero(th >= th.max() - 1e-12)))
    tail = sum(M[s][g] for s in range(n, STREAM))
    return (hits + tail) / STREAM, hits / n, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_list", default="10,20,50,100")
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--beta", type=float, default=1.0)
    args = ap.parse_args()

    M = load_matrix()
    FULL = load_full()
    ns = [int(x) for x in args.n_list.split(",")]

    miss = [(s, a) for s in range(STREAM) for a in range(len(ACTIONS)) if a not in M.get(s, {})]
    if miss:
        sids = sorted({s for s, _ in miss})
        acts = sorted({a for _, a in miss})
        print(f"보상 행렬 누락: sid {len(sids)}개 (범위 {min(sids)}-{max(sids)}), "
              f"액션 {len(acts)}종 -> {','.join(f'{ACTIONS[a][0]:g}:{ACTIONS[a][1]}' for a in acts)}")
        print("완료 후 다시 실행하십시오.")
        return

    print(f"=== Qwen3-1.7B thinking CoT, GSM8K 200문항 스트림, budget 512 "
          f"(seed {args.seeds}개, 동점 랜덤 파기) ===\n")
    from collections import Counter
    rows = []
    for n in ns:
        accs, exps, picks = [], [], Counter()
        for sd in range(args.seeds):
            acc, ex, g = run(M, n, sd, args.beta)
            accs.append(acc); exps.append(ex); picks[g] += 1
        rows.append((n, np.mean(accs), np.std(accs), np.mean(exps), picks))
        top = ", ".join(f"({ACTIONS[a][0]:g},{ACTIONS[a][1]}) {100*c/args.seeds:.0f}%"
                        for a, c in picks.most_common(3))
        print(f"N={n:3d}: 전체 {100*np.mean(accs):.1f} +- {100*np.std(accs):.1f}%  "
              f"(탐색 구간 {100*np.mean(exps):.1f}%)  선택 액션 {top}")

    fa = np.mean([FULL[s] for s in range(STREAM) if s in FULL]) if FULL else None
    snap = np.mean([M[s][SNAPKV] for s in range(STREAM)])
    print(f"\nFull {100*fa:.1f}% / SnapKV {100*snap:.1f}%" if fa is not None
          else f"\nSnapKV {100*snap:.1f}%")

    print("\n| 정확도 (%) | Full | SnapKV | " + " | ".join(f"Ours N={n}" for n, *_ in rows) + " |")
    print("|" + "---|" * (3 + len(rows)))
    print(f"| Budget 512 | {100*fa:.1f} | {100*snap:.1f} | "
          + " | ".join(f"{100*m:.1f} ± {100*s:.1f}" for _, m, s, _, _ in rows) + " |")


if __name__ == "__main__":
    main()
