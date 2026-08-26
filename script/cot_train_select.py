"""CoT 학습 데이터로 배포 프로토콜(LinUCB, TRUE-bandit) 선택기를 학습하고 액션을 고른다.

CoT는 단일 태스크 셀이므로 context는 상수 특징(단일 셀)이며, bandit은 그 셀의 최적
arm으로 수렴한다. 학습/선택은 전부 학습셋(sid 200-319)에서만 수행하고, 평가셋(0-49)은
건드리지 않는다.

  python script/cot_train_select.py
"""
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


def main():
    rows = []
    for f in glob.glob("result_txt/analysis/long_decoding/cot_train_qwen3-1.7b_*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if len(r.get("action_scores_gt", [])) == len(ACTIONS):
                rows.append(r)
    print(f"학습 샘플 {len(rows)}개, 액션 {len(ACTIONS)}개")

    S = np.array([r["action_scores_gt"] for r in rows], dtype=float)
    print("\n액션별 학습셋 평균 보상:")
    order = np.argsort(-S.mean(0))
    for i in order:
        print(f"  ({ACTIONS[i][0]:g},{ACTIONS[i][1]:3d})  {100*S[:, i].mean():5.1f}%")

    # 배포와 동일: TRUE-bandit LinUCB, seed 0, 64 epoch
    agent = RoutingNeuralUCB(seed=0, actions=ACTIONS)
    rng = np.random.default_rng(0)
    for _ in range(64):
        for i in rng.permutation(len(rows)):
            p = agent.phi("Few Shot", "qa_f1_score")
            a = agent.select(p)
            agent.update(p, a, float(S[i, a]))
    chosen = agent.greedy_action("Few Shot", "qa_f1_score")
    a, b = agent.action_ab(chosen)
    print(f"\n학습된 선택기의 액션: (a={float(a):g}, b={int(b)})  "
          f"[학습셋 평균 {100*S[:, chosen].mean():.1f}%]")
    print(f"참고 - 학습셋 최고 액션: {ACTIONS[order[0]]} ({100*S[:, order[0]].mean():.1f}%)")
    json.dump({"action": [float(a), int(b)], "n_train": len(rows),
               "train_mean": float(S[:, chosen].mean())},
              open("runs/waits_tables/cot_selected_action.json", "w"))
    print(f"{float(a):g}:{int(b)}")


if __name__ == "__main__":
    main()
