"""CoT용 학습 데이터 생성: 프롬프트별로 13개 (a,b) 액션 각각의 보상(정답=1/오답=0)을
RefreshKV 방식 스트리밍 압축 하에서 채점한다. RL/dataset.py의 CoT 버전.

출력: result_txt/analysis/long_decoding/cot_train_<model>.jsonl
  {sample_id, question, gt, action_scores_gt: [13개], gen_lens: [13개]}

  python script/gsm8k_cot_dataset.py --model qwen3-1.7b --gpus 0,1,2,3,4,5,6,7 \
      --budget 512 --start_idx 200 --n_samples 120
"""
import argparse
import json
import os
import sys
import multiprocessing as mp

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "script"))

import importlib.util
_spec = importlib.util.spec_from_file_location("st", os.path.join(REPO, "script/gsm8k_cot_stream.py"))
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)

from utils import load_model  # noqa: E402
from RL.action_grid import SIGMOID_A_VALUES, SIGMOID_B_VALUES  # noqa: E402

ACTIONS = [(float(a), int(b)) for a, b in zip(SIGMOID_A_VALUES, SIGMOID_B_VALUES)]


def _worker(gpu, model_name, budget, shard, rq, acts=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.set_grad_enabled(False)
    model, tok = load_model(model_name)
    dev = next(model.parameters()).device
    for row in shard:
        prompt = tok.apply_chat_template([{"role": "user", "content": row["q"]}],
                                         tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        scores, lens = [], []
        for (a, b) in (acts or ACTIONS):
            cfg = st.build_cfg("waits", budget, a=a, b=b)
            try:
                txt = st.stream_generate(model, tok, ids, cfg, refresh=True)
                ans = st.extract_number(txt)
                scores.append(1.0 if ans == row["gt"] else 0.0)
                lens.append(len(txt.split()))
            except Exception:
                scores.append(0.0); lens.append(0)
        key = "action_scores_gt" if acts is None else "action_scores_partial"
        out = {"sample_id": row["sid"], "question": row["q"], "gt": row["gt"],
               "prompt_tokens": int(ids.shape[1]), key: scores, "gen_lens": lens}
        if acts is not None:
            out["actions"] = [[float(a), int(b)] for a, b in acts]
        rq.put(out)
    rq.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--start_idx", type=int, default=200)
    ap.add_argument("--n_samples", type=int, default=120)
    ap.add_argument("--tag", default="")
    ap.add_argument("--sids", default=None, help="쉼표 구분 sid 목록 (start_idx/n_samples 대신)")
    ap.add_argument("--actions", default=None, help="a:b,a:b,... 부분 액션만 채점 (열 인덱스는 13-grid 기준 유지)")
    args = ap.parse_args()

    from datasets import load_dataset
    test = load_dataset("openai/gsm8k", "main", split="test")
    _idx = ([int(x) for x in args.sids.split(",")] if args.sids
            else list(range(args.start_idx, min(args.start_idx + args.n_samples, len(test)))))
    rows = [{"sid": i, "q": test[i]["question"],
             "gt": test[i]["answer"].split("####")[-1].strip().replace(",", "")}
            for i in _idx]

    outdir = os.path.join(REPO, "result_txt/analysis/long_decoding")
    os.makedirs(outdir, exist_ok=True)
    out_p = os.path.join(outdir, f"cot_train_{args.model}{args.tag}.jsonl")
    done = {json.loads(l)["sample_id"] for l in open(out_p)} if os.path.exists(out_p) else set()
    todo = [r for r in rows if r["sid"] not in done]
    _n_act = len(args.actions.split(",")) if args.actions else len(ACTIONS)
    print(f"{len(todo)}/{len(rows)} to score, {_n_act} actions", flush=True)

    gpus = [int(x) for x in args.gpus.split(",")]
    shards = [todo[i::len(gpus)] for i in range(len(gpus))]
    ctx = mp.get_context("spawn")
    rq = ctx.Queue()
    _acts = ([(float(x.split(":")[0]), int(x.split(":")[1])) for x in args.actions.split(",")]
             if args.actions else None)
    procs = [ctx.Process(target=_worker, args=(g, args.model, args.budget, shards[i], rq, _acts))
             for i, g in enumerate(gpus)]
    for p in procs:
        p.start()
    alive, n = len(procs), 0
    with open(out_p, "a") as f:
        while alive:
            item = rq.get()
            if item is None:
                alive -= 1
                continue
            f.write(json.dumps(item) + "\n"); f.flush(); n += 1
            if n % 5 == 0:
                print(f"[cot-data] {n}/{len(todo)}", flush=True)
    for p in procs:
        p.join()
    print("COT_DATA_DONE", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
