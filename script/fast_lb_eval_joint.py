"""Fast LB eval for joint-trained ChunkAttnEncoder + NeuralUCBAgent.

Encodes each LB sample on-the-fly with ChunkAttnEncoder, then looks up
scores from index.pt. Output format identical to fast_lb_eval.py.

Usage:
    python script/fast_lb_eval_joint.py \\
        --joint_ckpt runs/joint_v0/best.pt \\
        --run_name joint_v0 \\
        --model llama3-1b \\
        --gpu 0
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from RL.a2sf_model import SIGMOID_A_VALUES, SIGMOID_B_VALUES, ModelConfig
from RL.agent.neural_ucb_agent import NeuralUCBAgent
from RL.env.chunk_attn_encoder import ChunkAttnEncoder
from RL.env.encoder import (
    METRIC_TYPE_ORDER, TASK_TYPE_ORDER,
    metric_type_to_index, task_type_to_index,
)
from longbench_eval import evaluate_results

INDEX_PATH = "runs/fast_lb_eval/index.pt"
META_DIM   = 1 + len(METRIC_TYPE_ORDER) + len(TASK_TYPE_ORDER)


def build_meta(ex: dict, ds: str, max_seq_len: float = 131072.0) -> torch.Tensor:
    from longbench_eval import dataset2metric
    length   = int(ex.get("length", 0))
    seq_feat = min(float(length), max_seq_len) / max_seq_len

    metric_fn = dataset2metric.get(ds)
    metric_type = metric_fn.__name__ if metric_fn else "qa_f1_score"
    m_idx = metric_type_to_index(metric_type)
    m_oh  = torch.zeros(len(METRIC_TYPE_ORDER))
    m_oh[m_idx] = 1.0

    t_idx = task_type_to_index(task_type=None, dataset=ds)
    t_oh  = torch.zeros(len(TASK_TYPE_ORDER))
    t_oh[t_idx] = 1.0

    return torch.cat([torch.tensor([seq_feat]), m_oh, t_oh])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--joint_ckpt", required=True)
    p.add_argument("--run_name",   required=True)
    p.add_argument("--model",      default="llama3-1b")
    p.add_argument("--budget",     type=int, default=128)
    p.add_argument("--gpu",        default="0")
    p.add_argument("--index_path", default=INDEX_PATH)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    output_dir = f"result_txt/pred/{args.budget}/{args.run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # ── Load joint checkpoint ─────────────────────────────────────────
    print(f"Loading checkpoint: {args.joint_ckpt}")
    ckpt = torch.load(args.joint_ckpt, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]

    # ── Load target model (frozen, for embed_tokens) ──────────────────
    import warnings, logging
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)

    print(f"Loading {args.model} …")
    from RL.env.model_runner import A2SFModelRunner
    mc = ModelConfig.sigmoid(model=args.model)
    runner = A2SFModelRunner(mc)
    target_model = runner.model
    target_tokenizer = runner.tokenizer
    for p in target_model.parameters():
        p.requires_grad_(False)
    target_model.eval()

    d_model = int(target_model.config.hidden_size)

    # ── Build encoder ─────────────────────────────────────────────────
    encoder = ChunkAttnEncoder(
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        d_model=d_model,
        hidden=arch["hidden"],
        n_heads=arch["n_heads"],
        chunk_size=arch["chunk_size"],
        max_chunks=arch["max_chunks"],
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    # ── Build agent ───────────────────────────────────────────────────
    agent = NeuralUCBAgent(
        state_dim        = arch["state_dim"],
        a_values         = arch["a_values"],
        b_values         = arch["b_values"],
        num_metric_types = arch["num_metric_types"],
        num_task_types   = arch["num_task_types"],
        side_dim         = arch["hidden"],
        num_heads        = 1,
        backbone_depth   = 2,
        dropout          = 0.0,
        paired_actions   = True,
        num_hidden_pool  = 0,
        task_cond_head   = True,
        num_views        = 1,
    ).to(device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    print(f"Encoder: {sum(p.numel() for p in encoder.parameters()):,} params")
    print(f"Agent:   {sum(p.numel() for p in agent.parameters()):,} params")

    # ── Load index ────────────────────────────────────────────────────
    print(f"Loading index: {args.index_path}")
    index_d  = torch.load(args.index_path, map_location="cpu", weights_only=False)
    datasets = index_d["datasets"]

    # ── Eval loop ─────────────────────────────────────────────────────
    with open("config/dataset2maxlen.json") as f:
        d2l = json.load(f)

    with torch.no_grad():
        for ds in datasets:
            scores_mat = index_d[f"{ds}/scores"].float()   # (N, 13)
            preds_mat  = index_d[f"{ds}/preds"]            # list[N][13]
            answers_l  = index_d[f"{ds}/answers"]
            all_cls_l  = index_d.get(f"{ds}/all_classes",
                                     [None] * scores_mat.size(0))
            lengths_l  = index_d.get(f"{ds}/lengths",
                                     [None] * scores_mat.size(0))
            N = scores_mat.size(0)

            lb_path = os.path.join("datasets", "longbench", f"{ds}.jsonl")
            with open(lb_path) as f:
                lb_rows = [json.loads(l) for l in f if l.strip()]

            out_path = os.path.join(output_dir, f"{ds}.jsonl")
            ds_score_sum = 0.0
            with open(out_path, "w", encoding="utf-8") as fout:
                for i, ex in enumerate(lb_rows):
                    # Encode prompt
                    prompt = ex.get("input_prompt", "")
                    encoded = encoder.encode_context(
                        text=prompt, detach=True,
                    ).to(device)                           # (128,)
                    meta  = build_meta(ex, ds).to(device)
                    state = torch.cat([meta, encoded]).unsqueeze(0)  # (1, 146)

                    pred_r   = agent.forward(state)["reward_pred"].squeeze(0)  # (13,)
                    act_idx  = int(pred_r.argmax().item())
                    a_val    = float(SIGMOID_A_VALUES[act_idx])
                    b_val    = float(SIGMOID_B_VALUES[act_idx])

                    pred_text = str(preds_mat[i][act_idx]) \
                                if preds_mat[i][act_idx] is not None else ""
                    ans_i    = answers_l[i] if i < len(answers_l) else []
                    cls_i    = all_cls_l[i] if i < len(all_cls_l) else None
                    len_i    = lengths_l[i] if i < len(lengths_l) else None

                    record = {
                        "pred":        pred_text,
                        "answers":     ans_i,
                        "all_classes": cls_i,
                        "length":      int(len_i) if len_i is not None else None,
                        "a":           a_val,
                        "b":           int(round(b_val)),
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    ds_score_sum += float(scores_mat[i, act_idx].item())

            ds_avg = ds_score_sum / N
            print(f"  {ds:25s}  {ds_avg:.2f}", flush=True)

    evaluate_results(output_dir)


if __name__ == "__main__":
    main()
