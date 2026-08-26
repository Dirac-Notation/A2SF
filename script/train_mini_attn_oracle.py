#!/usr/bin/env python3
"""Train the MiniCrossAttn encoder to predict the future-aware oracle importance.

Target = the decode-time oracle attention each context token receives
(datasets/generate_oracle_labels.py output: per-layer (16, S), head-averaged,
L1-normalised). The encoder reads frozen token embeddings and predicts a
per-token importance whose top-128 should match the oracle's (-> Tanimoto@128).

Loss: listwise cross-entropy between the (optionally tempered) oracle importance
distribution and softmax(pred) over tokens. Rank-sensitive, smooth, matches the
"which tokens to keep" objective directly.

Layer flexibility (--target_layers):
  mean   : average the 16 oracle layers -> (S,)   [RL-compatible single output]
  <i>    : single layer i               -> (S,)
  <i-j>  : mean of layers i..j          -> (S,)
  all    : keep all 16 layers (16, S)   [n_out=16 head-mix readout, experimentation]

  python script/train_mini_attn_oracle.py \
      --train_file datasets/training/scored/faithful_v1/train.jsonl \
      --val_file   datasets/training/scored/faithful_v1/validation.jsonl \
      --oracle_dir runs/oracle_labels/llama3-1b_faithful \
      --target_layers mean --save_dir runs/mini_attn_faithful --gpu 0
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "script"))
from train_mini_attn import MiniCrossAttn  # noqa: E402

_NO_CHAT = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]
def _format_prompt(prompt, dataset_name, model_name):
    if str(dataset_name or "").strip().lower() not in _NO_CHAT:
        if "llama" in str(model_name).lower():
            return f"[INST]{prompt}[/INST]"
    return prompt


def _model_path(model_name):
    with open(os.path.join(REPO_ROOT, "config", "model2path.json")) as f:
        return json.load(f)[model_name]


def parse_target_layers(spec, n_layers=16):
    """Return (reduce_fn(imp16 -> target), n_out, label)."""
    spec = str(spec).strip().lower()
    if spec == "mean":
        return (lambda x: x.mean(0, keepdim=True), 1, "mean")          # (1,S)
    if spec == "all":
        return (lambda x: x, n_layers, "all")                          # (16,S)
    if "-" in spec:
        a, b = spec.split("-"); a, b = int(a), int(b)
        return (lambda x: x[a:b + 1].mean(0, keepdim=True), 1, f"{a}-{b}")
    i = int(spec)
    return (lambda x: x[i:i + 1], 1, f"L{i}")


def normalize_target(t, temp, sink_mask=0):
    """t: (n_out, S) nonneg. Zero attention-sink prefix, temper, L1-normalise."""
    t = t.clamp(min=0).float()
    if sink_mask > 0:
        t = t.clone(); t[:, :sink_mask] = 0.0
    if temp != 1.0:
        t = t.pow(temp)
    return t / (t.sum(-1, keepdim=True) + 1e-12)


def ce_loss(pred, target):
    """Listwise CE between the oracle distribution and the encoder's attention
    distribution. pred is the MiniCrossAttn readout = sum of softmax attention
    over the W observation queries -> already nonneg and ~normalised. We treat it
    AS the predicted distribution (just L1-renormalise); do NOT re-softmax (that
    flattens the tiny per-token values to uniform and kills the gradient)."""
    pred_dist = pred / (pred.sum(-1, keepdim=True) + 1e-9)
    return -(target * (pred_dist + 1e-9).log()).sum(-1).mean()


def bce_loss(pred, target, k=128):
    """Top-k binary cross-entropy: the oracle's top-k tokens are positives, the
    rest negatives. Directly optimises recall@k. pred/target: (n_out, S).
    target is the (sink-zeroed) oracle distribution; its top-k are the keep set.
    """
    S = pred.shape[-1]
    k = min(k, S)
    mask = torch.zeros_like(pred)
    topk_idx = target.topk(k, dim=-1).indices            # (n_out, k)
    mask.scatter_(-1, topk_idx, 1.0)
    pos_weight = torch.tensor(max(1.0, (S - k) / k), device=pred.device)
    return F.binary_cross_entropy_with_logits(pred, mask, pos_weight=pos_weight)


@torch.no_grad()
def topk_metrics(pred, target, k=128, sink_mask=0):
    """Mean Tanimoto and recall@k of top-k sets over n_out rows, sinks excluded
    from both sides (matches the original encoder's sink_mask convention)."""
    S = pred.shape[-1]
    k = min(k, S - sink_mask)
    tan, rec = [], []
    for r in range(pred.shape[0]):
        p = pred[r].clone(); t = target[r].clone()
        if sink_mask > 0:
            p[:sink_mask] = float("-inf"); t[:sink_mask] = float("-inf")
        pi = set(p.topk(k).indices.tolist())
        ti = set(t.topk(k).indices.tolist())
        inter = len(pi & ti)
        tan.append(inter / max(1, len(pi | ti)))
        rec.append(inter / max(1, k))
    return sum(tan) / len(tan), sum(rec) / len(rec)


def load_split(path, oracle_dir, tokenizer, model_name, reduce_fn, temp, sink_mask, max_len, device):
    """-> list of {ids (LongTensor CPU), target (n_out,S) fp16 CPU, dataset}."""
    recs, miss_oracle, miss_align = [], 0, 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = int(d["sample_id"])
            op = os.path.join(oracle_dir, f"{sid}.pt")
            if not os.path.exists(op):
                miss_oracle += 1; continue
            prompt = _format_prompt(d["input_prompt"], d.get("dataset", ""), model_name)
            ids = tokenizer(prompt, truncation=True, max_length=max_len,
                            return_tensors="pt").input_ids[0]
            ol = torch.load(op, map_location="cpu")
            imp = ol["imp_per_layer"].float()              # (16, S_o)
            if imp.shape[-1] != ids.shape[0]:
                miss_align += 1; continue
            target = normalize_target(reduce_fn(imp), temp, sink_mask).to(torch.float16)  # (n_out,S)
            recs.append({"ids": ids, "target": target, "dataset": d.get("dataset")})
    print(f"  {os.path.basename(path)}: {len(recs)} usable "
          f"(skip: no-oracle={miss_oracle}, misalign={miss_align})", flush=True)
    return recs


def compute_loss(pred, target, kind, topk):
    return bce_loss(pred, target, topk) if kind == "bce" else ce_loss(pred, target)


@torch.no_grad()
def evaluate(model, emb, recs, device, kind="bce", k=128, sink_mask=0):
    model.eval()
    tot_loss, tot_tan, tot_rec, n = 0.0, 0.0, 0.0, 0
    for r in recs:
        ids = r["ids"].to(device)
        target = r["target"].float().to(device)
        embeds = emb(ids).to(torch.float32)
        pred = model(embeds)[0]
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        tan, rec = topk_metrics(pred, target, k, sink_mask)
        tot_loss += float(compute_loss(pred, target, kind, k)); tot_tan += tan; tot_rec += rec; n += 1
    return tot_loss / max(1, n), tot_tan / max(1, n), tot_rec / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3-1b")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--val_file", required=True)
    ap.add_argument("--oracle_dir", required=True)
    ap.add_argument("--target_layers", default="mean")
    ap.add_argument("--loss", default="bce", choices=["bce", "ce"],
                    help="bce=top-k binary CE (directly optimises recall@k); "
                         "ce=listwise cross-entropy on the oracle distribution.")
    ap.add_argument("--topk", type=int, default=128, help="positive set size for bce / eval k.")
    ap.add_argument("--target_temp", type=float, default=1.0)
    ap.add_argument("--sink_mask", type=int, default=4,
                    help="exclude first N attention-sink tokens from target+topk "
                         "(matches the original encoder; sinks are trivially kept).")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=32768)
    ap.add_argument("--query_window", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--readout", default="crossattn", choices=["crossattn", "mlp"],
                    help="crossattn=observation-window attention (RL-compatible); "
                         "mlp=free per-token head (diagnostic; use with --loss bce).")
    ap.add_argument("--baseline_ckpt", default="runs/mini_attn_v5/mini_attn_best.pt",
                    help="existing encoder; eval its Tanimoto@128 on val as the bar to beat.")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max_train", type=int, default=0, help="0=all; else cap (smoke test).")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    torch.manual_seed(a.seed)
    device = "cuda"
    os.makedirs(a.save_dir, exist_ok=True)

    reduce_fn, n_out, lbl = parse_target_layers(a.target_layers)
    print(f"target_layers={a.target_layers} -> label={lbl} n_out={n_out} temp={a.target_temp}")

    mp = _model_path(a.model)
    print(f"loading {mp} (frozen embeddings) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(mp)
    tgt = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.bfloat16,
                                               device_map={"": device}).eval()
    emb = tgt.get_input_embeddings()
    for p in tgt.parameters():
        p.requires_grad_(False)
    embed_dim = tgt.config.hidden_size

    print("loading + tokenizing data ...", flush=True)
    tr = load_split(a.train_file, a.oracle_dir, tok, a.model, reduce_fn, a.target_temp, a.sink_mask, a.max_len, device)
    va = load_split(a.val_file, a.oracle_dir, tok, a.model, reduce_fn, a.target_temp, a.sink_mask, a.max_len, device)
    if a.max_train:
        tr = tr[:a.max_train]; va = va[:max(4, a.max_train // 5)]
        print(f"  [smoke] capped to train={len(tr)} val={len(va)}")

    model = MiniCrossAttn(embed_dim=embed_dim, hidden=a.hidden, n_heads=a.n_heads,
                          query_window=a.query_window, max_len=a.max_len, n_out=n_out,
                          readout=a.readout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MiniCrossAttn params: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    # baseline bar (existing ckpt, only meaningful for n_out=1)
    if n_out == 1 and a.baseline_ckpt and os.path.exists(a.baseline_ckpt):
        try:
            base = MiniCrossAttn(embed_dim=embed_dim, hidden=a.hidden, n_heads=a.n_heads,
                                 query_window=a.query_window, max_len=a.max_len, n_out=1).to(device)
            sd = torch.load(a.baseline_ckpt, map_location=device)
            inner = sd.get("model_state_dict") or sd.get("model") or sd
            miss, unexp = base.load_state_dict(inner, strict=False)
            assert not miss and not unexp, f"baseline key mismatch: missing={miss} unexpected={unexp}"
            _, btan, brec = evaluate(base, emb, va, device, kind=a.loss, k=a.topk, sink_mask=a.sink_mask)
            print(f"[baseline {os.path.basename(a.baseline_ckpt)}] val Tanimoto@128={btan:.4f} "
                  f"recall@128={brec:.4f}  (bar to beat; orig reported recall=0.484)")
            del base
        except Exception as e:
            print(f"[baseline] skipped: {e}")

    rng = torch.Generator().manual_seed(a.seed)
    best_rec, log = -1.0, []
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr), generator=rng).tolist()
        t0 = time.time(); run_loss = 0.0; opt.zero_grad()
        for i, idx in enumerate(perm):
            r = tr[idx]
            ids = r["ids"].to(device); target = r["target"].float().to(device)
            embeds = emb(ids).to(torch.float32)
            pred = model(embeds)[0]
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
            loss = compute_loss(pred, target, a.loss, a.topk) / a.accum
            loss.backward(); run_loss += loss.item() * a.accum
            if (i + 1) % a.accum == 0 or (i + 1) == len(perm):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
        vl, vtan, vrec = evaluate(model, emb, va, device, kind=a.loss, k=a.topk, sink_mask=a.sink_mask)
        tr_loss = run_loss / len(tr)
        log.append({"epoch": ep, "train_loss": tr_loss, "val_loss": vl,
                    "val_tanimoto": vtan, "val_recall": vrec})
        flag = ""
        if vrec > best_rec:
            best_rec = vrec
            torch.save({"model_state_dict": model.state_dict(),
                        "config": {"embed_dim": embed_dim, "hidden": a.hidden, "n_heads": a.n_heads,
                                   "query_window": a.query_window, "max_len": a.max_len, "n_out": n_out,
                                   "target_layers": a.target_layers, "target_temp": a.target_temp,
                                   "sink_mask": a.sink_mask},
                        "epoch": ep, "val_recall": vrec, "val_tanimoto": vtan},
                       os.path.join(a.save_dir, "mini_attn_best.pt"))
            flag = " *best"
        print(f"ep{ep:02d} train_ce={tr_loss:.4f} val_ce={vl:.4f} "
              f"val_tan@128={vtan:.4f} val_rec@128={vrec:.4f}{flag} ({time.time()-t0:.0f}s)", flush=True)
        with open(os.path.join(a.save_dir, "train.log.json"), "w") as f:
            json.dump({"best_recall": best_rec, "n_out": n_out, "target": lbl, "log": log}, f, indent=2)
    print(f"done. best val recall@128 = {best_rec:.4f} -> {a.save_dir}/mini_attn_best.pt")


if __name__ == "__main__":
    main()
