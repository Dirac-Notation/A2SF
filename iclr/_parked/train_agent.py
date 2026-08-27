"""Step 3: train the per-head (k, v, pos) -> (a, b) agents.

All L*H head agents are trained simultaneously as ONE batched module
(per-head weight tensors, einsum grouped linear) - no python loop over heads.

Input per token:  [k (D) | v (D) | pos/N (1)]
Per-head net:     token MLP -> mean||max pool -> MLP -> C logits (17 actions)
Loss:             soft cross-entropy toward softmax(-ratio / tau)
                  (full feedback: every action's reward is known offline)

Validation (held-out docs):
  regret  = ratio(chosen) - ratio(best)          per (doc, head), averaged
  vs fixed = regret of the single globally-best action (the baseline to beat)

Usage:
  python iclr/train_agent.py --dump_dir /data2/smp9898/iclr_traces/llama3-1b \
      --out iclr/runs/agent_1b --epochs 30 --gpu 0
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedLinear(nn.Module):
    """Independent linear layer per head-group: weight [G, in, out]."""

    def __init__(self, groups, d_in, d_out):
        super().__init__()
        self.w = nn.Parameter(torch.randn(groups, d_in, d_out) * (d_in ** -0.5))
        self.b = nn.Parameter(torch.zeros(groups, d_out))

    def forward(self, x):  # x: [G, ..., d_in]
        return torch.einsum("g...i,gio->g...o", x, self.w) + \
            self.b.view(self.b.shape[0], *([1] * (x.dim() - 2)), -1)


class HeadAgents(nn.Module):
    """G = L*H independent agents, batched."""

    def __init__(self, groups, d_feat, n_actions, d_tok=64, d_hid=256):
        super().__init__()
        self.tok1 = GroupedLinear(groups, d_feat, d_tok)
        self.tok2 = GroupedLinear(groups, d_tok, d_tok)
        self.head1 = GroupedLinear(groups, 2 * d_tok, d_hid)
        self.head2 = GroupedLinear(groups, d_hid, n_actions)

    def forward(self, feats):          # feats: [G, N, d_feat]
        e = F.relu(self.tok1(feats))
        e = F.relu(self.tok2(e))       # [G, N, d_tok]
        pooled = torch.cat([e.mean(1), e.max(1).values], -1)  # [G, 2*d_tok]
        return self.head2(F.relu(self.head1(pooled)))          # [G, C]


def doc_features(z, device):
    """npz -> [G, N, 2D+1] fp32 (G = L*H, flattened)."""
    K = torch.from_numpy(z["K"]).to(device, torch.float32)  # [L,H,N,D]
    V = torch.from_numpy(z["V"]).to(device, torch.float32)
    L, H, N, D = K.shape
    pos = torch.arange(N, device=device, dtype=torch.float32) / N
    pos = pos.view(1, 1, N, 1).expand(L, H, N, 1)
    f = torch.cat([K, V, pos], -1).view(L * H, N, 2 * D + 1)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    rw = np.load(os.path.join(args.dump_dir, "rewards.npz"))
    ratio = torch.from_numpy(rw["ratio"]).float()            # [Dn, C, L, H]
    doc_ids = rw["doc_ids"].tolist()
    cands = rw["candidates"].tolist()
    Dn, C, L, H = ratio.shape
    G = L * H
    ratio_g = ratio.permute(0, 2, 3, 1).reshape(Dn, G, C)    # [Dn, G, C]

    files = {int(os.path.basename(f)[4:8]): f
             for f in glob.glob(os.path.join(args.dump_dir, "doc_*.npz"))}
    n_val = max(1, int(Dn * args.val_frac))
    val_idx = list(range(Dn - n_val, Dn))
    tr_idx = list(range(Dn - n_val))
    print(f"[train] docs {Dn} (train {len(tr_idx)} / val {n_val}), G={G}, C={C}")

    # baseline: globally best single action on TRAIN, its regret on VAL
    tr_mean = ratio_g[tr_idx].mean((0, 1))                   # [C]
    fixed_best = int(tr_mean.argmin())
    val_r = ratio_g[val_idx]                                 # [nv, G, C]
    best_val = val_r.min(-1).values
    fixed_regret = (val_r[:, :, fixed_best] - best_val).mean().item()
    orac_regret = 0.0
    print(f"[train] fixed-best action = {cands[fixed_best]}  "
          f"val regret(fixed) = {fixed_regret:.5f}")

    z0 = np.load(files[doc_ids[0]])
    d_feat = 2 * z0["K"].shape[-1] + 1
    model = HeadAgents(G, d_feat, C).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    log = []
    best_val_regret = 1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.RandomState(ep).permutation(tr_idx)
        tot = 0.0
        for di in perm:
            z = np.load(files[doc_ids[di]])
            feats = doc_features(z, device)                  # [G, N, F]
            logits = model(feats)                            # [G, C]
            target = F.softmax(-ratio_g[di].to(device) / args.tau, -1)
            loss = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()

        # validation regret
        model.eval()
        regs, accs = [], []
        with torch.no_grad():
            for di in val_idx:
                z = np.load(files[doc_ids[di]])
                logits = model(doc_features(z, device))      # [G, C]
                choice = logits.argmax(-1).cpu()             # [G]
                r = ratio_g[di]
                reg = r.gather(1, choice[:, None])[:, 0] - r.min(-1).values
                regs.append(reg.mean().item())
                accs.append((choice == r.argmin(-1)).float().mean().item())
        vreg, vacc = float(np.mean(regs)), float(np.mean(accs))
        log.append({"epoch": ep, "loss": tot / len(perm),
                    "val_regret": vreg, "val_top1": vacc})
        marker = ""
        if vreg < best_val_regret:
            best_val_regret = vreg
            torch.save({"state_dict": model.state_dict(), "cands": cands,
                        "d_feat": d_feat, "G": G, "L": L, "H": H, "C": C},
                       os.path.join(args.out, "agent_best.pt"))
            marker = "  *best"
        print(f"[train] ep{ep:03d} loss={tot/len(perm):.4f} "
              f"val_regret={vreg:.5f} (fixed {fixed_regret:.5f}) "
              f"top1={vacc:.3f}{marker}", flush=True)

    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump({"log": log, "fixed_regret": fixed_regret,
                   "fixed_best": cands[fixed_best],
                   "best_val_regret": best_val_regret}, f, indent=2)
    print(f"[train] DONE best_val_regret={best_val_regret:.5f} "
          f"vs fixed {fixed_regret:.5f}")


if __name__ == "__main__":
    main()
