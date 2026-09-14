# Label-free routing

The deployed router (`RL/`) picks a sigmoid curve from a `(task, metric)`
lookup, which requires the task label at inference time. The code here removes that
dependency: the decision is made from a single short probe forward pass, and the training
corpus never touches LongBench, which stays a pure test set.

## Pipeline

```
datasets/make_docqa_div.py     build training prompts (no LongBench data)
datasets/build_recipe_v5.py    "  with reference answers
datasets/build_recipe_v6.py    "  from real long documents (LooGLE / QuALITY)
        |
iclr/gen_actions.py            full-cache + 13-action outputs for those prompts
        |
iclr/reward.py                 outputs -> 13-dim reward vector (proxy P0 or the task metric)
iclr/probe_v3.py               router input features from one short probe forward
        |
iclr/mlp_v3.py                 train the router, then evaluate on LongBench untouched
```

Shared helpers: `recipe_table.py` (action grid, length buckets), `clean_cell_table.py` (cell
labels), `proxy_gate.py` (store loading, proxy screening), `lb_full.py` (aligning full-cache
predictions to store indices).

## Router

Input is 24 dimensions, all produced by the same probe forward:

| block | dims | source |
|---|---|---|
| generation-length bucket one-hot | 3 | request metadata |
| probe output-format softmax | 5 | probe forward |
| probe hidden-state PCA | 16 | same forward, no extra compute |

An MLP (128 x 3) maps that to the 13 grid actions. The target is the raw reward and the loss
is listwise score-weighted, so the size of the gaps between actions survives instead of being
flattened to ranks; three seeds are ensembled. The PCA basis is fitted on the training corpus
alone. Preparation costs 74.7 ms, about 1.6% of a 16K-token prefill (`overhead_bench.py`).

```
python iclr/probe_v3.py  --model llama3-8b --corpus lb     --dump_hidden --gpu 0
python iclr/probe_v3.py  --model llama3-8b --corpus recipe --dump_hidden --gpu 0 \
    --recipe datasets/training/raw/recipe_poolz_llama3-8b.jsonl
python iclr/mlp_v3.py    --model llama3-8b --pool poolz --variant v1t64e --feats prob \
    --reward p0 --force_k 13 --hid_dim 16 --tgt raw --loss ce --hidden 128 --depth 3
```

## Ablations

`struct_search.py` searches the architecture against its own ceiling by training on LongBench
itself; those numbers are diagnostic and are never reported as performance. `ucb_v3.py`
compares bandit training (LinUCB, NeuralUCB) with the supervised router on identical inputs,
separating the linear/nonlinear axis from the bandit/full-information one.

## Closed branch: per-head (a, b)

`trace_dump.py`, `build_rewards.py`, `eval_lb.py` and `analyze_ladder.py` belong to an earlier
question: whether the sigmoid parameters should differ per attention head, scored by the
label-free future-attention AUC of KVP. The answer was no, across six independent negative
results (paired coordinate search found 0 of 128 configurations significant; greedy assembly
reached LB 30.53 against 36.55 for a uniform curve). The files are kept because the tracing
infrastructure is reusable, not because the branch is active.

## Conventions learned the hard way

1. Uniform curves over the 13-grid on LongBench are read from the store
   (`result_txt/backup/fast_store/`), never re-run. Full-cache predictions live there too.
2. A custom attention function must apply the causal mask itself when `mask=None`; follow the
   pattern in `utils_real_drop/compress.py`. Smoke tests must include a long context and a
   visual check of the generated text.
3. Scan for free GPUs immediately before launching. Fixed device lists collide with other
   users on the shared machines.
4. When stopping a multi-stage runner, kill the parent subshell first; killing only the child
   lets the next stage relaunch.
