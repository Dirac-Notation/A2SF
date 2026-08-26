"""WAITS action grid — the 13 paired (a, b) sigmoid-forgetting actions.

The routing policy chooses one of these per (task, metric); the chosen (a, b) is fed to
the `waits` scorer at compression time (see utils_real_drop/scorers/waits.py).

  w[q] = sigmoid(a * (q - (N - b - 0.5)))

a=0 makes b irrelevant (σ(0)=0.5 uniform) so only one (0, 1) entry is kept; the rest is
the cartesian product {0.01, 0.1, 10} × {1, 16, 32, 128}  →  13 unique actions total.
"""

_A_BASE = [0.01, 0.1, 10.0]
_B_BASE = [1.0, 16.0, 32.0, 128.0]

SIGMOID_A_VALUES = [0.0]
SIGMOID_B_VALUES = [1.0]
for _a in _A_BASE:
    for _b in _B_BASE:
        SIGMOID_A_VALUES.append(_a)
        SIGMOID_B_VALUES.append(_b)
assert len(SIGMOID_A_VALUES) == 13 == len(SIGMOID_B_VALUES)

# Indices of "hard-like + a=0" actions, used by the hard action-subset ablation.
HARD_LIKE_INDICES = [0, 9, 10, 11, 12]  # (0,1), (10,1), (10,16), (10,32), (10,128)

NUM_ACTIONS = len(SIGMOID_A_VALUES)
