"""RL/dev - development track for NeuralUCB agent-structure upgrades.

Track rules (see RL/README.md, "three tracks"):
  R (routing champion, LinUCB)  : RL/{action_grid,metadata,dataset,model,train}.py - deployed.
  P (submitted per-prompt)      : RL/{train_perprompt_submitted,a2sf_model}.py + agent/ + env/
                                  - VERSION-LOCKED reproduction (script/repro_2694.sh,
                                  script/smoke_neuralucb.sh). Never edit for new research.
  D (this package)              : upgrade experiments live HERE. Seeded with a copy of the
                                  submitted NeuralUCBAgent; env/ encoders may be imported
                                  read-only from RL.env. New agent variants, losses, and
                                  trainers go in this package only.

Before trusting any experiment here, run script/smoke_neuralucb.sh to confirm the locked
P track still reproduces (eval = 26.94, train losses match the seed-42 log).
"""
from .neural_ucb_agent import NeuralUCBAgent  # dev starting point (copy of submitted agent)

__all__ = ["NeuralUCBAgent"]
