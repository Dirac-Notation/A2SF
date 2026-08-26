"""RoutingNeuralUCB — the (task, metric) routing policy model (Architecture).

NeuralUCB contextual bandit over observable metadata state phi(s) = [task_oh | metric_oh].
The state is categorical, so the NeuralUCB Sherman-Morrison Sigma^-1 core reduces exactly to
disjoint per-arm LinUCB on the one-hot features (no deep net needed):

    A_a = lambda*I + sum phi phi^T   (over plays of arm a),   theta_a = A_a^-1 b_a
    select   a* = argmax_a theta_a.phi + beta*sqrt(phi^T A_a^-1 phi)     (UCB)
    update   on observing r: rank-1 Sherman-Morrison of A_{a*}^-1, b_{a*} += r*phi

At deploy the greedy policy (no bonus) maps each (task, metric) to one of the 13 actions; since
phi depends only on (task, metric) the policy is a lookup table, applied at eval via
`longbench.py --waits_table` (see export_table()).
"""
import numpy as np

from .action_grid import SIGMOID_A_VALUES, SIGMOID_B_VALUES, NUM_ACTIONS
from .metadata import (
    TASK_TYPE_ORDER, METRIC_TYPE_ORDER,
    task_type_to_index, metric_type_to_index, dataset_metric,
)


class RoutingNeuralUCB:
    def __init__(self, lam: float = 1.0, beta: float = 1.0, seed: int = 0, actions=None):
        """actions: list of (a, b) curves the agent selects among (arms).
        Default = the full 13-grid; pass e.g. the U5b 5-curve set for the champion."""
        self.D = len(TASK_TYPE_ORDER) + len(METRIC_TYPE_ORDER)
        self.actions = ([(float(a), float(b)) for a, b in zip(SIGMOID_A_VALUES, SIGMOID_B_VALUES)]
                        if actions is None else [(float(a), float(b)) for a, b in actions])
        self.A = len(self.actions)
        self.lam = float(lam)
        self.beta = float(beta)
        self.rng = np.random.RandomState(seed)
        self.Ainv = [np.eye(self.D) / self.lam for _ in range(self.A)]
        self.b = [np.zeros(self.D) for _ in range(self.A)]
        self.theta = [self.Ainv[a] @ self.b[a] for a in range(self.A)]
        self.plays = np.zeros(self.A, dtype=int)

    # ---- model artifact I/O (the SELECTOR is the saved object, not its decisions) ----
    def save(self, path: str) -> None:
        np.savez(path, Ainv=np.stack(self.Ainv), b=np.stack(self.b), plays=self.plays,
                 actions=np.asarray(self.actions, dtype=float),
                 lam=self.lam, beta=self.beta)

    @classmethod
    def load(cls, path: str) -> "RoutingNeuralUCB":
        z = np.load(path)
        agent = cls(lam=float(z["lam"]), beta=float(z["beta"]),
                    actions=[tuple(x) for x in z["actions"]])
        agent.Ainv = [z["Ainv"][a] for a in range(agent.A)]
        agent.b = [z["b"][a] for a in range(agent.A)]
        agent.theta = [agent.Ainv[a] @ agent.b[a] for a in range(agent.A)]
        agent.plays = z["plays"]
        return agent

    def phi(self, task_type: str, metric_type: str) -> np.ndarray:
        v = np.zeros(self.D)
        v[task_type_to_index(task_type)] = 1.0
        v[len(TASK_TYPE_ORDER) + metric_type_to_index(metric_type)] = 1.0
        return v

    def select(self, p: np.ndarray, epsilon: float = 0.0) -> int:
        """Pick an arm to play: epsilon-greedy explore, else UCB."""
        if epsilon > 0 and self.rng.rand() < epsilon:
            return int(self.rng.randint(self.A))
        ucb = np.array([self.theta[a] @ p + self.beta * np.sqrt(max(p @ self.Ainv[a] @ p, 0.0))
                        for a in range(self.A)])
        return int(ucb.argmax())

    def update(self, p: np.ndarray, action: int, reward: float) -> None:
        Ap = self.Ainv[action] @ p
        self.Ainv[action] -= np.outer(Ap, Ap) / (1.0 + p @ Ap)
        self.b[action] += float(reward) * p
        self.theta[action] = self.Ainv[action] @ self.b[action]
        self.plays[action] += 1

    def greedy_action(self, task_type: str, metric_type: str) -> int:
        """Deploy-time action (no exploration bonus)."""
        p = self.phi(task_type, metric_type)
        return int(np.array([self.theta[a] @ p for a in range(self.A)]).argmax())

    def action_ab(self, action: int):
        return self.actions[action]

    def export_table(self, datasets, dataset_to_task, n_samples) -> dict:
        """Build a `longbench.py --waits_table` block: {dataset: [[a, b], ...] per sample}.
        All samples of a dataset share the action the policy assigns to its (task, metric)."""
        table = {}
        for ds in datasets:
            task = dataset_to_task.get(ds, "unknown")
            met = dataset_metric(ds, task)
            a, b = self.action_ab(self.greedy_action(task, met))
            table[ds] = [[a, b]] * int(n_samples[ds])
        return table
