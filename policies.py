import numpy as np

REJECT = 0
ACCEPT = 1


class RandomPolicy:
    """Accept with probability 1/T each week (independent of quality)."""

    def __init__(self, T, seed=None):
        self.T = int(T)
        self._rng = np.random.default_rng(seed)

    def act(self, obs):
        return ACCEPT if self._rng.random() < 1.0 / self.T else REJECT


class ThresholdPolicy:
    """Accept iff the observed quality U >= u_min (fixed threshold)."""

    def __init__(self, u_min):
        self.u_min = float(u_min)

    def act(self, obs):
        _, u = float(obs[0]), float(obs[1])
        return ACCEPT if u >= self.u_min else REJECT


class OptimalPolicy:
    """Optimal finite-horizon policy from Problem 1(c), hardcoded for T=4, K=4."""

    THRESHOLDS = {1: 3.25, 2: 3.00, 3: 2.50, 4: 0.00}

    TABLE = {
        (1, 1): REJECT, (1, 2): REJECT, (1, 3): REJECT, (1, 4): ACCEPT,
        (2, 1): REJECT, (2, 2): REJECT, (2, 3): ACCEPT, (2, 4): ACCEPT,
        (3, 1): REJECT, (3, 2): REJECT, (3, 3): ACCEPT, (3, 4): ACCEPT,
        (4, 1): ACCEPT, (4, 2): ACCEPT, (4, 3): ACCEPT, (4, 4): ACCEPT,
    }

    def act(self, obs):
        t, u = int(round(float(obs[0]))), float(obs[1])
        thr = self.THRESHOLDS.get(t, 0.0)
        return ACCEPT if u >= thr else REJECT
