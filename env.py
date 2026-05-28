import gymnasium as gym
import numpy as np
from gymnasium import spaces

REJECT = 0
ACCEPT = 1


class ApartmentEnv(gym.Env):
    """Finite-horizon apartment-search environment."""

    metadata = {"render_modes": []}

    def __init__(self, T, K, seed=None, noise_std=0.0):
        super().__init__()
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")

        self.T = int(T)
        self.K = int(K)
        self.noise_std = float(noise_std)

        self.action_space = spaces.Discrete(2)  # 0 = reject, 1 = accept
        low = np.array([1.0, -np.inf], dtype=np.float32)
        high = np.array([float(self.T), np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._t = 1
        self._true_u = 0
        self._done = True

    def _draw_quality(self):
        """Draw U_t ~ Uniform{1, ..., K}."""
        return int(self._rng.integers(1, self.K + 1))

    def _observe(self):
        """Build the observation, applying noise to quality if configured."""
        u_obs = float(self._true_u)
        if self.noise_std > 0.0:
            u_obs += float(self._rng.normal(0.0, self.noise_std))
        return np.array([float(self._t), u_obs], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 1
        self._true_u = self._draw_quality()
        self._done = False
        info = {"true_u": self._true_u}
        return self._observe(), info

    def step(self, action):
        if self._done:
            raise RuntimeError("step() called on a terminated episode; call reset().")
        if action not in (REJECT, ACCEPT):
            raise ValueError(f"action must be 0 (reject) or 1 (accept), got {action}")

        if action == ACCEPT:
            reward = float(self._true_u)  # reward uses TRUE quality
            self._done = True
            obs = self._observe()
            return obs, reward, True, False, {"true_u": self._true_u}

        # REJECT
        if self._t >= self.T:
            # Rejected the last apartment -> sublet fallback, utility 0.
            reward = 0.0
            self._done = True
            obs = self._observe()
            return obs, reward, True, False, {"true_u": self._true_u}

        # Advance to next week with a fresh apartment.
        self._t += 1
        self._true_u = self._draw_quality()
        return self._observe(), 0.0, False, False, {"true_u": self._true_u}
