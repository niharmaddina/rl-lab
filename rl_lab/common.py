"""Shared helpers: algorithm registry, environment construction, evaluation."""
from __future__ import annotations

import platform
from datetime import datetime, timezone
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

ALGOS: dict[str, type[BaseAlgorithm]] = {"dqn": DQN, "ppo": PPO}


def make_training_env(env_id: str, n_envs: int, seed: int, monitor_dir: str) -> VecEnv:
    """Vectorised training env; every sub-env is wrapped in a Monitor writing
    ``<monitor_dir>/<rank>.monitor.csv``."""
    return make_vec_env(env_id, n_envs=n_envs, seed=seed, monitor_dir=monitor_dir)


def evaluate_policy_seeded(
    model: BaseAlgorithm,
    env_id: str,
    episodes: int = 20,
    seed_base: int = 10_000,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Run ``episodes`` full episodes, episode *i* reset with seed ``seed_base + i``.

    The seeds are disjoint from the small training seeds (0, 1, 2, ...), so the
    evaluation measures the policy on initial states it did not train from.
    """
    env = gym.make(env_id)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    returns: list[float] = []
    lengths: list[int] = []
    seeds = [seed_base + i for i in range(episodes)]
    for s in seeds:
        obs, _ = env.reset(seed=s)
        total, steps, done = 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            if discrete:
                action = int(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            done = terminated or truncated
        returns.append(total)
        lengths.append(steps)
    env.close()
    r = np.asarray(returns, dtype=float)
    return {
        "env": env_id,
        "episodes": episodes,
        "deterministic": deterministic,
        "seeds": seeds,
        "returns": returns,
        "lengths": lengths,
        "mean_return": float(r.mean()),
        "std_return": float(r.std()),
        "min_return": float(r.min()),
        "max_return": float(r.max()),
        "mean_length": float(np.mean(lengths)),
    }


def runtime_info() -> dict[str, str]:
    """Versions and platform, recorded alongside every result."""
    import gymnasium
    import stable_baselines3
    import torch

    return {
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
    }
