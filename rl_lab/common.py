"""Shared helpers: algorithm registry, environment construction, evaluation."""
from __future__ import annotations

import os
import platform
import time
from datetime import datetime, timezone
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback
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
    env: gym.Env | None = None,
) -> dict[str, Any]:
    """Run ``episodes`` full episodes, episode *i* reset with seed ``seed_base + i``.

    The seeds are disjoint from the small training seeds (0, 1, 2, ...), so the
    evaluation measures the policy on initial states it did not train from.
    ``env`` lets a caller reuse one ``gym.make(env_id)`` instance across repeated
    evaluations; because every episode is re-seeded on reset, the result is the
    same as with a fresh environment, which is what is made (and closed) when
    ``env`` is None.
    """
    own_env = env is None
    if env is None:
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
    if own_env:
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


class HeldOutEvalCallback(EvalCallback):
    """``EvalCallback`` that scores the agent with :func:`evaluate_policy_seeded`.

    stable-baselines3's own ``EvalCallback`` resets its evaluation env without a
    seed, so each periodic evaluation runs on different initial states and is
    not comparable with the final evaluation in ``rl_lab.train``. This subclass
    keeps the callback's bookkeeping (``evaluations.npz`` with ``timesteps``,
    ``results`` and ``ep_lengths``; ``best_model.zip`` rewritten whenever the
    mean return strictly improves on every earlier evaluation) but scores the
    agent with the exact function the final evaluation uses: ``episodes`` full
    episodes reset with seeds ``seed_base + i`` and deterministic actions. The
    only difference from the final evaluation is that one ``gym.make`` env is
    reused across evaluations instead of being created per evaluation; the
    per-episode reseeding makes the two equivalent.

    ``eval_freq`` counts calls of the callback, one per vectorised step, so
    callers pass ``max(eval_every // n_envs, 1)`` to evaluate every
    ``eval_every`` environment steps. ``evaluations.npz`` also stores the
    evaluation ``seeds``.
    """

    def __init__(
        self,
        env_id: str,
        episodes: int,
        seed_base: int,
        eval_freq: int,
        out_dir: str | os.PathLike,
        verbose: int = 0,
    ):
        self._eval_env_single = gym.make(env_id)
        super().__init__(
            eval_env=self._eval_env_single,
            n_eval_episodes=episodes,
            eval_freq=eval_freq,
            log_path=str(out_dir),
            best_model_save_path=str(out_dir),
            deterministic=True,
            verbose=verbose,
            warn=False,
        )
        self.env_id = env_id
        self.seed_base = seed_base
        self.best_step: int | None = None
        self.n_evaluations = 0
        self.eval_wall_time_s = 0.0

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        t0 = time.perf_counter()
        ev = evaluate_policy_seeded(
            self.model, self.env_id, episodes=self.n_eval_episodes, seed_base=self.seed_base,
            deterministic=True, env=self._eval_env_single,
        )
        self.eval_wall_time_s += time.perf_counter() - t0
        self.n_evaluations += 1

        self.evaluations_timesteps.append(self.num_timesteps)
        self.evaluations_results.append(ev["returns"])
        self.evaluations_length.append(ev["lengths"])
        np.savez(
            self.log_path,
            timesteps=self.evaluations_timesteps,
            results=self.evaluations_results,
            ep_lengths=self.evaluations_length,
            seeds=ev["seeds"],
        )

        self.last_mean_reward = ev["mean_return"]
        self.logger.record("eval/mean_reward", ev["mean_return"])
        self.logger.record("eval/mean_ep_length", ev["mean_length"])
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(self.num_timesteps)
        if self.verbose >= 1:
            print(f"eval step={self.num_timesteps} return={ev['mean_return']:.1f} +/- {ev['std_return']:.1f}")

        if ev["mean_return"] > self.best_mean_reward:
            self.best_mean_reward = ev["mean_return"]
            self.best_step = self.num_timesteps
            self.model.save(os.path.join(self.best_model_save_path, "best_model"))
        return True

    def _on_training_end(self) -> None:
        self._eval_env_single.close()


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
