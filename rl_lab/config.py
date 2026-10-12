"""Load and validate per-environment YAML configs.

A config file lives at ``configs/<env_id>.yaml`` and looks like::

    env: CartPole-v1
    algo: dqn
    timesteps: 100000
    n_envs: 1
    eval_episodes: 20
    eval_seed_base: 10000
    algo_kwargs:
      policy: MlpPolicy
      learning_rate: 2.3e-3
      ...

``algo_kwargs`` is passed verbatim to the stable-baselines3 constructor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

ALGO_NAMES = ("dqn", "ppo")
REQUIRED_KEYS = ("env", "algo", "timesteps", "algo_kwargs")
DEFAULTS: dict[str, Any] = {
    "n_envs": 1,
    "eval_episodes": 20,
    "eval_seed_base": 10_000,
}


def config_path(env_id: str, config_dir: str | Path | None = None) -> Path:
    """Return the path of the YAML config for ``env_id``."""
    base = Path(config_dir) if config_dir is not None else CONFIG_DIR
    return base / f"{env_id}.yaml"


def _coerce_numbers(obj: Any) -> Any:
    """PyYAML reads ``1e-4`` (no dot) as a string; convert such strings to floats."""
    if isinstance(obj, dict):
        return {k: _coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numbers(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip().lower()
        if any(c.isdigit() for c in s) and set(s) <= set("0123456789.e+-"):
            try:
                return float(s)
            except ValueError:
                return obj
    return obj


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, fill defaults and validate the required keys."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    cfg = {**DEFAULTS, **_coerce_numbers(raw)}

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    if cfg["algo"] not in ALGO_NAMES:
        raise ValueError(f"{path}: algo must be one of {ALGO_NAMES}, got {cfg['algo']!r}")
    if not isinstance(cfg["algo_kwargs"], dict):
        raise ValueError(f"{path}: algo_kwargs must be a mapping")
    if "policy" not in cfg["algo_kwargs"]:
        raise ValueError(f"{path}: algo_kwargs.policy is required (e.g. MlpPolicy)")
    for key in ("timesteps", "n_envs", "eval_episodes", "eval_seed_base"):
        cfg[key] = int(cfg[key])
        if cfg[key] <= 0 and key != "eval_seed_base":
            raise ValueError(f"{path}: {key} must be positive")
    return cfg
