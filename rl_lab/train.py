"""Train one (env, algo, seed) run and evaluate it on held-out seeds.

Example::

    python -m rl_lab.train --env CartPole-v1 --algo dqn --timesteps 100000 \
        --seed 0 --out runs/CartPole-v1/dqn/seed_0

Writes to ``--out``: ``<rank>.monitor.csv`` (per-episode training log),
``model.zip`` and ``eval.json`` (mean/std return over the evaluation seeds plus
the resolved config, timings and library versions).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from stable_baselines3.common.utils import set_random_seed

from rl_lab.common import ALGOS, evaluate_policy_seeded, make_training_env, runtime_info
from rl_lab.config import config_path, load_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, help="Gymnasium env id, e.g. CartPole-v1")
    p.add_argument("--algo", choices=sorted(ALGOS), default=None,
                   help="dqn or ppo; defaults to the value in the config")
    p.add_argument("--timesteps", type=int, default=None, help="override config timesteps")
    p.add_argument("--seed", type=int, default=0, help="training seed (default 0)")
    p.add_argument("--out", required=True, help="output directory for monitor csv, model.zip, eval.json")
    p.add_argument("--config", default=None, help="YAML path; defaults to configs/<env>.yaml")
    p.add_argument("--n-envs", type=int, default=None, help="override config n_envs")
    p.add_argument("--eval-episodes", type=int, default=None, help="override config eval_episodes (default 20)")
    p.add_argument("--eval-seed-base", type=int, default=None, help="first evaluation seed (default 10000)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--torch-threads", type=int, default=1,
                   help="torch intra-op threads; 1 is fastest for small MLPs and lets seeds run in parallel")
    p.add_argument("--verbose", type=int, default=0)
    return p


def train(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config or config_path(args.env))
    if cfg["env"] != args.env:
        raise ValueError(f"config env {cfg['env']!r} does not match --env {args.env!r}")
    algo = args.algo or cfg["algo"]
    timesteps = args.timesteps or cfg["timesteps"]
    n_envs = args.n_envs or cfg["n_envs"]
    eval_episodes = args.eval_episodes or cfg["eval_episodes"]
    eval_seed_base = args.eval_seed_base if args.eval_seed_base is not None else cfg["eval_seed_base"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    set_random_seed(args.seed)

    vec_env = make_training_env(args.env, n_envs=n_envs, seed=args.seed, monitor_dir=str(out))
    model = ALGOS[algo](env=vec_env, seed=args.seed, device=args.device, verbose=args.verbose, **cfg["algo_kwargs"])

    t0 = time.perf_counter()
    model.learn(total_timesteps=timesteps)
    train_wall = time.perf_counter() - t0
    vec_env.close()

    model.save(out / "model.zip")

    t1 = time.perf_counter()
    ev = evaluate_policy_seeded(model, args.env, episodes=eval_episodes, seed_base=eval_seed_base)
    eval_wall = time.perf_counter() - t1

    result = {
        "env": args.env,
        "algo": algo,
        "seed": args.seed,
        "timesteps": timesteps,
        "n_envs": n_envs,
        "train_wall_time_s": round(train_wall, 2),
        "eval_wall_time_s": round(eval_wall, 2),
        "fps": round(timesteps / train_wall, 1),
        "eval": ev,
        "config": {**cfg, "algo": algo, "timesteps": timesteps, "n_envs": n_envs},
        "runtime": runtime_info(),
    }
    (out / "eval.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "config.yaml").write_text(yaml.safe_dump(result["config"], sort_keys=False))
    print(
        f"{args.env} {algo} seed={args.seed} timesteps={timesteps} "
        f"train={train_wall:.0f}s eval_return={ev['mean_return']:.1f} +/- {ev['std_return']:.1f} "
        f"over {eval_episodes} held-out seeds -> {out}"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
