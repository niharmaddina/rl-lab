"""Train one (env, algo, seed) run and evaluate it on held-out seeds.

Example::

    python -m rl_lab.train --env CartPole-v1 --algo dqn --timesteps 100000 \
        --seed 0 --eval-every 5000 --out runs/CartPole-v1/dqn/seed_0

Writes to ``--out``: ``<rank>.monitor.csv`` (per-episode training log),
``model.zip`` (the final weights), ``config.yaml`` (the resolved config) and
``eval.json``. ``eval.json`` has an ``eval.final`` block (mean/std return of the
final weights over the evaluation seeds) plus the resolved config, timings and
library versions.

With ``--eval-every N`` the policy is also evaluated every N environment steps
during training on the same held-out seeds with the same deterministic policy
(``rl_lab.common.HeldOutEvalCallback``). The run directory then also holds
``evaluations.npz`` (``timesteps``, ``results``, ``ep_lengths``, ``seeds``) and
``best_model.zip`` (the checkpoint with the highest mean held-out return), and
``eval.json`` gains ``eval.best`` (that checkpoint re-evaluated on the same
seeds), ``eval.best_step`` and ``eval.best_fresh_seeds`` (the same checkpoint
on 20 further seeds it was not selected on, ``eval_seed_base + 10000`` onwards).
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

from rl_lab.common import ALGOS, HeldOutEvalCallback, evaluate_policy_seeded, make_training_env, runtime_info
from rl_lab.config import config_path, load_config

FRESH_SEED_OFFSET = 10_000  # best checkpoint is re-checked on eval_seed_base + this offset


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
    p.add_argument("--eval-every", type=int, default=0,
                   help="evaluate on the held-out seeds every N environment steps during training and keep "
                        "the best checkpoint as best_model.zip (default 0 = off)")
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
    eval_every = max(args.eval_every, 0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    set_random_seed(args.seed)

    vec_env = make_training_env(args.env, n_envs=n_envs, seed=args.seed, monitor_dir=str(out))
    model = ALGOS[algo](env=vec_env, seed=args.seed, device=args.device, verbose=args.verbose, **cfg["algo_kwargs"])

    callback = None
    if eval_every:
        callback = HeldOutEvalCallback(
            args.env, episodes=eval_episodes, seed_base=eval_seed_base,
            eval_freq=max(eval_every // n_envs, 1), out_dir=out, verbose=args.verbose,
        )

    t0 = time.perf_counter()
    model.learn(total_timesteps=timesteps, callback=callback)
    train_wall = time.perf_counter() - t0
    vec_env.close()

    model.save(out / "model.zip")

    t1 = time.perf_counter()
    final = evaluate_policy_seeded(model, args.env, episodes=eval_episodes, seed_base=eval_seed_base)
    best = best_fresh = None
    best_step = None
    if callback is not None and callback.best_step is not None:
        best_step = callback.best_step
        best_model = ALGOS[algo].load(out / "best_model.zip", device=args.device)
        best = evaluate_policy_seeded(best_model, args.env, episodes=eval_episodes, seed_base=eval_seed_base)
        best_fresh = evaluate_policy_seeded(
            best_model, args.env, episodes=eval_episodes, seed_base=eval_seed_base + FRESH_SEED_OFFSET
        )
    eval_wall = time.perf_counter() - t1

    in_training_eval_wall = callback.eval_wall_time_s if callback is not None else 0.0
    result = {
        "env": args.env,
        "algo": algo,
        "seed": args.seed,
        "timesteps": timesteps,
        "n_envs": n_envs,
        "eval_every": eval_every,
        "train_wall_time_s": round(train_wall, 2),
        "in_training_eval_wall_time_s": round(in_training_eval_wall, 2),
        "eval_wall_time_s": round(eval_wall, 2),
        "fps": round(timesteps / (train_wall - in_training_eval_wall), 1),
        "eval": {
            "final": final,
            "best": best,
            "best_step": best_step,
            "best_fresh_seeds": best_fresh,
            "n_evaluations": callback.n_evaluations if callback is not None else 0,
        },
        "config": {**cfg, "algo": algo, "timesteps": timesteps, "n_envs": n_envs},
        "runtime": runtime_info(),
    }
    (out / "eval.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "config.yaml").write_text(yaml.safe_dump(result["config"], sort_keys=False))
    msg = (
        f"{args.env} {algo} seed={args.seed} timesteps={timesteps} train={train_wall:.0f}s "
        f"final={final['mean_return']:.1f} +/- {final['std_return']:.1f}"
    )
    if best is not None:
        msg += (f" best={best['mean_return']:.1f} +/- {best['std_return']:.1f} at step {best_step}"
                f" (fresh seeds {best_fresh['mean_return']:.1f})")
    print(f"{msg} over {eval_episodes} held-out seeds -> {out}")
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
