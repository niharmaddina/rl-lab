"""Evaluate a saved model on held-out seeds.

Example::

    python -m rl_lab.evaluate --model runs/CartPole-v1/dqn/seed_0/model.zip \
        --env CartPole-v1 --episodes 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rl_lab.common import ALGOS, evaluate_policy_seeded


def infer_algo(model_path: Path) -> str:
    """Read the algo from the eval.json written next to the model, if present."""
    sidecar = model_path.parent / "eval.json"
    if sidecar.exists():
        algo = json.loads(sidecar.read_text()).get("algo")
        if algo in ALGOS:
            return algo
    raise ValueError("could not infer algorithm; pass --algo {dqn,ppo}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="path to model.zip")
    p.add_argument("--env", required=True, help="Gymnasium env id")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--algo", choices=sorted(ALGOS), default=None, help="inferred from eval.json if omitted")
    p.add_argument("--seed-base", type=int, default=10_000, help="episode i uses seed seed_base + i")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax/mean")
    p.add_argument("--out", default=None, help="optional path to write the JSON result")
    p.add_argument("--device", default="cpu")
    return p


def evaluate(args: argparse.Namespace) -> dict:
    model_path = Path(args.model)
    algo = args.algo or infer_algo(model_path)
    model = ALGOS[algo].load(model_path, device=args.device)
    result = evaluate_policy_seeded(
        model, args.env, episodes=args.episodes, seed_base=args.seed_base, deterministic=not args.stochastic
    )
    result["algo"] = algo
    result["model"] = str(model_path)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{args.env} {algo} {args.episodes} episodes: "
        f"mean return {result['mean_return']:.2f} +/- {result['std_return']:.2f} "
        f"(min {result['min_return']:.1f}, max {result['max_return']:.1f})"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    evaluate(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
