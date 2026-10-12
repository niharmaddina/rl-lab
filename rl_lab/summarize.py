"""Aggregate ``runs/<env>/<algo>/seed_*/eval.json`` into ``results/summary.json``
and print a Markdown results table.

Example::

    python -m rl_lab.summarize --runs-root runs --out results/summary.json \
        --hardware "Apple M1 Pro, CPU"
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def collect(runs_root: Path) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for f in sorted(runs_root.glob("*/*/seed_*/eval.json")):
        r = json.loads(f.read_text())
        groups[(r["env"], r["algo"])].append(r)

    entries = []
    for (env, algo), runs in sorted(groups.items()):
        runs.sort(key=lambda r: r["seed"])
        per_seed_means = np.array([r["eval"]["mean_return"] for r in runs])
        pooled = np.concatenate([r["eval"]["returns"] for r in runs])
        walls = [r["train_wall_time_s"] for r in runs]
        entries.append({
            "env": env,
            "algo": algo,
            "seeds": [r["seed"] for r in runs],
            "timesteps": runs[0]["timesteps"],
            "n_envs": runs[0]["n_envs"],
            "eval_episodes_per_seed": runs[0]["eval"]["episodes"],
            "eval_seed_base": runs[0]["eval"]["seeds"][0],
            "final_mean_return": float(per_seed_means.mean()),
            "final_std_return_across_seeds": float(per_seed_means.std()),
            "pooled_mean_return": float(pooled.mean()),
            "pooled_std_return": float(pooled.std()),
            "per_seed": [
                {
                    "seed": r["seed"],
                    "mean_return": r["eval"]["mean_return"],
                    "std_return": r["eval"]["std_return"],
                    "train_wall_time_s": r["train_wall_time_s"],
                    "fps": r["fps"],
                }
                for r in runs
            ],
            "wall_time_s_per_seed_mean": float(np.mean(walls)),
            "wall_time_s_total": float(np.sum(walls)),
            "date_utc": runs[0]["runtime"]["date_utc"],
            "versions": {k: runs[0]["runtime"][k] for k in ("python", "torch", "gymnasium", "stable_baselines3")},
        })
    return entries


def markdown_table(entries: list[dict]) -> str:
    lines = [
        "| Env | Algo | Seeds | Timesteps / seed | Eval return (mean +/- std across seeds) | Per-seed eval means | Wall time / seed |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        per_seed = ", ".join(f"{p['mean_return']:.1f}" for p in e["per_seed"])
        lines.append(
            f"| {e['env']} | {e['algo'].upper()} | {len(e['seeds'])} | {e['timesteps']:,} | "
            f"{e['final_mean_return']:.1f} +/- {e['final_std_return_across_seeds']:.1f} | {per_seed} | "
            f"{e['wall_time_s_per_seed_mean'] / 60:.1f} min |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--out", default="results/summary.json")
    p.add_argument("--hardware", default=None, help='e.g. "Apple M1 Pro, CPU"')
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    entries = collect(Path(args.runs_root))
    if not entries:
        print(f"no eval.json found under {args.runs_root}", file=sys.stderr)
        return 1
    doc = {"hardware": args.hardware, "experiments": entries}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(markdown_table(entries))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
