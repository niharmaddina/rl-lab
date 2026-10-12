"""Aggregate ``runs/<env>/<algo>/seed_*/eval.json`` into ``results/summary.json``
and print a Markdown results table.

Example::

    python -m rl_lab.summarize --runs-root runs --out results/summary.json \
        --hardware "Apple M1 Pro, CPU"

Every row reports the final weights (``final``) and, for runs trained with
``--eval-every``, the best checkpoint under periodic held-out evaluation
(``best``, with the step at which it was saved). Runs without a best checkpoint
show a dash in the ``best`` columns.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def eval_blocks(record: dict) -> tuple[dict, dict | None, int | None, dict | None]:
    """Return ``(final, best, best_step, best_fresh_seeds)`` from an eval.json record.

    Accepts the layout written before ``--eval-every`` existed, where ``eval``
    was the final block itself (then ``best`` is None).
    """
    ev = record["eval"]
    if "final" in ev:
        return ev["final"], ev.get("best"), ev.get("best_step"), ev.get("best_fresh_seeds")
    return ev, None, None, None


def collect(runs_root: Path) -> list[dict]:
    # One table row per (env, algo, timesteps budget); the run directory
    # runs/<env>/<name>/seed_* is recorded so the row can be traced back.
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    run_dirs: dict[tuple[str, str, int], str] = {}
    for f in sorted(runs_root.glob("*/*/seed_*/eval.json")):
        r = json.loads(f.read_text())
        key = (r["env"], r["algo"], int(r["timesteps"]))
        groups[key].append(r)
        run_dirs[key] = str(f.parent.parent.relative_to(runs_root))

    entries = []
    for key, runs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[0][2])):
        env, algo, _ = key
        runs.sort(key=lambda r: r["seed"])
        blocks = [eval_blocks(r) for r in runs]
        finals = [b[0] for b in blocks]
        per_seed_means = np.array([fin["mean_return"] for fin in finals])
        pooled = np.concatenate([fin["returns"] for fin in finals])
        walls = [r["train_wall_time_s"] for r in runs]

        has_best = all(b[1] is not None for b in blocks)
        best_means = np.array([b[1]["mean_return"] for b in blocks]) if has_best else None
        fresh = [b[3]["mean_return"] for b in blocks if b[3] is not None]

        entries.append({
            "env": env,
            "algo": algo,
            "run_dir": run_dirs[key],
            "seeds": [r["seed"] for r in runs],
            "timesteps": runs[0]["timesteps"],
            "n_envs": runs[0]["n_envs"],
            "eval_every": runs[0].get("eval_every", 0),
            "eval_episodes_per_seed": finals[0]["episodes"],
            "eval_seed_base": finals[0]["seeds"][0],
            "final_mean_return": float(per_seed_means.mean()),
            "final_std_return_across_seeds": float(per_seed_means.std()),
            "pooled_mean_return": float(pooled.mean()),
            "pooled_std_return": float(pooled.std()),
            "best_mean_return": float(best_means.mean()) if has_best else None,
            "best_std_return_across_seeds": float(best_means.std()) if has_best else None,
            "best_fresh_seeds_mean_return": float(np.mean(fresh)) if has_best and len(fresh) == len(runs) else None,
            "per_seed": [
                {
                    "seed": r["seed"],
                    "mean_return": fin["mean_return"],
                    "std_return": fin["std_return"],
                    "best_mean_return": best["mean_return"] if best is not None else None,
                    "best_std_return": best["std_return"] if best is not None else None,
                    "best_step": step,
                    "best_fresh_seeds_mean_return": fresh_b["mean_return"] if fresh_b is not None else None,
                    "train_wall_time_s": r["train_wall_time_s"],
                    "in_training_eval_wall_time_s": r.get("in_training_eval_wall_time_s", 0.0),
                    "fps": r["fps"],
                }
                for r, (fin, best, step, fresh_b) in zip(runs, blocks)
            ],
            "wall_time_s_per_seed_mean": float(np.mean(walls)),
            "wall_time_s_total": float(np.sum(walls)),
            "date_utc": runs[0]["runtime"]["date_utc"],
            "versions": {k: runs[0]["runtime"][k] for k in ("python", "torch", "gymnasium", "stable_baselines3")},
        })
    return entries


def format_step(step: int) -> str:
    """45000 -> ``45k``, 1500 -> ``1.5k``, 800 -> ``800``."""
    if step < 1000:
        return str(step)
    return f"{step / 1000:g}k"


def markdown_table(entries: list[dict]) -> str:
    lines = [
        "| Env | Algo | Seeds | Timesteps / seed | Final return (mean +/- std across seeds) | Per-seed final "
        "| Best-checkpoint return (mean +/- std across seeds) | Per-seed best @ step | Wall time / seed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        per_seed = ", ".join(f"{p['mean_return']:.1f}" for p in e["per_seed"])
        if e.get("best_mean_return") is not None:
            best = f"{e['best_mean_return']:.1f} +/- {e['best_std_return_across_seeds']:.1f}"
            per_seed_best = ", ".join(
                f"{p['best_mean_return']:.1f} @{format_step(p['best_step'])}" if p.get("best_mean_return") is not None
                else "-"
                for p in e["per_seed"]
            )
        else:
            best = per_seed_best = "-"
        lines.append(
            f"| {e['env']} | {e['algo'].upper()} | {len(e['seeds'])} | {e['timesteps']:,} | "
            f"{e['final_mean_return']:.1f} +/- {e['final_std_return_across_seeds']:.1f} | {per_seed} | "
            f"{best} | {per_seed_best} | "
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
        print(f"no {args.runs_root}/<env>/<name>/seed_*/eval.json found (that layout is what train.py is given in the README)", file=sys.stderr)
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
