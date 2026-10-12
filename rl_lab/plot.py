"""Learning curves from stable-baselines3 Monitor CSVs, one line per seed.

Example::

    python -m rl_lab.plot --runs runs/CartPole-v1/dqn/seed_* \
        --out results/CartPole-v1/curve.png --window 20

Each ``--runs`` directory holds one or more ``*.monitor.csv`` files (one per
vectorised sub-env). Episodes are ordered by wall-clock time, the x axis is the
cumulative number of environment steps, and the y axis is a trailing moving
average of the episode return. The thick line is the mean across seeds on a
common timestep grid; the band is +/- one standard deviation across seeds.
If a run directory holds ``evaluations.npz`` (written by ``train --eval-every``),
a vertical tick in the seed's colour marks the step of its best checkpoint, i.e.
the periodic held-out evaluation with the highest mean return.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Light-surface palette (validated categorical slots 1-3 + text tokens).
SEED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"


def load_monitor(run_dir: str | Path) -> pd.DataFrame:
    """Concatenate all ``*.monitor.csv`` in ``run_dir``; add a ``timesteps`` column.

    Monitor files start with one ``#{json}`` header line, then ``r,l,t`` rows
    (episode return, episode length, seconds since start).
    """
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("*.monitor.csv"))
    if not files:
        raise FileNotFoundError(f"no *.monitor.csv in {run_dir}")
    frames = [pd.read_csv(f, skiprows=1) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("t", kind="stable").reset_index(drop=True)
    df["timesteps"] = df["l"].cumsum()
    return df


def moving_average(values: pd.Series | np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over ``window`` episodes (shorter at the start)."""
    return pd.Series(np.asarray(values, dtype=float)).rolling(window, min_periods=1).mean().to_numpy()


def read_meta(run_dir: Path) -> dict:
    sidecar = run_dir / "eval.json"
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    return {}


def best_checkpoint(run_dir: Path) -> tuple[int, float] | None:
    """``(timesteps, mean_return)`` of the best periodic evaluation in ``evaluations.npz``.

    ``np.argmax`` returns the first index of the maximum, which is the same
    checkpoint ``EvalCallback`` saved (it only overwrites ``best_model.zip`` on a
    strict improvement). Returns None when the file is absent.
    """
    f = Path(run_dir) / "evaluations.npz"
    if not f.exists():
        return None
    data = np.load(f)
    means = np.asarray(data["results"], dtype=float).mean(axis=1)
    i = int(np.argmax(means))
    return int(data["timesteps"][i]), float(means[i])


def plot_runs(run_dirs: list[Path], out: Path, window: int = 20, title: str | None = None) -> dict:
    curves = []
    for d in run_dirs:
        df = load_monitor(d)
        curves.append((d, df["timesteps"].to_numpy(), moving_average(df["r"], window)))

    meta = read_meta(run_dirs[0])
    env = meta.get("env", run_dirs[0].parent.parent.name if len(run_dirs[0].parts) >= 3 else "")
    algo = meta.get("algo", "")

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Seeds are identified in the legend rather than by end-of-line labels:
    # converged seeds end at the same value and end labels would overprint.
    best_steps: dict[str, int] = {}
    for i, (d, x, y) in enumerate(curves):
        color = SEED_COLORS[i % len(SEED_COLORS)]
        ax.plot(x, y, color=color, linewidth=1.2, alpha=0.9, label=d.name.replace("_", " "))
        best = best_checkpoint(d)
        if best is not None:
            step, _ = best
            best_steps[d.name] = step
            y_at = float(np.interp(step, x, y))
            # Vertical tick on the seed's own curve; the surface-coloured tick
            # underneath is a 2px ring so it stays legible where curves cross.
            ax.plot([step], [y_at], linestyle="none", marker="|", markersize=14, markeredgewidth=5.5,
                    color=SURFACE, zorder=3)
            ax.plot([step], [y_at], linestyle="none", marker="|", markersize=12, markeredgewidth=1.8,
                    color=color, zorder=4)

    summary: dict = {"n_seeds": len(curves), "window": window}
    if best_steps:
        summary["best_steps"] = best_steps
    if len(curves) > 1:
        x_max = min(x[-1] for _, x, _ in curves)
        grid = np.linspace(0, x_max, 400)
        stack = np.vstack([np.interp(grid, x, y) for _, x, y in curves])
        mean, std = stack.mean(axis=0), stack.std(axis=0)
        ax.plot(grid, mean, color=TEXT_PRIMARY, linewidth=2.0, label=f"mean of {len(curves)} seeds")
        ax.fill_between(grid, mean - std, mean + std, color=TEXT_PRIMARY, alpha=0.08, linewidth=0)
        summary["final_mean_smoothed"] = float(mean[-1])
        summary["final_std_smoothed"] = float(std[-1])
    if best_steps:
        ax.plot([], [], linestyle="none", marker="|", markersize=10, markeredgewidth=1.8, color=TEXT_PRIMARY,
                label="best checkpoint (held-out eval)")
    # Legend below the axes so it can never cover the data.
    n_labels = len(curves) + (1 if len(curves) > 1 else 0) + (1 if best_steps else 0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=min(5, n_labels),
              frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)

    if title is None:
        parts = [p for p in (env, algo.upper() if algo else "") if p]
        title = " - ".join(parts) if parts else "learning curve"
        title += f" (moving average, window {window} episodes)"
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left")
    ax.set_xlabel("environment steps", color=TEXT_SECONDARY)
    ax.set_ylabel("episode return", color=TEXT_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.ticklabel_format(axis="x", style="plain")
    ax.margins(x=0.03)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    summary["out"] = str(out)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run directories (one per seed)")
    p.add_argument("--out", required=True, help="output PNG path")
    p.add_argument("--window", type=int, default=20, help="moving-average window in episodes")
    p.add_argument("--title", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = plot_runs([Path(r) for r in args.runs], Path(args.out), window=args.window, title=args.title)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
