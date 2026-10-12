"""summarize.collect / markdown_table on both eval.json layouts."""
import json
from pathlib import Path

from rl_lab.summarize import collect, eval_blocks, format_step, markdown_table

RUNTIME = {"date_utc": "2026-10-12", "python": "3.11.3", "torch": "2.14.1", "gymnasium": "1.4.0",
           "stable_baselines3": "2.9.0"}


def block(returns, seed_base=10_000):
    import numpy as np
    r = np.asarray(returns, dtype=float)
    return {"env": "CartPole-v1", "episodes": len(returns), "deterministic": True,
            "seeds": [seed_base + i for i in range(len(returns))], "returns": list(returns),
            "lengths": [int(x) for x in returns], "mean_return": float(r.mean()), "std_return": float(r.std()),
            "min_return": float(r.min()), "max_return": float(r.max()), "mean_length": float(r.mean())}


def write_run(d: Path, seed: int, final, best=None, best_step=None, timesteps=1000, old_layout=False):
    d.mkdir(parents=True)
    if old_layout:
        ev = block(final)
    else:
        ev = {"final": block(final), "best": block(best) if best else None, "best_step": best_step,
              "best_fresh_seeds": block(best, 20_000) if best else None, "n_evaluations": 2 if best else 0}
    rec = {"env": "CartPole-v1", "algo": "dqn", "seed": seed, "timesteps": timesteps, "n_envs": 1,
           "eval_every": 500 if best else 0, "train_wall_time_s": 60.0, "eval_wall_time_s": 1.0, "fps": 1000.0,
           "eval": ev, "config": {}, "runtime": RUNTIME}
    (d / "eval.json").write_text(json.dumps(rec))


def test_eval_blocks_accepts_both_layouts():
    final = block([1, 2, 3])
    assert eval_blocks({"eval": final}) == (final, None, None, None)
    new = {"eval": {"final": final, "best": block([5, 5]), "best_step": 500, "best_fresh_seeds": block([4, 4])}}
    f, b, step, fresh = eval_blocks(new)
    assert f is final and b["mean_return"] == 5 and step == 500 and fresh["mean_return"] == 4


def test_format_step():
    assert format_step(45_000) == "45k" and format_step(1_500) == "1.5k" and format_step(800) == "800"


def test_collect_and_table_with_and_without_best(tmp_path: Path):
    runs = tmp_path / "runs"
    # Row 1: new layout with best checkpoints, 2 seeds.
    write_run(runs / "CartPole-v1" / "dqn" / "seed_0", 0, final=[10, 20], best=[500, 500], best_step=45_000,
              timesteps=100_000)
    write_run(runs / "CartPole-v1" / "dqn" / "seed_1", 1, final=[30, 40], best=[400, 500], best_step=1_500,
              timesteps=100_000)
    # Row 2: pre --eval-every layout, no best checkpoint.
    write_run(runs / "CartPole-v1" / "dqn-50k" / "seed_0", 0, final=[100, 200], timesteps=50_000, old_layout=True)

    entries = collect(runs)
    assert [e["timesteps"] for e in entries] == [100_000, 50_000]
    e_best, e_old = entries

    assert e_best["final_mean_return"] == 25.0 and e_best["best_mean_return"] == 475.0
    assert e_best["best_std_return_across_seeds"] == 25.0
    assert e_best["best_fresh_seeds_mean_return"] == 475.0
    assert [p["best_step"] for p in e_best["per_seed"]] == [45_000, 1_500]
    assert e_best["eval_every"] == 500

    assert e_old["final_mean_return"] == 150.0
    assert e_old["best_mean_return"] is None and e_old["best_std_return_across_seeds"] is None
    assert e_old["per_seed"][0]["best_step"] is None and e_old["eval_every"] == 0

    table = markdown_table(entries).splitlines()
    assert table[0].count("|") == table[1].count("|") == 10
    assert "| 500.0 @45k, 450.0 @1.5k |" in table[2] and "| 475.0 +/- 25.0 |" in table[2]
    assert table[3].endswith("| 150.0 +/- 0.0 | 150.0 | - | - | 1.0 min |")
