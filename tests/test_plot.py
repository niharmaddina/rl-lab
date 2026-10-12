import json
from pathlib import Path

import numpy as np

from rl_lab import plot

HEADER = '#{"t_start": 1700000000.0, "env_id": "CartPole-v1"}\n'


def write_monitor(d: Path, rows, rank: int = 0):
    d.mkdir(parents=True, exist_ok=True)
    body = "r,l,t\n" + "".join(f"{r},{l},{t}\n" for r, l, t in rows)
    (d / f"{rank}.monitor.csv").write_text(HEADER + body)


def test_load_monitor_cumulative_timesteps(tmp_path: Path):
    write_monitor(tmp_path, [(10, 10, 0.1), (20, 20, 0.2), (30, 30, 0.3)])
    df = plot.load_monitor(tmp_path)
    assert df["timesteps"].tolist() == [10, 30, 60]
    assert df["r"].tolist() == [10, 20, 30]


def test_load_monitor_merges_subenvs_by_time(tmp_path: Path):
    write_monitor(tmp_path, [(1, 5, 0.1), (3, 5, 0.3)], rank=0)
    write_monitor(tmp_path, [(2, 5, 0.2)], rank=1)
    df = plot.load_monitor(tmp_path)
    assert df["r"].tolist() == [1, 2, 3]
    assert df["timesteps"].tolist() == [5, 10, 15]


def test_moving_average():
    out = plot.moving_average(np.array([1.0, 3.0, 5.0, 7.0]), window=2)
    assert out.tolist() == [1.0, 2.0, 4.0, 6.0]


def test_plot_tiny_csv_writes_png(tmp_path: Path, capsys):
    for seed in range(2):
        rows = [(10 * (i + 1) + seed, 10, 0.1 * (i + 1)) for i in range(6)]
        write_monitor(tmp_path / f"seed_{seed}", rows)
    out = tmp_path / "curve.png"
    rc = plot.main(["--runs", str(tmp_path / "seed_0"), str(tmp_path / "seed_1"),
                    "--out", str(out), "--window", "3"])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 1000
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["n_seeds"] == 2 and summary["window"] == 3
    assert "final_mean_smoothed" in summary


def test_best_checkpoint_tick_when_npz_present(tmp_path: Path, capsys):
    for seed in range(2):
        rows = [(10 * (i + 1) + seed, 10, 0.1 * (i + 1)) for i in range(6)]
        write_monitor(tmp_path / f"seed_{seed}", rows)
    # seed_0 has periodic evaluations: best mean (first occurrence of the max) at step 40.
    np.savez(tmp_path / "seed_0" / "evaluations.npz", timesteps=[20, 40, 60],
             results=[[5.0, 7.0], [9.0, 11.0], [10.0, 10.0]], ep_lengths=[[5, 7], [9, 11], [10, 10]])
    assert plot.best_checkpoint(tmp_path / "seed_0") == (40, 10.0)
    assert plot.best_checkpoint(tmp_path / "seed_1") is None

    out = tmp_path / "curve.png"
    rc = plot.main(["--runs", str(tmp_path / "seed_0"), str(tmp_path / "seed_1"), "--out", str(out), "--window", "3"])
    assert rc == 0 and out.exists()
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["best_steps"] == {"seed_0": 40}
