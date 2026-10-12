"""Short CartPole DQN train -> save -> evaluate round trips (a few seconds each on CPU)."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from rl_lab import evaluate, train
from rl_lab.config import CONFIG_DIR, load_config


@pytest.fixture(scope="module")
def smoke_config(tmp_path_factory) -> Path:
    """Repo CartPole config with learning_starts lowered so 500 steps include gradient updates."""
    cfg = load_config(CONFIG_DIR / "CartPole-v1.yaml")
    cfg["algo_kwargs"].update(learning_starts=100, train_freq=4, gradient_steps=1, buffer_size=1000)
    p = tmp_path_factory.mktemp("cfg") / "CartPole-v1.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def run_train(out: Path, smoke_config: Path, seed: int = 0, timesteps: int = 500, extra: list[str] = ()) -> dict:
    return train.train(train.build_parser().parse_args([
        "--env", "CartPole-v1", "--algo", "dqn", "--timesteps", str(timesteps), "--seed", str(seed),
        "--out", str(out), "--config", str(smoke_config), "--eval-episodes", "3", *extra,
    ]))


def test_train_eval_round_trip(tmp_path: Path, smoke_config: Path):
    out = tmp_path / "run"
    result = run_train(out, smoke_config)

    assert (out / "model.zip").exists()
    assert (out / "eval.json").exists()
    assert list(out.glob("*.monitor.csv")), "Monitor CSV should be written to --out"
    assert not (out / "best_model.zip").exists(), "no best checkpoint without --eval-every"
    assert not (out / "evaluations.npz").exists()

    on_disk = json.loads((out / "eval.json").read_text())
    assert on_disk["timesteps"] == 500 and on_disk["algo"] == "dqn" and on_disk["seed"] == 0
    assert on_disk["eval_every"] == 0
    final = on_disk["eval"]["final"]
    assert final["episodes"] == 3
    assert final["seeds"] == [10_000, 10_001, 10_002]
    assert len(final["returns"]) == 3
    assert final["mean_return"] == pytest.approx(result["eval"]["final"]["mean_return"])
    assert final["std_return"] >= 0
    assert on_disk["eval"]["best"] is None and on_disk["eval"]["best_step"] is None
    assert on_disk["eval"]["n_evaluations"] == 0
    assert on_disk["train_wall_time_s"] > 0

    # Reload the saved model through evaluate.py on the same held-out seeds.
    ev = evaluate.evaluate(evaluate.build_parser().parse_args([
        "--model", str(out / "model.zip"), "--env", "CartPole-v1", "--episodes", "3",
        "--out", str(tmp_path / "re-eval.json"),
    ]))
    assert ev["algo"] == "dqn"
    assert ev["returns"] == final["returns"], "deterministic policy on identical seeds must match"
    assert (tmp_path / "re-eval.json").exists()


def test_eval_every_writes_best_checkpoint(tmp_path: Path, smoke_config: Path):
    out = tmp_path / "run"
    result = run_train(out, smoke_config, timesteps=2000, extra=["--eval-every", "500"])

    assert (out / "best_model.zip").exists()
    assert (out / "evaluations.npz").exists()
    npz = np.load(out / "evaluations.npz")
    assert npz["timesteps"].tolist() == [500, 1000, 1500, 2000]
    assert npz["results"].shape == (4, 3) and npz["ep_lengths"].shape == (4, 3)
    assert npz["seeds"].tolist() == [10_000, 10_001, 10_002], "periodic evaluation uses the held-out seeds"

    on_disk = json.loads((out / "eval.json").read_text())
    ev = on_disk["eval"]
    assert on_disk["eval_every"] == 500 and ev["n_evaluations"] == 4
    assert ev["final"]["seeds"] == [10_000, 10_001, 10_002]
    assert ev["best"]["seeds"] == [10_000, 10_001, 10_002]
    assert ev["best_step"] in npz["timesteps"].tolist()
    assert ev["best"]["mean_return"] == pytest.approx(result["eval"]["best"]["mean_return"])

    # best_model.zip reloaded and re-evaluated on the same seeds reproduces the
    # best periodic evaluation exactly (same protocol, same seeds, same weights).
    means = npz["results"].mean(axis=1)
    assert ev["best"]["mean_return"] == pytest.approx(means.max())
    assert ev["best_step"] == int(npz["timesteps"][int(np.argmax(means))])
    assert ev["best"]["returns"] == npz["results"][int(np.argmax(means))].tolist()

    # The same checkpoint is also scored on seeds it was not selected on.
    assert ev["best_fresh_seeds"]["seeds"] == [20_000, 20_001, 20_002]
    assert on_disk["in_training_eval_wall_time_s"] > 0

    # evaluate.py works on the best checkpoint too (algo read from the sidecar eval.json).
    re = evaluate.evaluate(evaluate.build_parser().parse_args([
        "--model", str(out / "best_model.zip"), "--env", "CartPole-v1", "--episodes", "3",
    ]))
    assert re["returns"] == ev["best"]["returns"]


def test_same_seed_is_reproducible(tmp_path: Path, smoke_config: Path):
    a = run_train(tmp_path / "a", smoke_config, seed=1)
    b = run_train(tmp_path / "b", smoke_config, seed=1)
    assert a["eval"]["final"]["returns"] == b["eval"]["final"]["returns"]


def test_eval_every_does_not_change_training(tmp_path: Path, smoke_config: Path):
    """The periodic evaluation only reads the policy; the final weights are those
    of a run without --eval-every."""
    a = run_train(tmp_path / "a", smoke_config, seed=1, timesteps=1000)
    b = run_train(tmp_path / "b", smoke_config, seed=1, timesteps=1000, extra=["--eval-every", "500"])
    assert a["eval"]["final"]["returns"] == b["eval"]["final"]["returns"]


def test_env_mismatch_rejected(tmp_path: Path, smoke_config: Path):
    with pytest.raises(ValueError, match="does not match"):
        train.train(train.build_parser().parse_args([
            "--env", "Acrobot-v1", "--out", str(tmp_path / "x"), "--config", str(smoke_config),
        ]))
