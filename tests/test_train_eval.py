"""500-step CartPole DQN train -> save -> evaluate round trip (a few seconds on CPU)."""
import json
from pathlib import Path

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


def run_train(out: Path, smoke_config: Path, seed: int = 0) -> dict:
    return train.train(train.build_parser().parse_args([
        "--env", "CartPole-v1", "--algo", "dqn", "--timesteps", "500", "--seed", str(seed),
        "--out", str(out), "--config", str(smoke_config), "--eval-episodes", "3",
    ]))


def test_train_eval_round_trip(tmp_path: Path, smoke_config: Path):
    out = tmp_path / "run"
    result = run_train(out, smoke_config)

    assert (out / "model.zip").exists()
    assert (out / "eval.json").exists()
    assert list(out.glob("*.monitor.csv")), "Monitor CSV should be written to --out"

    on_disk = json.loads((out / "eval.json").read_text())
    assert on_disk["timesteps"] == 500 and on_disk["algo"] == "dqn" and on_disk["seed"] == 0
    assert on_disk["eval"]["episodes"] == 3
    assert on_disk["eval"]["seeds"] == [10_000, 10_001, 10_002]
    assert len(on_disk["eval"]["returns"]) == 3
    assert on_disk["eval"]["mean_return"] == pytest.approx(result["eval"]["mean_return"])
    assert on_disk["eval"]["std_return"] >= 0
    assert on_disk["train_wall_time_s"] > 0

    # Reload the saved model through evaluate.py on the same held-out seeds.
    ev = evaluate.evaluate(evaluate.build_parser().parse_args([
        "--model", str(out / "model.zip"), "--env", "CartPole-v1", "--episodes", "3",
        "--out", str(tmp_path / "re-eval.json"),
    ]))
    assert ev["algo"] == "dqn"
    assert ev["returns"] == on_disk["eval"]["returns"], "deterministic policy on identical seeds must match"
    assert (tmp_path / "re-eval.json").exists()


def test_same_seed_is_reproducible(tmp_path: Path, smoke_config: Path):
    a = run_train(tmp_path / "a", smoke_config, seed=1)
    b = run_train(tmp_path / "b", smoke_config, seed=1)
    assert a["eval"]["returns"] == b["eval"]["returns"]


def test_env_mismatch_rejected(tmp_path: Path, smoke_config: Path):
    with pytest.raises(ValueError, match="does not match"):
        train.train(train.build_parser().parse_args([
            "--env", "Acrobot-v1", "--out", str(tmp_path / "x"), "--config", str(smoke_config),
        ]))
