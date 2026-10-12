from pathlib import Path

import pytest
import yaml

from rl_lab.common import ALGOS
from rl_lab.config import CONFIG_DIR, config_path, load_config

CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def test_configs_exist():
    assert CONFIGS, f"no configs in {CONFIG_DIR}"


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_config_loads_and_validates(path: Path):
    cfg = load_config(path)
    assert cfg["env"] == path.stem, "config file must be named <env_id>.yaml"
    assert cfg["algo"] in ALGOS
    assert isinstance(cfg["timesteps"], int) and cfg["timesteps"] > 0
    assert cfg["eval_episodes"] == 20
    assert cfg["algo_kwargs"]["policy"] == "MlpPolicy"
    lr = cfg["algo_kwargs"]["learning_rate"]
    assert isinstance(lr, float) and 0 < lr < 1, "learning_rate must parse as a float, not a string"


def test_config_path_resolution():
    assert config_path("CartPole-v1") == CONFIG_DIR / "CartPole-v1.yaml"
    assert config_path("X-v0", "/tmp/cfgs") == Path("/tmp/cfgs/X-v0.yaml")


def test_scientific_notation_string_is_coerced(tmp_path: Path):
    p = tmp_path / "Foo-v0.yaml"
    p.write_text(yaml.safe_dump({"env": "Foo-v0", "algo": "dqn", "timesteps": 10,
                                 "algo_kwargs": {"policy": "MlpPolicy", "learning_rate": "1e-4"}}))
    cfg = load_config(p)
    assert cfg["algo_kwargs"]["learning_rate"] == pytest.approx(1e-4)
    assert cfg["n_envs"] == 1 and cfg["eval_seed_base"] == 10_000


def test_missing_keys_rejected(tmp_path: Path):
    p = tmp_path / "Bar-v0.yaml"
    p.write_text("env: Bar-v0\nalgo: dqn\n")
    with pytest.raises(ValueError, match="missing required keys"):
        load_config(p)
    p.write_text("env: Bar-v0\nalgo: sac\ntimesteps: 1\nalgo_kwargs: {policy: MlpPolicy}\n")
    with pytest.raises(ValueError, match="algo must be one of"):
        load_config(p)
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
