# rl-lab

One clean, reproducible home for small reinforcement-learning experiments:
DQN and PPO from stable-baselines3 on Gymnasium tasks, with seeds, per-task
YAML configs, Monitor logs, learning curves and a results table that is
regenerated from the run directories rather than typed by hand.

The repository replaces three stand-alone scripts written in 2024 (DQN on
CartPole, PPO on CarRacing with `VecTransposeImage`, PPO on Humanoid/CartPole
with `make_vec_env`). Those scripts hard-coded one run each, rendered to a
window, assumed a CUDA device and saved nothing but the weights. Everything
here runs headless on a CPU, is driven from the command line, records what it
did, and evaluates every trained policy on seeds it never trained from.

## What is here

```
rl_lab/
  train.py       train one (env, algo, seed) run; writes monitor csv, model.zip, eval.json
  evaluate.py    load a model.zip and evaluate it on held-out seeds
  plot.py        learning curves across seed directories (moving average, PNG)
  summarize.py   aggregate eval.json files into results/summary.json + a Markdown table
  config.py      load/validate configs/<env>.yaml
  common.py      algorithm registry, env construction, seeded evaluation
configs/         CartPole-v1.yaml (dqn), Acrobot-v1.yaml (dqn), LunarLander-v3.yaml (ppo)
tests/           pytest smoke tests (config loading, 500-step train+eval round trip, plotting)
scripts/         run_experiments.sh (launch everything in the README), make_results.sh (plots + summary)
results/         committed curves and summary.json; the runs/ directory with models is not committed
.github/         workflow that runs the tests on every push
```

## Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # exact versions used for the results below
pip install -e .                        # adds rl-train / rl-evaluate / rl-plot / rl-summarize
```

`requirements.txt` pins the exact versions behind every number in this README
(torch 2.14.1, gymnasium 1.4.0, stable-baselines3 2.9.0, Python 3.11.3).
LunarLander-v3 needs `gymnasium[box2d]`; on macOS arm64 the `Box2D` wheel
installed from pip without a swig build. If box2d does not install on your
machine, the two DQN tasks still run (classic control has no extra
dependencies).

## Use

Train one run (the config supplies the algorithm kwargs; CLI flags override
`timesteps`, `n_envs`, `eval_episodes`):

```bash
python -m rl_lab.train --env CartPole-v1 --algo dqn --timesteps 100000 --seed 0 \
    --out runs/CartPole-v1/dqn/seed_0
```

The run directory then contains `0.monitor.csv` (every training episode:
return, length, wall time), `model.zip`, `config.yaml` (the resolved config)
and `eval.json` (mean/std return over 20 held-out seeds plus timings and
library versions).

Evaluate a saved model (20 episodes, seeds 10000..10019 by default; the
algorithm is read from the sidecar `eval.json`):

```bash
python -m rl_lab.evaluate --model runs/CartPole-v1/dqn/seed_0/model.zip --env CartPole-v1 --episodes 20
```

Plot learning curves for a set of seeds and aggregate a results table:

```bash
python -m rl_lab.plot --runs runs/CartPole-v1/dqn/seed_* --out results/CartPole-v1/curve.png --window 20
python -m rl_lab.summarize --runs-root runs --out results/summary.json --hardware "Apple M1 Pro, CPU"
```

Reproduce everything in this README (twelve background processes, one per
seed; about 15 minutes wall-clock on the hardware below, the longest single
run being LunarLander seed 1 at 13.0 minutes) and rebuild the figures:

```bash
PYTHON=.venv/bin/python scripts/run_experiments.sh
PYTHON=.venv/bin/python scripts/make_results.sh
```

Run the tests (14 tests, about 5 s on CPU; the round-trip test trains a 500-step DQN):

```bash
pytest
```

## Seeds and evaluation protocol

* Training seed `k` seeds Python/NumPy/torch (`set_random_seed`), the
  environment reset, and the SB3 algorithm. Two runs with the same seed and
  config produce identical evaluation returns on this machine
  (`tests/test_train_eval.py::test_same_seed_is_reproducible`). This also
  held at full scale: the three LunarLander-v3 runs were trained twice end to
  end (the second time by accident, from a launcher script edited while the
  first launch was still running; `scripts/run_experiments.sh` is now written
  so that cannot recur), and the second pass matched the first at every
  checkpoint recorded from it (the 20-episode moving average at 250k and 500k
  steps, the step at which each seed first crossed a return of 200) and
  reproduced all 60 evaluation returns exactly.
* Evaluation uses a fresh, unwrapped environment and 20 full episodes with
  reset seeds 10000..10019, disjoint from the training seeds 0, 1, 2. Actions
  are deterministic (argmax for DQN, the mean action for PPO).
* `final mean return` in the table is the mean over seeds of each seed's
  20-episode mean; the `+/-` is the standard deviation of those per-seed
  means (so it measures seed-to-seed variation, not episode noise).
  `results/summary.json` also carries the pooled statistics over all 60
  episodes and the per-seed values.

## Results

Every number below was measured on 2026-10-12 on an Apple M1 Pro, CPU only
(one torch thread per run; the wall times were taken while up to nine runs
shared the machine), with the versions in `requirements.txt`. Each row is 3
training seeds (0, 1, 2); each seed is evaluated for 20 episodes on held-out
reset seeds 10000..10019 with deterministic actions. The table is the output of `rl_lab.summarize`; the
same numbers, per seed and per episode, are in `results/summary.json`.

| Env | Algo | Seeds | Timesteps / seed | Eval return (mean +/- std across seeds) | Per-seed eval means | Wall time / seed |
|---|---|---|---|---|---|---|
| CartPole-v1 | DQN | 3 | 100,000 | 106.1 +/- 68.2 | 42.6, 75.0, 200.7 | 1.1 min |
| CartPole-v1 | DQN | 3 | 50,000 | 280.4 +/- 155.4 | 163.6, 500.0, 177.5 | 0.4 min |
| Acrobot-v1 | DQN | 3 | 100,000 | -74.3 +/- 5.1 | -67.8, -75.0, -80.3 | 2.0 min |
| LunarLander-v3 | PPO | 3 | 1,000,000 | 278.5 +/- 1.6 | 277.2, 277.5, 280.8 | 12.0 min |

For scale: CartPole-v1 returns are capped at 500 per episode; Acrobot-v1 pays
-1 per step until the tip reaches the target height (-500 if it never does);
LunarLander-v3 counts an episode return of 200 as solved.

### Learning curves

Training-episode return (with the exploration noise of training, so below the
deterministic evaluation numbers above), 20-episode moving average, one thin
line per seed, thick line the mean across seeds with a +/- 1 std band.

![CartPole-v1 DQN, 100k steps](results/CartPole-v1/curve.png)

![CartPole-v1 DQN, 50k steps](results/CartPole-v1/curve-dqn-50k.png)

![Acrobot-v1 DQN, 100k steps](results/Acrobot-v1/curve.png)

![LunarLander-v3 PPO, 1M steps](results/LunarLander-v3/curve.png)

### What converged and what did not

* **Acrobot-v1 / DQN converged on every seed.** The curves rise from -500 to
  about -100 within 25k steps and stay flat; the final deterministic policies
  score -67.8, -75.0 and -80.3, a spread of 5.1 across seeds. The training
  moving average at 100k (about -86 to -88) sits below the evaluation number
  only because training still acts with epsilon 0.1.
* **CartPole-v1 / DQN did not converge to a stable policy at 100k steps.** On
  all three seeds the training moving average climbs to a peak of 298-466
  between 43k and 54k steps, then collapses to under 100 and oscillates for
  the rest of training; the final weights score 42.6, 75.0 and 200.7 on the
  held-out seeds. The 2024 script trained for this budget and never measured
  the result; measured, the final policy is poor. This is the known instability of DQN on CartPole with this configuration
  (rl-baselines3-zoo trains it for 50k steps, which is where the peaks are).
* **The 50k-step rerun confirms the problem is stability, not budget.** Same
  config, stopped at the zoo budget: seed 1 scores a perfect 500.0 +/- 0.0 on
  all 20 evaluation episodes, while seeds 0 and 2 were in the middle of a
  collapse when training stopped and score 163.6 and 177.5. The final policy
  depends on when training happens to stop. The fix is to keep the best
  checkpoint under periodic held-out evaluation (an `EvalCallback`), which is
  the first roadmap item, not to tune the budget.
* **LunarLander-v3 / PPO converged on every seed.** The final policies score
  277.2, 277.5 and 280.8 (a spread of 1.6 across seeds), and every one of the
  60 evaluation episodes lands with a return above 250 (minimum 250.6, maximum
  311.0), well past the 200 that counts as solved. The seeds differ in how
  fast they got there: the 20-episode training average first crossed 200 at
  about 361k steps (seed 2) and 364k steps (seed 0), but only at about 577k
  steps for seed 1, which sat near 130 between 250k and 500k steps before
  taking off. By 1M steps the training average of every seed is between 271
  and 280. The PPO settings lifted from the 2024 CarRacing script
  (n_steps 2048, batch 64, lr 2.5e-4, 4 envs, 64x64 networks) transfer to
  this task without change.

The repository reports what the final weights do; it does not cherry-pick the
best seed or the best moment of training.

## Where the hyperparameters come from

* **CartPole-v1 / DQN.** The 2024 `CartPole/main.py` used `DQN("MlpPolicy", env)`
  with stable-baselines3 defaults and `total_timesteps=100000`. The MLP policy
  and the 100k budget are kept. The algorithm kwargs are the
  rl-baselines3-zoo CartPole-v1 DQN values (lr 2.3e-3, batch 64, buffer 100k,
  target update every 10 steps, train_freq 256 with 128 gradient steps,
  exploration fraction 0.16 to eps 0.04, two 256-unit layers); the SB3
  defaults are tuned for Atari (lr 1e-4, target update every 10k steps, 1M
  buffer) and are known to learn CartPole slowly and unstably.
* **LunarLander-v3 / PPO.** The 2024 `CarRacing/main.py` used PPO with
  `n_steps=2048`, `batch_size=64`, `learning_rate=2.5e-4`, four parallel
  environments and `net_arch=dict(pi=[64, 64], vf=[64, 64])`. All of those are
  reused verbatim. Its `use_sde=True` only applies to continuous action spaces
  and is dropped; `gamma=0.999`, `gae_lambda=0.98` and `ent_coef=0.01` are the
  rl-baselines3-zoo LunarLander values. CarRacing itself (pixel observations,
  `CnnPolicy`, `VecTransposeImage`) is not run here: it takes hours on a CPU
  and would dominate the budget of a repo meant to be re-run in minutes.
* **Acrobot-v1 / DQN.** No 2024 script covered it; the kwargs are the
  rl-baselines3-zoo Acrobot-v1 DQN values. Acrobot was chosen over
  MountainCar-v0 because epsilon-greedy DQN reliably solves it within 100k
  steps, whereas MountainCar's sparse reward usually needs more exploration
  machinery than this repository ships.
* The Humanoid script (`Humanoid/popcorn1.py`) actually trained PPO on
  CartPole with `make_vec_env`; its only reusable idea, the vectorised
  environment with Monitor wrappers, is what `rl_lab.common.make_training_env`
  does.

## Roadmap

* Add the original pixel-based CarRacing-v3 PPO config behind an optional
  extra, with a frame-skip wrapper, and report it on a GPU box.
* MountainCar-v0 with a count-based or RND exploration bonus, to make the
  second DQN task harder than Acrobot.
* Hyperparameter sweeps driven from the same YAML files (a `sweep:` block
  expanded into one run directory per setting) and a `--resume` flag.
* Evaluation callbacks during training, so the curve shows held-out return
  over time rather than only the training-episode return.
* A DQN-defaults baseline next to the tuned CartPole run, to quantify how much
  the zoo hyperparameters are worth on this task.

## License

MIT, Copyright (c) 2024-2026 Nihar Maddina. See `LICENSE`.
