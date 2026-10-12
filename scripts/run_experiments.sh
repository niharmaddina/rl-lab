#!/usr/bin/env bash
# Launch every experiment reported in the README in the background, one process
# per seed, logging to runs/<env>/<algo>/seed_<k>/train.log. Poll with:
#   tail -n 2 runs/*/*/seed_*/train.log
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

launch() {  # env algo timesteps seed
  local out="runs/$1/$2/seed_$4"
  mkdir -p "$out"
  nohup "$PY" -m rl_lab.train --env "$1" --algo "$2" --timesteps "$3" --seed "$4" --out "$out" \
    > "$out/train.log" 2>&1 &
  echo "started $1 $2 seed=$4 -> $out (pid $!)"
}

for s in 0 1 2; do launch CartPole-v1 dqn 100000 "$s"; done
for s in 0 1 2; do launch Acrobot-v1  dqn 100000 "$s"; done
for s in 0 1 2; do launch LunarLander-v3 ppo "${LUNAR_TIMESTEPS:-1000000}" "$s"; done
wait
echo "all runs finished"
