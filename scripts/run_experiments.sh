#!/usr/bin/env bash
# Launch every experiment reported in the README in the background, one process
# per seed, logging to runs/<env>/<name>/seed_<k>/train.log. Poll with:
#   tail -n 2 runs/*/*/seed_*/train.log
#
# The body lives in main() and the file ends with the call, so bash parses the
# whole script before anything runs: editing this file while a launched copy is
# still waiting on its children cannot make that copy re-execute launch lines.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

launch() {  # env algo timesteps seed [run-name] [eval-every]
  local out="runs/$1/${5:-$2}/seed_$4"
  mkdir -p "$out"
  nohup "$PY" -m rl_lab.train --env "$1" --algo "$2" --timesteps "$3" --seed "$4" --out "$out" \
    --eval-every "${6:-0}" > "$out/train.log" 2>&1 &
  echo "started $1 $2 seed=$4 -> $out (pid $!)"
}

main() {
  # CartPole 100k keeps the best checkpoint under held-out evaluation every 5k
  # steps (the README's best columns). The other rows were run before
  # --eval-every existed and show a dash there; pass a sixth argument to add it.
  for s in 0 1 2; do launch CartPole-v1 dqn 100000 "$s" dqn 5000; done
  for s in 0 1 2; do launch CartPole-v1 dqn 50000  "$s" dqn-50k; done   # rl-zoo budget, same config
  for s in 0 1 2; do launch Acrobot-v1  dqn 100000 "$s"; done
  for s in 0 1 2; do launch LunarLander-v3 ppo "${LUNAR_TIMESTEPS:-1000000}" "$s"; done
  wait
  echo "all runs finished"
}

main "$@"
