#!/usr/bin/env bash
# Build results/<env>/curve.png for every env under runs/ and results/summary.json.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
for envdir in runs/*/; do
  env="$(basename "$envdir")"
  for algodir in "$envdir"*/; do
    name="$(basename "$algodir")"           # dqn / ppo -> curve.png; dqn-50k -> curve-dqn-50k.png
    case "$name" in dqn|ppo) png="curve.png" ;; *) png="curve-$name.png" ;; esac
    "$PY" -m rl_lab.plot --runs "$algodir"seed_* --out "results/$env/$png" --window 20
  done
done
"$PY" -m rl_lab.summarize --runs-root runs --out results/summary.json --hardware "${HARDWARE:-Apple M1 Pro, CPU}"
