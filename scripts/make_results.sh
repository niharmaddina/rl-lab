#!/usr/bin/env bash
# Build results/<env>/curve.png for every env under runs/ and results/summary.json.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
for envdir in runs/*/; do
  env="$(basename "$envdir")"
  for algodir in "$envdir"*/; do
    "$PY" -m rl_lab.plot --runs "$algodir"seed_* --out "results/$env/curve.png" --window 20
  done
done
"$PY" -m rl_lab.summarize --runs-root runs --out results/summary.json --hardware "${HARDWARE:-Apple M1 Pro, CPU}"
