#!/usr/bin/env bash
# Usage: ./run.sh [stage]   (stage defaults to "m0")
set -euo pipefail

cd "$(dirname "$0")"

STAGE="${1:-m0}"

if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
fi

case "$STAGE" in
  m0)
    "$PYTHON" -m src.m0_data --config configs/data.yaml
    ;;
  m1)
    "$PYTHON" -m src.train_teacher --config configs/teacher.yaml
    ;;
  m2)
    "$PYTHON" -m src.baseline_onestep --config configs/baseline.yaml
    ;;
  m3)
    "$PYTHON" -m src.train_dmd --config configs/dmd.yaml
    ;;
  *)
    echo "Unknown stage: $STAGE (expected 'm0', 'm1', 'm2', or 'm3')" >&2
    exit 1
    ;;
esac
