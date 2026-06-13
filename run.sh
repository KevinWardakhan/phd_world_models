#!/usr/bin/env bash
# Usage: ./run.sh [stage]
#   stage in {m0, m1, m2, m3, m4, m5}; omit it (or pass "all") to run the whole
#   pipeline in order: m0 -> m1 -> m2 -> m3 -> m4 -> m5.
set -euo pipefail

cd "$(dirname "$0")"

STAGE="${1:-all}"

VENV_DIR=".venv"
PYTHON="$VENV_DIR/bin/python"
STAMP="$VENV_DIR/.requirements.installed"

if [ ! -x "$PYTHON" ]; then
  echo "=== Creating virtualenv ($VENV_DIR) ==="
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install -r requirements.txt
fi
PYTHON="$VENV_DIR/bin/python"

run_stage() {
  case "$1" in
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
    m4)
      "$PYTHON" -m src.train_dmd2 --config configs/dmd2.yaml
      ;;
    m5)
      "$PYTHON" -m src.train_dmd2_fewstep --config configs/dmd2_fewstep.yaml
      ;;
    *)
      echo "Unknown stage: $1 (expected 'm0', 'm1', 'm2', 'm3', 'm4', 'm5', or 'all')" >&2
      return 1
      ;;
  esac
}

if [ "$STAGE" = "all" ]; then
  for s in m0 m1 m2 m3 m4 m5; do
    echo "=== Running stage $s ==="
    run_stage "$s"
  done
else
  run_stage "$STAGE"
fi
