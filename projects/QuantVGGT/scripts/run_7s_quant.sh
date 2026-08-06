#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE=7s DTYPE="${DTYPE:-quarot_w4a4}" EXP="${EXP:-a44}" CUDA="${CUDA:-0}" \
  KF="${KF:-100}" RESUME="${RESUME:-1}" SCENE_ROOT="${SCENE_ROOT:-}" \
  exec "$SCRIPT_DIR/run_quant_eval.sh" "$@"
