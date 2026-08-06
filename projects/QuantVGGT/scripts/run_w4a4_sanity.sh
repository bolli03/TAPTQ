#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE=co3d DTYPE=quarot_w4a4 EXP=a44 CUDA="${CUDA:-0}" RESUME=1 exec "$SCRIPT_DIR/run_quant_eval.sh" "$@"
