#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE=co3d DTYPE=quarot_w4a8 EXP=a48 CUDA="${CUDA:-1}" RESUME=0 exec "$SCRIPT_DIR/run_quant_eval.sh" "$@"
