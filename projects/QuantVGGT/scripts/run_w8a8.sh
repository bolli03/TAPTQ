#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE=co3d DTYPE=quarot_w8a8 EXP=a88 CUDA="${CUDA:-2}" exec "$SCRIPT_DIR/run_quant_eval.sh" "$@"
