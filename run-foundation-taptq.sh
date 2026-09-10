#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?usage: $0 <vggt_omega|d4rt> [smoke|calib|calib_compensate|test]}
MODE=${2:-smoke}
shift $(( $# >= 2 ? 2 : 1 ))
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_ROOT="$ROOT/Pi3-evaluation"

case "$MODEL" in
  vggt_omega)
    export VGGT_OMEGA_MODEL_PATH=${VGGT_OMEGA_MODEL_PATH:-$ROOT/models/vggt_omega_1b_512.pt}
    if [[ ! -f "$VGGT_OMEGA_MODEL_PATH" ]]; then
      echo "VGGT-Omega checkpoint missing: $VGGT_OMEGA_MODEL_PATH" >&2
      echo "Request access: https://huggingface.co/facebook/VGGT-Omega" >&2
      exit 2
    fi
    EXTRA=(foundation.image_resolution=512)
    ;;
  d4rt)
    export D4RT_MODEL_PATH=${D4RT_MODEL_PATH:-$ROOT/models/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt}
    if [[ -z "$D4RT_MODEL_PATH" || ! -f "$D4RT_MODEL_PATH" ]]; then
      if [[ "$MODE" != smoke ]]; then
        echo "D4RT checkpoint missing; set D4RT_MODEL_PATH for $MODE." >&2
        exit 2
      fi
      export D4RT_ALLOW_RANDOM_INIT=1
      EXTRA=(foundation.image_resolution=256 foundation.temporal_size=10 foundation.query_grid=16 foundation.allow_random_init=true)
    else
      export D4RT_CONFIG_PATH=${D4RT_CONFIG_PATH:-$(dirname "$D4RT_MODEL_PATH")/model.yaml}
      EXTRA=(foundation.image_resolution=256 foundation.temporal_size=32 foundation.query_grid=16)
    fi
    ;;
  *)
    echo "unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

export TAPTQ_MODEL=$MODEL
export TAPTQ_BITS=${TAPTQ_BITS:-4,8}
export TAPTQ_METRIC=${TAPTQ_METRIC:-hessian}
export TAPTQ_CONFIG=${TAPTQ_CONFIG:-PTQ4ViT}

if [[ "$MODE" == "calib_compensate" || "$MODE" == "compensate_eval" || "$MODE" == "e2e" ]]; then
  if [[ "$MODEL" == "d4rt" ]]; then
    echo "D4RT layer QwT is disabled until a trained community checkpoint is supplied." >&2
    exit 2
  fi
  EXTRA+=(compensate.strategy=layer)
fi

PYTHON_BIN=${PYTHON_BIN:-python}
cd "$EVAL_ROOT"
exec "$PYTHON_BIN" mv_recon/taptq.py \
  evaluation=mv_recon \
  model_name="$MODEL" \
  mode="$MODE" \
  "${EXTRA[@]}" \
  "$@"
