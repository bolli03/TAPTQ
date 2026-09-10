#!/usr/bin/env bash
set -euo pipefail
GPU=${1:?GPU index required}
TASK=${2:?task name required}
W=/tmp/tmm-eval-code
R=/mnt/cephfs/pansicheng/tmm-results/eval/7scenes_pi3_kf40_recheck
C=/mnt/cephfs/pansicheng/tmm-results/checkpoints
V=/mnt/cephfs/pansicheng/tmm-venv/bin/python
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$W"
export LD_LIBRARY_PATH="/tmp/tmm-libs:${LD_LIBRARY_PATH:-}"
export TMM_DATA_ROOT=/mnt/cephfs/pansicheng/data
export TMM_CACHE_ROOT=/mnt/cephfs/pansicheng/cache
export VGGT_MODEL_PATH=/mnt/cephfs/pansicheng/models/models--facebook--VGGT-1B
export TAPTQ_MODEL=vggt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$R/logs" "$R/$TASK"
cd "$W"
common=(evaluation=mv_recon model_name=vggt "eval_datasets=[7scenes-dense]" output_dir="$R/$TASK" save_suffix="$TASK")
case "$TASK" in
  ptq4vit_w6a6|ptq4vit_w8a8)
    bits=${TASK#ptq4vit_w}; wb=${bits%a*}; ab=${bits#*a}
    export TAPTQ_BITS="$wb,$ab" TAPTQ_CONFIG=PTQ4ViT
    exec "$V" mv_recon/taptq.py "${common[@]}" mode=test "ptq.bit=[$wb,$ab]" ptq.search_mode=ternary ptq.linear_channelwise=false ckpt.path="$C/ptq4vit_dtu8_w${wb}a${ab}.pt"
    ;;
  taptq_w6a6|taptq_w8a8)
    bits=${TASK#taptq_w}; wb=${bits%a*}; ab=${bits#*a}
    export TAPTQ_BITS="$wb,$ab" TAPTQ_CONFIG=PTQ4ViT
    exec "$V" mv_recon/taptq.py "${common[@]}" mode=compensate_eval "ptq.bit=[$wb,$ab]" ptq.search_mode=ternary ptq.linear_channelwise=false "optim_datasets=[DTU_train_8]" ckpt.path="$C/w${wb}a${ab}_ternary.pt" ++compensate.strategy=module ++compensate.tail_ratio=0.1 ++compensate.tau_thr=0.007 ++compensate.rank=16 ++compensate.skip_p=0.0
    ;;
  *) echo "Unknown task: $TASK" >&2; exit 2 ;;
esac
