#!/usr/bin/env bash
set -Eeuo pipefail
GPU=${1:?GPU index required}
TASK=${2:?task name required}
W=/tmp/tmm-eval-code
R=/mnt/cephfs/pansicheng/tmm-results/eval/eth3d_pi3_kf5_validation
C=/mnt/cephfs/pansicheng/tmm-results/checkpoints
V=/mnt/cephfs/pansicheng/tmm-venv/bin/python
ETH3D=/mnt/cephfs/pansicheng/data/eth3d_pi3_final

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$W"
export LD_LIBRARY_PATH="/tmp/tmm-libs:${LD_LIBRARY_PATH:-}"
export TMM_DATA_ROOT=/mnt/cephfs/pansicheng/data
export TMM_CACHE_ROOT=/mnt/cephfs/pansicheng/cache/eth3d_pi3_kf5_validation
export VGGT_MODEL_PATH=/mnt/cephfs/pansicheng/models/models--facebook--VGGT-1B
export TAPTQ_MODEL=vggt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -f "$ETH3D/.READY"
mkdir -p "$R/logs" "$R/$TASK" "$TMM_CACHE_ROOT"
cd "$W"
common=(evaluation=mv_recon model_name=vggt "eval_datasets=[ETH3D]" "data.ETH3D.cfg.ETH3D_DIR=$ETH3D" "data.ETH3D.cfg.cache_file=$TMM_CACHE_ROOT/eth3d_mv_recon_cache.npy" output_dir="$R/$TASK" save_suffix="$TASK")

case "$TASK" in
  fp)
    export TAPTQ_BITS=4,8 TAPTQ_CONFIG=PTQ4ViT
    exec "$V" mv_recon/taptq.py "${common[@]}" mode=fp_eval "ptq.bit=[4,8]" ptq.search_mode=ternary ptq.linear_channelwise=false
    ;;
  taptq_w4a8|taptq_w8a8)
    bits=${TASK#taptq_w}; wb=${bits%a*}; ab=${bits#*a}; rank=16
    [[ "$wb" == 4 && "$ab" == 8 ]] && rank=128
    export TAPTQ_BITS="$wb,$ab" TAPTQ_CONFIG=PTQ4ViT
    exec "$V" mv_recon/taptq.py "${common[@]}" mode=compensate_eval "ptq.bit=[$wb,$ab]" ptq.search_mode=ternary ptq.linear_channelwise=false "optim_datasets=[DTU_train_8]" ckpt.path="$C/w${wb}a${ab}_ternary.pt" ++compensate.strategy=module ++compensate.tail_ratio=0.1 ++compensate.tau_thr=0.007 ++compensate.rank="$rank" ++compensate.skip_p=0.0
    ;;
  *) echo "Unknown task: $TASK" >&2; exit 2 ;;
esac
