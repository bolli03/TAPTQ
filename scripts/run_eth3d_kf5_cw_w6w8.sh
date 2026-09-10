#!/usr/bin/env bash
set -Eeuo pipefail

GPU=${1:?GPU index required}
TASK=${2:?task required}
W=/tmp/tmm-eval-code
R=/mnt/cephfs/pansicheng/tmm-results/eval/eth3d_kf5_w6w8_cw_recheck
C=/mnt/cephfs/pansicheng/tmm-results/checkpoints
V=/mnt/cephfs/pansicheng/tmm-venv/bin/python
ETH3D=/mnt/cephfs/pansicheng/data/eth3d_pi3_final
CACHE=/mnt/cephfs/pansicheng/cache/eth3d_kf5_campaign

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$W"
export LD_LIBRARY_PATH="/tmp/tmm-libs/usr/lib/x86_64-linux-gnu:/tmp/tmm-libs:${LD_LIBRARY_PATH:-}"
export TMM_DATA_ROOT=/mnt/cephfs/pansicheng/data
export TMM_CACHE_ROOT="$CACHE"
export VGGT_MODEL_PATH=/mnt/cephfs/pansicheng/models/models--facebook--VGGT-1B
export TAPTQ_MODEL=vggt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -f "$ETH3D/.READY"
mkdir -p "$R/$TASK" "$R/logs" "$CACHE"
cd "$W"
common=(evaluation=mv_recon model_name=vggt "eval_datasets=[ETH3D]" "data.ETH3D.cfg.ETH3D_DIR=$ETH3D" "data.ETH3D.cfg.cache_file=$CACHE/eth3d_mv_recon_cache.npy" output_dir="$R/$TASK" save_suffix="$TASK")

case "$TASK" in
  nc_w6a6)
    export TAPTQ_BITS=6,6 TAPTQ_CONFIG=PTQ4ViT
    exec "$V" mv_recon/taptq.py "${common[@]}" mode=compensate_eval "ptq.bit=[6,6]" ptq.search_mode=ternary ptq.linear_channelwise=false "optim_datasets=[DTU_train_8]" ckpt.path="$C/w6a6_ternary.pt" ++compensate.strategy=module ++compensate.tail_ratio=0.1 ++compensate.tau_thr=0.007 ++compensate.rank=16 ++compensate.skip_p=0.0
    ;;
  nc_w8a8)
    export TAPTQ_BITS=8,8 TAPTQ_CONFIG=PTQ4ViT
    exec "$V" mv_recon/taptq.py "${common[@]}" mode=compensate_eval "ptq.bit=[8,8]" ptq.search_mode=ternary ptq.linear_channelwise=false "optim_datasets=[DTU_train_8]" ckpt.path="$C/w8a8_ternary.pt" ++compensate.strategy=module ++compensate.tail_ratio=0.1 ++compensate.tau_thr=0.007 ++compensate.rank=16 ++compensate.skip_p=0.0
    ;;
  cw_w6a6)
    export TAPTQ_BITS=6,6 TAPTQ_CONFIG=PTQ4ViT_channelwise
    exec "$V" mv_recon/taptq.py "${common[@]}" mode=test "ptq.bit=[6,6]" ptq.search_mode=ternary ptq.linear_channelwise=true ckpt.path="$C/channelwise_w6a6_rho001_tau005_r256.pt"
    ;;
  cw_w8a8)
    export TAPTQ_BITS=8,8 TAPTQ_CONFIG=PTQ4ViT_channelwise
    exec "$V" mv_recon/taptq.py "${common[@]}" mode=test "ptq.bit=[8,8]" ptq.search_mode=ternary ptq.linear_channelwise=true ckpt.path="$C/channelwise_w8a8_rho001_tau005_r256.pt"
    ;;
  *)
    echo "Unknown task: $TASK" >&2
    exit 2
    ;;
esac
