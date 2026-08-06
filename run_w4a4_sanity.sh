#!/bin/bash
# W4A4 sanity check using provided params (resume_qs)
# Should reproduce the paper's W4A4 AUC@30 ≈ 88.2

set -e
cd /data/minimax-dialogue/users/boli/QuantVGGT/QuantVGGT/evaluation
source /data/minimax-dialogue/users/boli/QuantVGGT/venv_qvggt/bin/activate
export http_proxy=http://pac-internal.xaminim.com:3129
export https_proxy=http://pac-internal.xaminim.com:3129
export PYTHONPATH=/data/minimax-dialogue/users/boli/QuantVGGT/QuantVGGT

# Use specific GPU
export CUDA_VISIBLE_DEVICES=${CUDA:-0}

LOG_DIR=/data/minimax-dialogue/users/boli/QuantVGGT/logs
mkdir -p $LOG_DIR
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/w4a4_sanity_${TS}.log

python run_co3d.py \
    --model_path /data/minimax-dialogue/users/boli/QuantVGGT/QuantVGGT/VGGT-1B/model_tracker_fixed_e20.pt \
    --co3d_dir /data/minimax-dialogue/users/boli/QuantVGGT/co3d_data \
    --co3d_anno_dir /data/minimax-dialogue/users/boli/QuantVGGT/QuantVGGT/co3d_v2_annotations \
    --dtype quarot_w4a4 \
    --seed 0 \
    --lac \
    --lwc \
    --cache_path /data/minimax-dialogue/users/boli/QuantVGGT/QuantVGGT/evaluation/outputs/calib_data.pt \
    --class_mode all \
    --each_nsamples 10 \
    --exp_name a44 \
    --fast_eval \
    --resume_qs 2>&1 | tee $LOG_FILE

echo "LOG: $LOG_FILE"
