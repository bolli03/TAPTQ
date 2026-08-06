#!/bin/bash
# NRGBD point cloud reconstruction evaluation
# Same usage pattern as run_7s_quant.sh

set -e
cd /data/minimax-dialogue/users/boli/QuantVGGT/QuantVGGT/evaluation
source /data/minimax-dialogue/users/boli/QuantVGGT/venv_qvggt/bin/activate
export http_proxy=http://pac-internal.xaminim.com:3129
export https_proxy=http://pac-internal.xaminim.com:3129
export PYTHONPATH=/data/minimax-dialogue/users/boli/QuantVGGT/QuantVGGT
export CUDA_VISIBLE_DEVICES=${CUDA:-0}

DTYPE=${DTYPE:-quarot_w4a4}
EXP=${EXP:-a44}
KF=${KF:-100}
RESUME=${RESUME:-1}
SCENE_ROOT=${SCENE_ROOT:-/data/minimax-dialogue/users/boli/QuantVGGT/datasets/nrgbd}

LOG_DIR=/data/minimax-dialogue/users/boli/QuantVGGT/logs
mkdir -p $LOG_DIR
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=$LOG_DIR/nr_${EXP}_kf${KF}_${TS}.log

RESUME_FLAG=""
[ "$RESUME" = "1" ] && RESUME_FLAG="--resume_qs"

python run_7andN.py \
    --model_path /data/minimax-dialogue/users/boli/QuantVGGT/QuantVGGT/VGGT-1B/model_tracker_fixed_e20.pt \
    --co3d_dir /data/minimax-dialogue/users/boli/QuantVGGT/co3d_data \
    --co3d_anno_dir /data/minimax-dialogue/users/boli/QuantVGGT/QuantVGGT/co3d_v2_annotations \
    --class_mode all \
    --each_nsamples 10 \
    --dtype $DTYPE \
    --lwc --lac \
    --cache_path /data/minimax-dialogue/users/boli/QuantVGGT/QuantVGGT/evaluation/outputs/calib_data.pt \
    --output_dir "./eval_results_nr_${EXP}" \
    --kf $KF \
    --dataset nr \
    --dataset_path $SCENE_ROOT \
    --exp_name $EXP \
    $RESUME_FLAG 2>&1 | tee $LOG_FILE

echo "LOG: $LOG_FILE"
