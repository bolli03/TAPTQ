#!/usr/bin/env bash
set -euo pipefail

GPU=${1:?GPU index required}
TASK=${2:?task name required}
W=${TMM_WORKDIR:-/tmp/tmm-eval-code}
R=${TMM_RESULT_ROOT:-/mnt/cephfs/pansicheng/tmm-results/eval/7scenes_kf200_verified}
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

common=(evaluation=mv_recon model_name=vggt "eval_datasets=[7scenes-sparse]" output_dir="$R/$TASK" save_suffix="$TASK")

case "$TASK" in
  fp)
    export TAPTQ_BITS=4,8 TAPTQ_CONFIG=PTQ4ViT
    exec "$V" mv_recon/taptq.py "${common[@]}" mode=fp_eval "ptq.bit=[4,8]" ptq.search_mode=ternary ptq.linear_channelwise=false
    ;;
  ptq4vit_w4a8|ptq4vit_w6a6|ptq4vit_w8a8)
    bits=${TASK#ptq4vit_w}; wb=${bits%a*}; ab=${bits#*a}
    export TAPTQ_BITS="$wb,$ab" TAPTQ_CONFIG=PTQ4ViT
    exec "$V" mv_recon/taptq.py "${common[@]}" mode=test "ptq.bit=[$wb,$ab]" ptq.search_mode=ternary ptq.linear_channelwise=false ckpt.path="$C/ptq4vit_dtu8_w${wb}a${ab}.pt"
    ;;
  taptq_w4a8|taptq_w6a6|taptq_w8a8)
    bits=${TASK#taptq_w}; wb=${bits%a*}; ab=${bits#*a}; rank=16
    [[ "$wb" == 4 && "$ab" == 8 ]] && rank=128
    export TAPTQ_BITS="$wb,$ab" TAPTQ_CONFIG=PTQ4ViT
    exec "$V" mv_recon/taptq.py "${common[@]}" mode=compensate_eval "ptq.bit=[$wb,$ab]" ptq.search_mode=ternary ptq.linear_channelwise=false "optim_datasets=[DTU_train_8]" ckpt.path="$C/w${wb}a${ab}_ternary.pt" ++compensate.strategy=module ++compensate.tail_ratio=0.1 ++compensate.tau_thr=0.007 ++compensate.rank="$rank" ++compensate.skip_p=0.0
    ;;
  rtn_w4a8) method=rtn; bits=4,8; ckpt=official_rtn_w4a8.pt ;;
  rtn_w6a6) method=rtn; bits=6,6; ckpt=official_rtn_dtu8_w6a6.pt ;;
  rtn_w8a8) method=rtn; bits=8,8; ckpt=paper_vggt_rtn_w8a8.pt ;;
  repq_w4a8) method=repq; bits=4,8; ckpt=official_repq_w4a8.pt ;;
  repq_w6a6) method=repq; bits=6,6; ckpt=official_repq_dtu8_w6a6.pt ;;
  repq_w8a8) method=repq; bits=8,8; ckpt=paper_vggt_repq_w8a8.pt ;;
  erq_w4a8) method=erq; bits=4,8; ckpt=official_zysxmu_erq_twopart_w4a8.pt ;;
  erq_w6a6) method=erq; bits=6,6; ckpt=official_zysxmu_erq_twopart_dtu8_w6a6.pt ;;
  erq_w8a8) method=erq; bits=8,8; ckpt=paper_vggt_erq_twopart_w8a8.pt ;;
  gptq_w4a8) method=gptq; bits=4,8; ckpt=official_gptq_dtu8_w4a8.pt ;;
  gptq_w6a6) method=gptq; bits=6,6; ckpt=official_gptq_dtu8_w6a6.pt ;;
  gptq_w8a8) method=gptq; bits=8,8; ckpt=paper_vggt_gptq_w8a8.pt ;;
  smoothquant_w4a8) method=smoothquant; bits=4,8; ckpt=official_smoothquant_dtu8_w4a8_a05.pt ;;
  smoothquant_w6a6) method=smoothquant; bits=6,6; ckpt=official_smoothquant_dtu8_w6a6_a05.pt ;;
  smoothquant_w8a8) method=smoothquant; bits=8,8; ckpt=paper_vggt_smoothquant_a03_w8a8.pt ;;
  *) echo "Unknown task: $TASK" >&2; exit 2 ;;
esac

wb=${bits%,*}; ab=${bits#*,}
exec "$V" mv_recon/baseline_quant.py "${common[@]}" mode=test baseline.method="$method" "baseline.bit=[$wb,$ab]" ckpt.path="$C/$ckpt"
