#!/usr/bin/env bash
set -Eeuo pipefail

GPU=${1:?GPU index required}
GROUP=${2:?group 0-6 required}
W=/tmp/tmm-eval-code
R=/mnt/cephfs/pansicheng/tmm-results/eval/7scenes_w4a8_r128_ablation
C=/mnt/cephfs/pansicheng/tmm-results/checkpoints
V=/mnt/cephfs/pansicheng/tmm-venv/bin/python

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$W"
export LD_LIBRARY_PATH="/tmp/tmm-libs:${LD_LIBRARY_PATH:-}"
export TMM_DATA_ROOT=/mnt/cephfs/pansicheng/data
export TMM_CACHE_ROOT=/mnt/cephfs/pansicheng/cache/7scenes_w4a8_r128_ablation
export VGGT_MODEL_PATH=/mnt/cephfs/pansicheng/models/models--facebook--VGGT-1B
export TAPTQ_MODEL=vggt
export TAPTQ_BITS=4,8
export TAPTQ_CONFIG=PTQ4ViT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$R/logs" "$TMM_CACHE_ROOT"
cd "$W"

run_one() {
  local tag=$1 rho=$2 tau=$3
  local out="$R/$tag"
  local log="$R/logs/$tag.log"
  if [[ -f "$out/7scenes-dense/_all_samples.csv" ]]; then
    echo "[$(date -Is)] already complete: $tag"
    return 0
  fi
  mkdir -p "$out"
  echo "[$(date -Is)] start: $tag rho=$rho tau=$tau rank=128"
  "$V" mv_recon/taptq.py \
    evaluation=mv_recon model_name=vggt mode=compensate_eval \
    "eval_datasets=[7scenes-dense]" \
    output_dir="$out" save_suffix="$tag" \
    "ptq.bit=[4,8]" ptq.search_mode=ternary ptq.linear_channelwise=false \
    "optim_datasets=[DTU_train_8]" ckpt.path="$C/w4a8_ternary.pt" \
    ++compensate.strategy=module ++compensate.tail_ratio="$rho" \
    ++compensate.tau_thr="$tau" ++compensate.rank=128 \
    ++compensate.skip_p=0.0 >"$log" 2>&1
  echo "[$(date -Is)] complete: $tag"
}

case "$GROUP" in
  0) run_one rho_0001_r128 0.001 0.007; run_one rho_0005_r128 0.005 0.007 ;;
  1) run_one rho_0010_r128 0.010 0.007; run_one rho_0020_r128 0.020 0.007 ;;
  2) run_one rho_0050_r128 0.050 0.007; run_one rho_0500_r128 0.500 0.007 ;;
  3) run_one rho_1000_r128 1.000 0.007; run_one tau_0001_r128 0.100 0.001 ;;
  4) run_one tau_0003_r128 0.100 0.003; run_one tau_0005_r128 0.100 0.005 ;;
  5) run_one tau_0010_r128 0.100 0.010; run_one tau_0020_r128 0.100 0.020 ;;
  6) run_one tau_0050_r128 0.100 0.050 ;;
  *) echo "Unknown group: $GROUP" >&2; exit 2 ;;
esac
