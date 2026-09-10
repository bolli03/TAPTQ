#!/usr/bin/env bash
set -uo pipefail

GROUP=${1:?group 0-3}
GPU=${2:?gpu}
WAIT_PID=${3:-}

W=/tmp/tmm-multimodel-702478252
R=/mnt/cephfs/pansicheng/tmm-results
V=/mnt/cephfs/pansicheng/tmm-venv/bin/python
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$W/libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$W/repo"
export TMM_DATA_ROOT=/mnt/cephfs/pansicheng/data
export TMM_CACHE_ROOT=/mnt/cephfs/pansicheng/cache
export VGGT_MODEL_PATH=/mnt/cephfs/pansicheng/models/models--facebook--VGGT-1B
export TAPTQ_MODEL=vggt
export TAPTQ_BITS=4,8
export TAPTQ_CONFIG=PTQ4ViT
cd "$W/repo"

if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
fi
while nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader | grep -q .; do
  sleep 60
done

run_one() {
  local tag=$1 rho=$2 tau=$3 rank=$4
  local out="$R/eval/vggt_channelwise_ablation/$tag"
  if [[ -f "$out/tre_diff/all_metrics.csv" ]]; then
    echo "Skip completed: $tag"
    return 0
  fi
  "$V" mv_recon/taptq.py \
    evaluation=mv_recon model_name=vggt mode=compensate_eval \
    ckpt.path="$R/checkpoints/w4a8_channelwise.pt" \
    ptq.quant_config_name=PTQ4ViT '+ptq.bit=[4,8]' \
    ptq.linear_channelwise=true ptq.search_mode=ternary \
    'eval_datasets=[7scenes-dense]' output_dir="$out" save_suffix="$tag" \
    ++compensate.strategy=module ++compensate.tail_ratio="$rho" \
    ++compensate.tau_thr="$tau" ++compensate.rank="$rank" \
    ++compensate.skip_p=0.0 || echo "Failed: $tag" >&2
}

case "$GROUP" in
  0)
    run_one channelwise_rho01_tau005_r128 0.1 0.005 128
    run_one channelwise_rho05_tau005_r128 0.5 0.005 128
    run_one channelwise_rho1_tau005_r128 1.0 0.005 128
    ;;
  1)
    run_one channelwise_rho001_tau0_r128 0.01 0 128
    run_one channelwise_rho001_tau001_r128 0.01 0.001 128
    run_one channelwise_rho001_tau003_r128 0.01 0.003 128
    ;;
  2)
    run_one channelwise_rho001_tau01_r128 0.01 0.01 128
    run_one channelwise_rho001_tau02_r128 0.01 0.02 128
    run_one channelwise_rho001_tau05_r128 0.01 0.05 128
    ;;
  3)
    run_one channelwise_rho001_tau005_r8 0.01 0.005 8
    run_one channelwise_rho001_tau005_r16 0.01 0.005 16
    run_one channelwise_rho001_tau005_r32 0.01 0.005 32
    ;;
  *)
    echo "Unknown group: $GROUP" >&2
    exit 2
    ;;
esac
