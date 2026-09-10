#!/usr/bin/env bash
set -euo pipefail
W=/tmp/tmm-multimodel-702478252
R=/mnt/cephfs/pansicheng/tmm-results
export LD_LIBRARY_PATH="$W/libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$W/repo"
cd "$W/repo"
V=/mnt/cephfs/pansicheng/tmm-venv/bin/python
export W R V
run_external() {
  model=$1
  gpu=$2
  repo=/mnt/cephfs/pansicheng/vendor/$model
  weights=/mnt/cephfs/pansicheng/models/$model
  ckpt="$R/checkpoints/${model}_taptq_channelwise_dtu8_w4a8.pt"
  out="$R/eval/${model}_taptq_channelwise_dtu8_w4a8"
  export CUDA_VISIBLE_DEVICES=$gpu
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  "$V" mv_recon/external_taptq.py --model "$model" --weights "$weights" --repo-root "$repo" --mode calib_compensate --checkpoint "$ckpt" --device cuda --bit 4 8 --search-mode ternary --channelwise --metric hessian --quant-config PTQ4ViT_channelwise --data-root /mnt/cephfs/pansicheng/data/dtu_8 --cache-file "/mnt/cephfs/pansicheng/cache/${model}_dtu8_224.npy" --seq-id-map datasets/seq-id-maps/DTUTrain_8_mv-recon_seq-id-map-kf5.json --load-img-size 224 --image-size 224 --rank 16 --tau 0.007 --max-qwt-rows 8192
  "$V" mv_recon/external_taptq.py --model "$model" --weights "$weights" --repo-root "$repo" --mode test --checkpoint "$ckpt" --device cuda --bit 4 8 --search-mode ternary --channelwise --metric hessian --quant-config PTQ4ViT_channelwise --data-root /mnt/cephfs/pansicheng/data/dtu_8 --cache-file "/mnt/cephfs/pansicheng/cache/${model}_dtu8_224.npy" --seq-id-map datasets/seq-id-maps/DTUTrain_8_mv-recon_seq-id-map-kf5.json --load-img-size 518 --image-size 512 --eval-data-root /mnt/cephfs/pansicheng/data/7scenes --eval-seq-id-map datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf40.json --eval-output-dir "$out"
}
nohup bash -c "$(declare -f run_external); run_external dust3r 6" >"$R/logs/dust3r_taptq_channelwise_dtu8_w4a8.log" 2>&1 < /dev/null &
echo DUST3R_PID=$!
nohup bash -c "$(declare -f run_external); run_external mast3r 7" >"$R/logs/mast3r_taptq_channelwise_dtu8_w4a8.log" 2>&1 < /dev/null &
echo MAST3R_PID=$!
