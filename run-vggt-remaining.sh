#!/usr/bin/env bash
set -euo pipefail
W=/tmp/tmm-multimodel-702478252
R=/mnt/cephfs/pansicheng/tmm-results
export LD_LIBRARY_PATH="$W/libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$W/repo"
export TMM_DATA_ROOT=/mnt/cephfs/pansicheng/data
export TMM_CACHE_ROOT=/mnt/cephfs/pansicheng/cache
export VGGT_MODEL_PATH=/mnt/cephfs/pansicheng/models/models--facebook--VGGT-1B
cd "$W/repo"
V=/mnt/cephfs/pansicheng/tmm-venv/bin/python
cat > /tmp/run_ptq4vit_gpu2.sh <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2 TAPTQ_MODEL=vggt TAPTQ_BITS=4,8 TAPTQ_CONFIG=PTQ4ViT
V=/mnt/cephfs/pansicheng/tmm-venv/bin/python
R=/mnt/cephfs/pansicheng/tmm-results
cd /tmp/tmm-multimodel-702478252/repo
"$V" mv_recon/taptq.py evaluation=mv_recon model_name=vggt mode=calib "ptq.bit=[4,8]" ptq.search_mode=ternary ptq.linear_channelwise=false "optim_datasets=[DTU_train_8]" "test_datasets=[DTU_train_8]" ckpt.path="$R/checkpoints/ptq4vit_dtu8_w4a8.pt"
"$V" mv_recon/taptq.py evaluation=mv_recon model_name=vggt mode=test "ptq.bit=[4,8]" ptq.search_mode=ternary ptq.linear_channelwise=false ckpt.path="$R/checkpoints/ptq4vit_dtu8_w4a8.pt" "eval_datasets=[7scenes-dense,ETH3D]" output_dir="$R/eval/ptq4vit_dtu8_w4a8" save_suffix=ptq4vit_dtu8_w4a8
INNER
chmod +x /tmp/run_ptq4vit_gpu2.sh
nohup /tmp/run_ptq4vit_gpu2.sh >"$R/logs/ptq4vit_dtu8_w4a8.log" 2>&1 < /dev/null &
echo PTQ4VIT_PID=$!
nohup env CUDA_VISIBLE_DEVICES=3 TAPTQ_MODEL=vggt TAPTQ_BITS=4,8 TAPTQ_CONFIG=PTQ4ViT_channelwise "$V" mv_recon/taptq.py evaluation=mv_recon model_name=vggt mode=e2e "ptq.bit=[4,8]" ptq.search_mode=ternary ptq.linear_channelwise=true "optim_datasets=[DTU_train_8]" "test_datasets=[DTU_train_8]" ckpt.path="$R/checkpoints/vggt_taptq_channelwise_dtu8_w4a8_refresh.pt" "eval_datasets=[7scenes-dense,ETH3D]" output_dir="$R/eval/vggt_taptq_channelwise_dtu8_w4a8_refresh" save_suffix=vggt_taptq_channelwise_dtu8_w4a8_refresh >"$R/logs/vggt_taptq_channelwise_dtu8_w4a8_refresh.log" 2>&1 < /dev/null &
echo VGGT_CW_TAPTQ_PID=$!
