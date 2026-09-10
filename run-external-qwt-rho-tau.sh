#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?model}
GPU=${2:?gpu}
RHO=${3:?rho}

W=/tmp/tmm-multimodel-702478252
R=/mnt/cephfs/pansicheng/tmm-results
V=/mnt/cephfs/pansicheng/tmm-venv/bin/python
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$W/libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$W/repo"
cd "$W/repo"

while nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader | grep -q .; do
  sleep 60
done

SRC="$R/checkpoints/${MODEL}_taptq_channelwise_dtu8_w4a8_clean.pt"
RHO_TAG=${RHO//./p}
BASE="$R/checkpoints/${MODEL}_qwt_rho${RHO_TAG}_tau0_fit08.pt"
COMMON=(
  mv_recon/external_taptq.py
  --model "$MODEL"
  --weights "/mnt/cephfs/pansicheng/models/$MODEL"
  --repo-root "/mnt/cephfs/pansicheng/vendor/$MODEL"
  --device cuda --bit 4 8 --search-mode ternary --channelwise
  --metric hessian --quant-config PTQ4ViT_channelwise
  --data-root /mnt/cephfs/pansicheng/data/dtu_8
  --cache-file "/mnt/cephfs/pansicheng/cache/${MODEL}_dtu8_224.npy"
  --seq-id-map datasets/seq-id-maps/DTUTrain_8_mv-recon_seq-id-map-kf5.json
)

"$V" "${COMMON[@]}" --mode compensate_test --checkpoint "$SRC" \
  --save-checkpoint "$BASE" --load-img-size 224 --image-size 224 \
  --rho "$RHO" --tau 0 --fit-error-max 0.8 --rank 16 --max-qwt-rows 8192

for TAU in 0 0.003 0.007 0.02; do
  TAU_TAG=${TAU//./p}
  DST="$R/checkpoints/${MODEL}_qwt_rho${RHO_TAG}_tau${TAU_TAG}_fit08.pt"
  SRC="$BASE" DST="$DST" TAU="$TAU" "$V" - <<'PY'
import os
import torch
src, dst, tau = os.environ["SRC"], os.environ["DST"], float(os.environ["TAU"])
x = torch.load(src, map_location="cpu", weights_only=False)
records = x.get("qwt_records", [])
keep = {
    item["name"] for item in records
    if item.get("selected") and float(item.get("score", -1)) >= tau
}
x["qwt_meta"] = [item for item in x.get("qwt_meta", []) if item["name"] in keep]
x["qwt_records"] = [dict(item, selected=item["name"] in keep) for item in records]
x["state_dict"] = {
    key: value for key, value in x["state_dict"].items()
    if ".taptq_qwt." not in key or key.split(".taptq_qwt.")[0] in keep
}
torch.save(x, dst)
print(f"tau={tau}: kept {len(keep)} QwT branches -> {dst}")
PY
  OUT="$R/eval/qwt_rho_tau/${MODEL}_rho${RHO_TAG}_tau${TAU_TAG}_fit08"
  if ! "$V" "${COMMON[@]}" --mode test --checkpoint "$DST" \
    --load-img-size 518 --image-size 512 \
    --eval-data-root /mnt/cephfs/pansicheng/data/7scenes \
    --eval-seq-id-map datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf40-qwt-ablation4.json \
    --eval-output-dir "$OUT"; then
    echo "Evaluation failed: model=$MODEL rho=$RHO tau=$TAU" >&2
  fi
done
