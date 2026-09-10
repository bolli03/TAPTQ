#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?model}
GPU=${2:?gpu}
WAIT_PID=${3:-}

W=/tmp/tmm-multimodel-702478252
R=/mnt/cephfs/pansicheng/tmm-results
V=/mnt/cephfs/pansicheng/tmm-venv/bin/python
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$W/libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$W/repo"
cd "$W/repo"

if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
fi
while nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader | grep -q .; do
  sleep 60
done

CLEAN="$R/checkpoints/${MODEL}_taptq_channelwise_dtu8_w4a8_clean.pt"
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

for RANK in 32 64 128; do
  BASE="$R/checkpoints/${MODEL}_qwt_rho0p1_tau0_fit08_r${RANK}.pt"
  if [[ ! -f "$BASE" ]]; then
    "$V" "${COMMON[@]}" --mode compensate_test --checkpoint "$CLEAN" \
      --save-checkpoint "$BASE" --load-img-size 224 --image-size 224 \
      --rho 0.1 --tau 0 --fit-error-max 0.8 --rank "$RANK" --max-qwt-rows 8192
  fi
  for FIT in 0.4 0.6 0.8; do
    FIT_TAG=${FIT//./p}
    for TAU in 0.005 0.007; do
      TAU_TAG=${TAU//./p}
      DST="$R/checkpoints/${MODEL}_qwt_rho0p1_tau${TAU_TAG}_fit${FIT_TAG}_r${RANK}.pt"
      SRC="$BASE" DST="$DST" TAU="$TAU" FIT="$FIT" "$V" - <<'PY'
import os
import torch
src = os.environ["SRC"]
dst = os.environ["DST"]
tau = float(os.environ["TAU"])
fit = float(os.environ["FIT"])
x = torch.load(src, map_location="cpu", weights_only=False)
records = x.get("qwt_records", [])
keep = {
    item["name"] for item in records
    if item.get("selected")
    and float(item.get("score", -1)) >= tau
    and float(item.get("fit_error", float("inf"))) <= fit
}
x["qwt_meta"] = [item for item in x.get("qwt_meta", []) if item["name"] in keep]
x["qwt_records"] = [dict(item, selected=item["name"] in keep) for item in records]
x["state_dict"] = {
    key: value for key, value in x["state_dict"].items()
    if ".taptq_qwt." not in key or key.split(".taptq_qwt.")[0] in keep
}
torch.save(x, dst)
print(f"rank={x['qwt_meta'][0]['rank'] if x['qwt_meta'] else 'none'} tau={tau} fit={fit}: kept {len(keep)} QwT branches")
PY
      OUT="$R/eval/qwt_rank/${MODEL}_rho0p1_tau${TAU_TAG}_fit${FIT_TAG}_r${RANK}"
      if [[ ! -f "$OUT/_all_samples.json" ]]; then
        if ! "$V" "${COMMON[@]}" --mode test --checkpoint "$DST" \
          --load-img-size 518 --image-size 512 \
          --eval-data-root /mnt/cephfs/pansicheng/data/7scenes \
          --eval-seq-id-map datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf40-qwt-ablation4.json \
          --eval-output-dir "$OUT"; then
          echo "Evaluation failed: model=$MODEL rho=0.1 tau=$TAU fit=$FIT rank=$RANK" >&2
        fi
      fi
    done
  done
done
