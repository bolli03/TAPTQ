#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   bash run_eth3d_kf5_8gpu_campaign.sh tune
#   bash run_eth3d_kf5_8gpu_campaign.sh summarize-tune
#   bash run_eth3d_kf5_8gpu_campaign.sh baselines
#   bash run_eth3d_kf5_8gpu_campaign.sh summarize-baselines
#
# `baselines` is deliberately blocked until `summarize-tune` has produced a
# complete tuning manifest. It then launches the unified VGGT baselines on all
# eight GPUs and records each completed result independently.

ACTION=${1:?action: tune | summarize-tune | baselines | summarize-baselines}
W=/tmp/tmm-eval-code
R=/mnt/cephfs/pansicheng/tmm-results/eval/eth3d_kf5_w4a8_campaign
C=/mnt/cephfs/pansicheng/tmm-results/checkpoints
V=/mnt/cephfs/pansicheng/tmm-venv/bin/python
ETH3D=/mnt/cephfs/pansicheng/data/eth3d_pi3_final
CACHE=/mnt/cephfs/pansicheng/cache/eth3d_kf5_campaign
STATUS=$R/status

SCENES=(courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains)

require_ready() {
  test -f "$ETH3D/.READY"
  test -f "$ETH3D/.ARCHIVE_SHA256"
  test -s "$ETH3D/.ARCHIVE_SHA256"
}

configure_runtime() {
  local gpu=$1 cw=$2
  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTHONPATH="$W"
  export LD_LIBRARY_PATH="/tmp/tmm-libs/usr/lib/x86_64-linux-gnu:/tmp/tmm-libs:${LD_LIBRARY_PATH:-}"
  export TMM_DATA_ROOT=/mnt/cephfs/pansicheng/data
  export TMM_CACHE_ROOT="$CACHE"
  export VGGT_MODEL_PATH=/mnt/cephfs/pansicheng/models/models--facebook--VGGT-1B
  export TAPTQ_MODEL=vggt
  export TAPTQ_BITS=4,8
  export TAPTQ_CONFIG=PTQ4ViT
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  [[ "$cw" == true ]] && export TAPTQ_CONFIG=PTQ4ViT_channelwise
}

mark_status() {
  local tag=$1 phase=$2 gpu=$3 state=$4 detail=$5
  TAG="$tag" PHASE="$phase" GPU="$gpu" STATE="$state" DETAIL="$detail" STATUS="$STATUS" "$V" - <<'PY'
import json, os, tempfile
from pathlib import Path
out = Path(os.environ['STATUS']) / f"{os.environ['TAG']}.json"
out.parent.mkdir(parents=True, exist_ok=True)
payload = {
    'tag': os.environ['TAG'], 'phase': os.environ['PHASE'],
    'gpu': int(os.environ['GPU']), 'state': os.environ['STATE'],
    'detail': os.environ['DETAIL'],
}
with tempfile.NamedTemporaryFile('w', delete=False, dir=out.parent, suffix='.tmp') as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write('\n')
    temp = f.name
os.replace(temp, out)
PY
}

run_taptq() {
  local gpu=$1 tag=$2 cw=$3 rho=$4 tau=$5 rank=$6
  local out="$R/tune/$tag" log="$R/logs/$tag.log"
  local ckpt="$C/w4a8_ternary.pt"
  [[ "$cw" == true ]] && ckpt="$C/w4a8_channelwise.pt"
  [[ -f "$out/ETH3D/_all_samples.csv" ]] && return 0

  configure_runtime "$gpu" "$cw"
  mkdir -p "$out" "$R/logs" "$STATUS" "$CACHE"
  mark_status "$tag" tune "$gpu" running "rho=$rho tau=$tau rank=$rank channelwise=$cw"
  if (
    cd "$W"
    "$V" mv_recon/taptq.py \
      evaluation=mv_recon model_name=vggt mode=compensate_eval \
      "eval_datasets=[ETH3D]" \
      "data.ETH3D.cfg.ETH3D_DIR=$ETH3D" \
      "data.ETH3D.cfg.cache_file=$CACHE/eth3d_mv_recon_cache.npy" \
      output_dir="$out" save_suffix="$tag" \
      "ptq.bit=[4,8]" ptq.search_mode=ternary "ptq.linear_channelwise=$cw" \
      "optim_datasets=[DTU_train_8]" ckpt.path="$ckpt" \
      ++compensate.strategy=module ++compensate.tail_ratio="$rho" \
      ++compensate.tau_thr="$tau" ++compensate.rank="$rank" \
      ++compensate.skip_p=0.0
  ) >"$log" 2>&1; then
    test -f "$out/ETH3D/_all_samples.csv"
    mark_status "$tag" tune "$gpu" completed "$out/ETH3D/_all_samples.csv"
  else
    mark_status "$tag" tune "$gpu" failed "$log"
    return 1
  fi
}

# Format: tag|channelwise|rho|tau|rank.  All candidates are W4A8 and vary only
# compensation knobs or channel-wise quantization from their correct quant-only source.
TUNE_TASKS=(
  nc_rho005_tau001_r128\|false\|0.05\|0.001\|128
  nc_rho005_tau003_r128\|false\|0.05\|0.003\|128
  nc_rho005_tau005_r128\|false\|0.05\|0.005\|128
  nc_rho005_tau007_r128\|false\|0.05\|0.007\|128
  nc_rho005_tau010_r128\|false\|0.05\|0.010\|128
  nc_rho010_tau001_r128\|false\|0.10\|0.001\|128
  nc_rho010_tau003_r128\|false\|0.10\|0.003\|128
  nc_rho010_tau005_r128\|false\|0.10\|0.005\|128
  nc_rho010_tau010_r128\|false\|0.10\|0.010\|128
  nc_rho010_tau003_r256\|false\|0.10\|0.003\|256
  nc_rho010_tau005_r256\|false\|0.10\|0.005\|256
  nc_rho010_tau007_r256\|false\|0.10\|0.007\|256
  nc_rho010_tau005_r064\|false\|0.10\|0.005\|64
  cw_rho001_tau005_r128\|true\|0.001\|0.005\|128
  cw_rho005_tau005_r128\|true\|0.005\|0.005\|128
  cw_rho010_tau001_r128\|true\|0.010\|0.001\|128
  cw_rho010_tau003_r128\|true\|0.010\|0.003\|128
  cw_rho010_tau005_r128\|true\|0.010\|0.005\|128
  cw_rho010_tau007_r128\|true\|0.010\|0.007\|128
  cw_rho010_tau005_r064\|true\|0.010\|0.005\|64
  cw_rho010_tau005_r256\|true\|0.010\|0.005\|256
  cw_rho050_tau005_r128\|true\|0.050\|0.005\|128
  cw_rho100_tau005_r128\|true\|0.100\|0.005\|128
  cw_rho010_tau010_r128\|true\|0.010\|0.010\|128
)

worker_tune() {
  local gpu=$1
  local i spec tag cw rho tau rank
  for i in "${!TUNE_TASKS[@]}"; do
    (( i % 8 == gpu )) || continue
    IFS='|' read -r tag cw rho tau rank <<<"${TUNE_TASKS[$i]}"
    run_taptq "$gpu" "$tag" "$cw" "$rho" "$tau" "$rank" || true
  done
}

summarize_tune() {
  ROOT="$R/tune" OUT="$R/tune_summary.csv" BEST="$R/best_taptq.json" EXPECTED="${#TUNE_TASKS[@]}" "$V" - <<'PY'
import csv, json, os
from pathlib import Path
root, expected = Path(os.environ['ROOT']), int(os.environ['EXPECTED'])
rows = []
for p in sorted(root.glob('*/ETH3D/_all_samples.csv')):
    with p.open() as f:
        samples = list(csv.DictReader(f))
    if len(samples) != 13:
        continue
    mean = lambda k: sum(float(x[k]) for x in samples) / len(samples)
    tag = p.parents[1].name
    rows.append({
        'tag': tag, 'artifact': str(p), 'scenes': len(samples),
        'Acc-mean': mean('Acc-mean'), 'Acc-med': mean('Acc-med'),
        'Comp-mean': mean('Comp-mean'), 'Comp-med': mean('Comp-med'),
        'NC-mean': (mean('NC1-mean') + mean('NC2-mean')) / 2,
        'NC-med': (mean('NC1-med') + mean('NC2-med')) / 2,
    })
if len(rows) != expected:
    raise SystemExit(f'Incomplete tune matrix: {len(rows)}/{expected} valid 13-scene results')
rows.sort(key=lambda x: (x['Acc-mean'], x['Comp-mean'], -x['NC-mean']))
fields = list(rows[0])
with Path(os.environ['OUT']).open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
with Path(os.environ['BEST']).open('w') as f:
    json.dump(rows[0], f, indent=2); f.write('\n')
print(json.dumps(rows[0], indent=2))
PY
}

# Format: tag|method|bits|checkpoint.  Baselines are only launched after a full
# 24-case tuning matrix is summarized, and every output is isolated by tag.
BASELINE_TASKS=(
  ptq4vit_w4a8\|ptq4vit\|4,8\|ptq4vit_dtu8_w4a8.pt
  rtn_w4a8\|rtn\|4,8\|official_rtn_w4a8.pt
  repq_w4a8\|repq\|4,8\|official_repq_w4a8.pt
  erq_w4a8\|erq\|4,8\|official_zysxmu_erq_twopart_w4a8.pt
  gptq_w4a8\|gptq\|4,8\|official_gptq_dtu8_w4a8.pt
  smoothquant_w4a8\|smoothquant\|4,8\|official_smoothquant_dtu8_w4a8_a05.pt
  ptq4vit_w6a6\|ptq4vit\|6,6\|ptq4vit_dtu8_w6a6.pt
  rtn_w6a6\|rtn\|6,6\|official_rtn_dtu8_w6a6.pt
  repq_w6a6\|repq\|6,6\|official_repq_dtu8_w6a6.pt
  erq_w6a6\|erq\|6,6\|official_zysxmu_erq_twopart_dtu8_w6a6.pt
  gptq_w6a6\|gptq\|6,6\|official_gptq_dtu8_w6a6.pt
  smoothquant_w6a6\|smoothquant\|6,6\|official_smoothquant_dtu8_w6a6_a05.pt
  ptq4vit_w8a8\|ptq4vit\|8,8\|ptq4vit_dtu8_w8a8.pt
  rtn_w8a8\|rtn\|8,8\|paper_vggt_rtn_w8a8.pt
  repq_w8a8\|repq\|8,8\|paper_vggt_repq_w8a8.pt
  erq_w8a8\|erq\|8,8\|paper_vggt_erq_twopart_w8a8.pt
  gptq_w8a8\|gptq\|8,8\|paper_vggt_gptq_w8a8.pt
  smoothquant_w8a8\|smoothquant\|8,8\|paper_vggt_smoothquant_a03_w8a8.pt
)

run_baseline() {
  local gpu=$1 tag=$2 method=$3 bits=$4 ckpt=$5
  local out="$R/baselines/$tag" log="$R/logs/$tag.log"
  local wb=${bits%,*} ab=${bits#*,}
  [[ -f "$out/ETH3D/_all_samples.csv" ]] && return 0
  test -f "$C/$ckpt"
  configure_runtime "$gpu" false
  mkdir -p "$out" "$R/logs" "$STATUS" "$CACHE"
  mark_status "$tag" baseline "$gpu" running "method=$method bits=$bits checkpoint=$ckpt"
  if (
    cd "$W"
    if [[ "$method" == ptq4vit ]]; then
      "$V" mv_recon/taptq.py \
        evaluation=mv_recon model_name=vggt mode=test \
        "eval_datasets=[ETH3D]" \
        "data.ETH3D.cfg.ETH3D_DIR=$ETH3D" \
        "data.ETH3D.cfg.cache_file=$CACHE/eth3d_mv_recon_cache.npy" \
        output_dir="$out" save_suffix="$tag" \
        "ptq.bit=[$wb,$ab]" ptq.search_mode=ternary ptq.linear_channelwise=false \
        ckpt.path="$C/$ckpt"
    else
      "$V" mv_recon/baseline_quant.py \
        evaluation=mv_recon model_name=vggt mode=test \
        "eval_datasets=[ETH3D]" \
        "data.ETH3D.cfg.ETH3D_DIR=$ETH3D" \
        "data.ETH3D.cfg.cache_file=$CACHE/eth3d_mv_recon_cache.npy" \
        output_dir="$out" save_suffix="$tag" \
        baseline.method="$method" "baseline.bit=[$wb,$ab]" ckpt.path="$C/$ckpt"
    fi
  ) >"$log" 2>&1; then
    test -f "$out/ETH3D/_all_samples.csv"
    mark_status "$tag" baseline "$gpu" completed "$out/ETH3D/_all_samples.csv"
  else
    mark_status "$tag" baseline "$gpu" failed "$log"
    return 1
  fi
}

worker_baselines() {
  local gpu=$1
  local i spec tag method bits ckpt
  for i in "${!BASELINE_TASKS[@]}"; do
    (( i % 8 == gpu )) || continue
    IFS='|' read -r tag method bits ckpt <<<"${BASELINE_TASKS[$i]}"
    run_baseline "$gpu" "$tag" "$method" "$bits" "$ckpt" || true
  done
}

summarize_baselines() {
  ROOT="$R/baselines" OUT="$R/baseline_summary.csv" EXPECTED="${#BASELINE_TASKS[@]}" "$V" - <<'PY'
import csv, os
from pathlib import Path
root, expected = Path(os.environ['ROOT']), int(os.environ['EXPECTED'])
rows=[]
for p in sorted(root.glob('*/ETH3D/_all_samples.csv')):
    with p.open() as f: samples=list(csv.DictReader(f))
    if len(samples)!=13: continue
    mean=lambda k:sum(float(x[k]) for x in samples)/len(samples)
    rows.append({'tag':p.parents[1].name,'artifact':str(p),'scenes':len(samples),
                 'Acc-mean':mean('Acc-mean'),'Acc-med':mean('Acc-med'),
                 'Comp-mean':mean('Comp-mean'),'Comp-med':mean('Comp-med'),
                 'NC-mean':(mean('NC1-mean')+mean('NC2-mean'))/2,
                 'NC-med':(mean('NC1-med')+mean('NC2-med'))/2})
if len(rows)!=expected:
    raise SystemExit(f'Incomplete baseline matrix: {len(rows)}/{expected} valid 13-scene results')
rows.sort(key=lambda x:x['tag'])
with Path(os.environ['OUT']).open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f'wrote {os.environ["OUT"]}')
PY
}

launch_workers() {
  local phase=$1
  local fn=worker_tune
  [[ "$phase" == baseline ]] && fn=worker_baselines
  mkdir -p "$R/logs" "$STATUS" "$CACHE"
  for gpu in {0..7}; do
    nohup bash "$0" "${phase}-worker" "$gpu" >"$R/logs/${phase}_gpu${gpu}.runner.log" 2>&1 &
  done
  echo "Launched $phase workers on GPU0-GPU7."
}

case "$ACTION" in
  tune)
    require_ready
    launch_workers tune
    ;;
  tune-worker)
    require_ready
    worker_tune "${2:?GPU required}"
    ;;
  summarize-tune)
    summarize_tune
    ;;
  baselines)
    require_ready
    test -f "$R/best_taptq.json" || { echo 'Run summarize-tune successfully before baselines.' >&2; exit 2; }
    launch_workers baseline
    ;;
  baseline-worker)
    require_ready
    worker_baselines "${2:?GPU required}"
    ;;
  summarize-baselines)
    summarize_baselines
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
