#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/workspace/TAPTQ/data
PREP=$ROOT/eth3d_pi3_work
RUNROOT=$ROOT/eth3d_pi3_runroot
RAW=$ROOT/eth3d_pi3_raw
LOG=$ROOT/eth3d_pi3_pipeline.log
PYTHON=/data/workspace/TAPTQ/miniconda3/bin/python
LIBROOT=$ROOT/eth3d_pi3_libs
ARCHIVE=$ROOT/eth3d_pi3.tar.zst
ARCHIVE_NEW=$ARCHIVE.new
CEPH_DIR=/mnt/cephfs970/pansicheng/data
SCENES=(courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains)
AFFECTED=(terrace)
MAP=/data/workspace/TAPTQ/Pi3-evaluation/datasets/seq-id-maps/ETH3D_mv-recon_seq-id-map-kf5.json

export LD_LIBRARY_PATH="$LIBROOT/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
log(){ echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }

mkdir -p "$RUNROOT/data"
ln -sfn "$PREP" "$RUNROOT/data/eth3d"
rm -f "$ROOT/ETH3D_PI3_PIPELINE_COMPLETED"

log "rebuild start: restore complete official depths without frame skipping"
for scene in "${AFFECTED[@]}"; do
  package="$RAW/${scene}_dslr_depth.7z"
  test -f "$package"
  python3 -m py7zr t "$package" >/dev/null
  rm -rf "$PREP/$scene/ground_truth_depth/dslr_images"
  python3 -m py7zr x "$package" "$PREP" >/dev/null
  test -d "$PREP/$scene/ground_truth_depth/dslr_images"
  test -z "$(find "$PREP/$scene/ground_truth_depth/dslr_images" -type f -size 0 -print -quit)"
  rm -rf "$PREP/$scene/images/custom_undistorted" \
         "$PREP/$scene/ground_truth_depth/custom_undistorted" \
         "$PREP/$scene/custom_undistorted_cam"
  log "restored official depth and cleared derived outputs: $scene"
done

cd "$RUNROOT"
for scene in "${AFFECTED[@]}"; do
  ETH3D_SEQUENCES="$scene" "$PYTHON" /data/workspace/TAPTQ/Pi3-evaluation/datasets/preprocess/prepare_eth3d.py >"$LOG.$scene.rebuild" 2>&1
  log "preprocessed complete scene: $scene"
done

PREP="$PREP" MAP="$MAP" python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ['PREP'])
with open(os.environ['MAP']) as f:
    seq_map = json.load(f)
for scene, ids in seq_map.items():
    images = sorted((root / scene / 'images' / 'custom_undistorted').glob('*.JPG'))
    depths = sorted((root / scene / 'ground_truth_depth' / 'custom_undistorted').glob('*.JPG'))
    cameras = sorted((root / scene / 'custom_undistorted_cam').glob('*.npz'))
    if not images or len(images) != len(depths) or len(images) != len(cameras):
        raise RuntimeError(f'{scene}: inconsistent image/depth/camera counts: {len(images)}/{len(depths)}/{len(cameras)}')
    if max(ids) >= len(images):
        raise RuntimeError(f'{scene}: kf=5 requires frame {max(ids)}, but only {len(images)} frames exist')
    for frame_id in ids:
        name = images[frame_id].name
        if not (root / scene / 'ground_truth_depth' / 'custom_undistorted' / name).is_file():
            raise RuntimeError(f'{scene}: missing mapped depth for {name}')
        if not (root / scene / 'custom_undistorted_cam' / name.replace('.JPG', '.npz')).is_file():
            raise RuntimeError(f'{scene}: missing mapped camera for {name}')
    print(f'{scene}: frames={len(images)}, kf5_max={max(ids)}')
PY

PACK_PATHS=()
for scene in "${SCENES[@]}"; do
  ni=$(find "$PREP/$scene/images/custom_undistorted" -type f -name '*.JPG' | wc -l)
  nd=$(find "$PREP/$scene/ground_truth_depth/custom_undistorted" -type f -name '*.JPG' | wc -l)
  nc=$(find "$PREP/$scene/custom_undistorted_cam" -type f -name '*.npz' | wc -l)
  test "$ni" -gt 0 && test "$ni" -eq "$nd" && test "$ni" -eq "$nc"
  log "$scene images=$ni depths=$nd cameras=$nc"
  PACK_PATHS+=("$scene/images/custom_undistorted" "$scene/ground_truth_depth/custom_undistorted" "$scene/custom_undistorted_cam")
done

rm -f "$ARCHIVE_NEW" "$ARCHIVE_NEW.sha256"
tar -C "$PREP" --zstd --transform='s,^,eth3d_pi3/,' -cf "$ARCHIVE_NEW" "${PACK_PATHS[@]}"
zstd -t "$ARCHIVE_NEW"
sha256sum "$ARCHIVE_NEW" > "$ARCHIVE_NEW.sha256"
mv -f "$ARCHIVE_NEW" "$ARCHIVE"
mv -f "$ARCHIVE_NEW.sha256" "$ARCHIVE.sha256"

remote_archive="$CEPH_DIR/$(basename "$ARCHIVE")"
remote_new="$remote_archive.new"
rsync -ah --partial --append-verify --info=progress2 "$ARCHIVE" "$remote_new" >>"$LOG" 2>&1
rsync -ah --partial --info=progress2 "$ARCHIVE.sha256" "$remote_new.sha256" >>"$LOG" 2>&1
local_sha=$(sha256sum "$ARCHIVE" | cut -d ' ' -f1)
remote_sha=$(sha256sum "$remote_new" | cut -d ' ' -f1)
test "$local_sha" = "$remote_sha"
mv -f "$remote_new" "$remote_archive"
mv -f "$remote_new.sha256" "$remote_archive.sha256"
touch "$ROOT/ETH3D_PI3_PIPELINE_COMPLETED"
log "rebuild complete sha256=$local_sha"
