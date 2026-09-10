#!/usr/bin/env bash
set -Eeuo pipefail

DATA=/mnt/cephfs/pansicheng/data
DEST=$DATA/eth3d_pi3_final
ARCHIVE=$DATA/eth3d_pi3.tar.zst
SHA_FILE=$ARCHIVE.sha256
SCENES=(courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains)

expected_sha=$(awk 'NR == 1 {print $1}' "$SHA_FILE")
actual_sha=$(sha256sum "$ARCHIVE" | awk '{print $1}')
test "$actual_sha" = "$expected_sha"

if [[ -f "$DEST/.READY" && -f "$DEST/.ARCHIVE_SHA256" ]] && [[ "$(<"$DEST/.ARCHIVE_SHA256")" == "$actual_sha" ]]; then
  echo "ETH3D Pi3 dataset is already ready for archive $actual_sha: $DEST"
  exit 0
fi

if [[ -e "$DEST" ]]; then
  backup="${DEST}.invalid_pre_restore_$(date +%Y%m%d_%H%M%S)"
  mv "$DEST" "$backup"
  echo "Preserved previous ETH3D extraction: $backup"
fi

mkdir -p "$DEST"
tar --zstd --strip-components=1 -xf "$ARCHIVE" -C "$DEST"
for scene in "${SCENES[@]}"; do
  images=$(find "$DEST/$scene/images/custom_undistorted" -type f -name '*.JPG' | wc -l)
  depths=$(find "$DEST/$scene/ground_truth_depth/custom_undistorted" -type f -name '*.JPG' | wc -l)
  cameras=$(find "$DEST/$scene/custom_undistorted_cam" -type f -name '*.npz' | wc -l)
  test "$images" -gt 0 && test "$images" -eq "$depths" && test "$images" -eq "$cameras"
  echo "$scene images=$images depths=$depths cameras=$cameras"
done
printf '%s\n' "$actual_sha" > "$DEST/.ARCHIVE_SHA256"
touch "$DEST/.READY"
echo "ETH3D Pi3 dataset ready: $DEST"
