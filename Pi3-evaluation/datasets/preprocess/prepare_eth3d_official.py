"""Convert ETH3D official undistorted/COLMAP data to the TMM dataset layout."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image


def _qvec_to_rot(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float32)


def _read_cameras(path: Path) -> dict[int, tuple[np.ndarray, int, int]]:
    cameras = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        camera_id, model = int(fields[0]), fields[1]
        width, height = int(fields[2]), int(fields[3])
        params = np.asarray([float(x) for x in fields[4:]], dtype=np.float32)
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1:3]
        else:
            raise ValueError(f"Unsupported undistorted ETH3D camera model: {model}")
        cameras[camera_id] = (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32), width, height)
    return cameras


def _read_images(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    lines = path.read_text().splitlines()
    images = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 10:
            continue
        qvec = np.asarray([float(x) for x in fields[1:5]], dtype=np.float32)
        tvec = np.asarray([float(x) for x in fields[5:8]], dtype=np.float32)
        camera_id = int(fields[8])
        name = fields[9].split("/")[-1]
        images[name] = (_qvec_to_rot(qvec), tvec, camera_id)
        if i < len(lines):
            i += 1
    return images


def _read_points(path: Path) -> np.ndarray:
    points = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 4:
            points.append([float(fields[1]), float(fields[2]), float(fields[3])])
    if not points:
        raise ValueError(f"No sparse points found in {path}")
    return np.asarray(points, dtype=np.float32)


def _write_sequence(root: Path, sequence: str, output_size: int) -> None:
    seq_root = root / sequence
    calib_root = seq_root / "dslr_calibration_undistorted"
    image_root = seq_root / "images" / "dslr_images_undistorted"
    out_images = seq_root / "images" / "custom_undistorted"
    out_depth = seq_root / "ground_truth_depth" / "custom_undistorted"
    out_cams = seq_root / "custom_undistorted_cam"
    out_images.mkdir(parents=True, exist_ok=True)
    out_depth.mkdir(parents=True, exist_ok=True)
    out_cams.mkdir(parents=True, exist_ok=True)

    cameras = _read_cameras(calib_root / "cameras.txt")
    images = _read_images(calib_root / "images.txt")
    points = _read_points(calib_root / "points3D.txt")

    for name, (rotation, translation, camera_id) in sorted(images.items()):
        source = image_root / name
        if not source.exists():
            continue
        image = Image.open(source).convert("RGB")
        original_width, original_height = image.size
        scale = output_size / original_width
        width = output_size
        height = max(1, round(original_height * scale))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        image.save(out_images / name, quality=95)

        intrinsic, _, _ = cameras[camera_id]
        intrinsic = intrinsic.copy()
        intrinsic[0] *= scale
        intrinsic[1] *= scale
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = translation
        np.savez(out_cams / name.replace(".JPG", ".npz"), intrinsics=intrinsic, extrinsics=extrinsic)

        camera_points = (rotation @ points.T + translation[:, None]).T
        valid = camera_points[:, 2] > 1e-5
        camera_points = camera_points[valid]
        depth = np.zeros((height, width), dtype=np.float32)
        if len(camera_points):
            u = np.rint(intrinsic[0, 0] * camera_points[:, 0] / camera_points[:, 2] + intrinsic[0, 2]).astype(np.int64)
            v = np.rint(intrinsic[1, 1] * camera_points[:, 1] / camera_points[:, 2] + intrinsic[1, 2]).astype(np.int64)
            valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            flat = depth.reshape(-1)
            indices = v[valid] * width + u[valid]
            values = camera_points[valid, 2]
            order = np.argsort(values)
            for index, value in zip(indices[order], values[order]):
                if flat[index] == 0:
                    flat[index] = value
        depth.tofile(out_depth / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-size", type=int, default=518)
    parser.add_argument("--sequences", nargs="*", default=None)
    args = parser.parse_args()
    root = Path(args.data_root)
    sequences = args.sequences or sorted(p.name for p in root.iterdir() if (p / "dslr_calibration_undistorted").is_dir())
    for sequence in sequences:
        print(f"Preparing ETH3D {sequence}", flush=True)
        _write_sequence(root, sequence, args.output_size)


if __name__ == "__main__":
    main()
