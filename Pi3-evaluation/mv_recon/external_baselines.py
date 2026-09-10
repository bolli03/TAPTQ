"""Unified E1 adapter for DUSt3R and MASt3R point-map baselines.

The adapter keeps the dataset, alignment, ICP, and Acc/Comp/NC protocol used by
TAPTQ while leaving the external model's native pair/global-alignment inference
inside this module. It deliberately does not quantize the external models.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mv_recon.utils import accuracy, completion, umeyama

LOGGER = logging.getLogger("external_baselines")


def load_external_model(model_name: str, weights: str, device: str, repo_root: str | None = None):
    """Load a local DUSt3R/MASt3R Hugging Face directory or checkpoint."""
    name = model_name.lower()
    if repo_root:
        sys.path.insert(0, str(Path(repo_root).resolve()))
    if name == "dust3r":
        from dust3r.model import AsymmetricCroCo3DStereo
        model = AsymmetricCroCo3DStereo.from_pretrained(weights)
    elif name == "mast3r":
        from mast3r.model import AsymmetricMASt3R
        model = AsymmetricMASt3R.from_pretrained(weights)
    else:
        raise ValueError(f"Unsupported external model: {model_name}")
    return model.to(device).eval()


def predict_pointmaps(
    filelist: list[str],
    model: torch.nn.Module,
    device: str,
    image_size: int = 512,
    niter: int = 300,
    scene_graph: str = "complete",
) -> list[np.ndarray]:
    """Run native pair inference and global alignment, returning one map/view."""
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.device import to_numpy
    from dust3r.utils.image import load_images

    patch_size = getattr(model, "patch_size", 16)
    square_ok = bool(getattr(model, "square_ok", False))
    images = load_images(filelist, size=image_size, patch_size=patch_size, square_ok=square_ok, verbose=False)
    if len(images) == 1:
        images = [images[0], dict(images[0], idx=1, instance="1")]
    pairs = make_pairs(images, scene_graph=scene_graph, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=False)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer, verbose=False)
    scene.compute_global_alignment(init="mst", niter=niter, schedule="linear", lr=0.01)
    return [np.asarray(x) for x in to_numpy(scene.get_pts3d())]


def _resize_pointmap(points: np.ndarray, height: int, width: int) -> np.ndarray:
    if points.shape[:2] == (height, width):
        return points
    tensor = torch.from_numpy(points).permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0).numpy()


def evaluate_sequence(
    pred_maps: list[np.ndarray],
    gt_maps: np.ndarray,
    valid_mask: np.ndarray,
    dataset_name: str,
) -> dict[str, float]:
    """Apply TAPTQ's Sim(3)+ICP+normal metric to one sequence."""
    import open3d as o3d

    if len(pred_maps) != len(gt_maps):
        raise ValueError(f"View count mismatch: prediction={len(pred_maps)}, gt={len(gt_maps)}")
    pred = np.stack([_resize_pointmap(p, gt_maps.shape[1], gt_maps.shape[2]) for p in pred_maps], axis=0)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(gt_maps).all(axis=-1) & np.isfinite(pred).all(axis=-1)
    if not np.any(valid):
        raise ValueError("No valid points remain after external-model resizing")
    c, R, t = umeyama(pred[valid].T, gt_maps[valid].T)
    pred = c * np.einsum("nhwj,ij->nhwi", pred, R) + t.T
    pred_points = pred[valid].reshape(-1, 3)
    gt_points = gt_maps[valid].reshape(-1, 3)
    pred_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_points))
    gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_points))
    threshold = 100 if "dtu" in dataset_name.lower() else 0.1
    reg = o3d.pipelines.registration.registration_icp(
        pred_pcd, gt_pcd, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pred_pcd.transform(reg.transformation)
    pred_pcd.estimate_normals()
    gt_pcd.estimate_normals()
    acc, acc_med, nc1, nc1_med = accuracy(
        gt_pcd.points, pred_pcd.points, np.asarray(gt_pcd.normals), np.asarray(pred_pcd.normals)
    )
    comp, comp_med, nc2, nc2_med = completion(
        gt_pcd.points, pred_pcd.points, np.asarray(gt_pcd.normals), np.asarray(pred_pcd.normals)
    )
    return {
        "Acc-mean": float(acc), "Acc-med": float(acc_med),
        "Comp-mean": float(comp), "Comp-med": float(comp_med),
        "NC1-mean": float(nc1), "NC1-med": float(nc1_med),
        "NC2-mean": float(nc2), "NC2-med": float(nc2_med),
        "NC-mean": float((nc1 + nc2) / 2), "NC-med": float((nc1_med + nc2_med) / 2),
    }


def evaluate_dataset(dataset: Any, seq_id_map_path: str, model: torch.nn.Module, device: str, dataset_name: str, output_dir: str, image_size: int = 512, niter: int = 300) -> dict[str, float]:
    """Evaluate a TAPTQ-compatible dataset with the shared E1 protocol."""
    with open(seq_id_map_path, "r", encoding="utf-8") as handle:
        seq_id_map = json.load(handle)
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for seq_name, ids in seq_id_map.items():
        data = dataset.get_data(sequence_name=seq_name, ids=ids)
        pred_maps = predict_pointmaps(data["image_paths"], model, device, image_size=image_size, niter=niter)
        metrics = evaluate_sequence(pred_maps, data["pointclouds"], data["valid_mask"], dataset_name)
        rows.append({"seq": seq_name, **metrics})
    if not rows:
        raise ValueError("The sequence map is empty")
    keys = [k for k in rows[0] if k != "seq"]
    summary = {k: float(np.mean([row[k] for row in rows])) for k in keys}
    with open(Path(output_dir) / "_all_samples.json", "w", encoding="utf-8") as handle:
        json.dump({"rows": rows, "summary": summary}, handle, indent=2)
    LOGGER.info("%s: %s", dataset_name, summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("dust3r", "mast3r"), required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--dataset", choices=("7scenes",), default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--seq-id-map", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--load-img-size", type=int, default=518)
    parser.add_argument("--cache-file", default=None)
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    model = load_external_model(args.model, args.weights, args.device, args.repo_root)
    if args.dataset is None:
        LOGGER.info("Loaded %s; no dataset requested.", args.model)
        raise SystemExit(0)
    if not all((args.data_root, args.seq_id_map, args.output_dir)):
        raise ValueError("--dataset requires --data-root, --seq-id-map, and --output-dir")
    if args.dataset == "7scenes":
        dataset_path = Path(__file__).resolve().parents[1] / "datasets" / "sevenscenes.py"
        spec = importlib.util.spec_from_file_location("tmm_sevenscenes", dataset_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load SevenScenes dataset from {dataset_path}")
        dataset_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset_module)
        dataset = dataset_module.SevenScenes(
            SEVENSCENES_DIR=args.data_root,
            split="test",
            load_img_size=args.load_img_size,
            cache_file=args.cache_file or os.path.join(args.output_dir, "7scenes_cache.npy"),
        )
    evaluate_dataset(
        dataset=dataset,
        seq_id_map_path=args.seq_id_map,
        model=model,
        device=args.device,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        image_size=args.image_size,
        niter=args.niter,
    )
