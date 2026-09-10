#!/usr/bin/env python3
import argparse
import csv
import json
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch


QUANT_ROOT = Path(__file__).resolve().parents[1]
TMM_ROOT = Path(os.environ.get("TMM_EVAL_ROOT", "/tmp/tmm-eval-code"))
sys.path.insert(0, str(QUANT_ROOT))
sys.path.insert(1, str(TMM_ROOT))

from evaluation.run_7andN import load_model
from datasets.eth3d import ETH3D
from mv_recon.utils import accuracy, completion, umeyama
from utils.interfaces import infer_mv_pointclouds


def parse_args():
    parser = argparse.ArgumentParser("QuantVGGT ETH3D Pi3-E1 evaluator")
    parser.add_argument("--bits", required=True, choices=("w4a8", "w6a6", "w8a8"))
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--quant-output-root", required=True)
    parser.add_argument("--calib-cache", required=True)
    parser.add_argument("--eth3d-dir", required=True)
    parser.add_argument("--seq-id-map", required=True)
    parser.add_argument("--cache-file", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    bit_map = {"w4a8": (4, 8), "w6a6": (6, 6), "w8a8": (8, 8)}
    wbit, abit = bit_map[args.bits]
    quant_eval_dir = QUANT_ROOT / "evaluation"
    assets = Path(args.quant_output_root)
    outputs_link = quant_eval_dir / "outputs"
    if outputs_link.is_symlink() or outputs_link.exists():
        outputs_link.unlink() if outputs_link.is_symlink() else None
    if not outputs_link.exists():
        outputs_link.symlink_to(assets, target_is_directory=True)

    os.chdir(quant_eval_dir)
    model, _ = load_model(
        "cuda",
        each_nsamples=0,
        min_num_images=0,
        num_frames=0,
        category=["apple"],
        co3d_anno_dir="/tmp",
        co3d_dir="/tmp",
        model_path=args.model_path,
        dtype=f"quarot_{args.bits}",
        resume_qs=True,
        lwc=True,
        lac=True,
        exp_name=args.exp_name,
        cache_path=args.calib_cache,
    )
    if model is None:
        raise RuntimeError("QuantVGGT loader did not return a model")

    dataset = ETH3D(ETH3D_DIR=args.eth3d_dir, load_img_size=518, cache_file=args.cache_file)
    with open(args.seq_id_map) as f:
        seq_id_map = json.load(f)

    output_root = Path(args.output_dir) / "ETH3D"
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "_all_samples.csv"
    if csv_path.exists():
        csv_path.unlink()

    fields = ["seq", "Acc-mean", "Acc-med", "Comp-mean", "Comp-med", "NC1-mean", "NC1-med", "NC2-mean", "NC2-med"]
    with csv_path.open("w", newline="") as csv_file, torch.no_grad():
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for sequence_name, ids in seq_id_map.items():
            data = dataset.get_data(sequence_name=sequence_name, ids=ids)
            filelist = data["image_paths"]
            images = data["images"]
            gt_pts = data["pointclouds"]
            valid_mask = data["valid_mask"]
            height, width = images.shape[-2:]
            pred_pts = infer_mv_pointclouds(filelist, model, argparse.Namespace(load_img_size=518, device="cuda", verbose=False), (height, width))
            if pred_pts.shape != gt_pts.shape:
                raise RuntimeError(f"{sequence_name}: prediction {pred_pts.shape} != ground truth {gt_pts.shape}")

            scale, rotation, translation = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
            pred_pts = scale * np.einsum("nhwj,ij->nhwi", pred_pts, rotation) + translation.T
            colors = images.permute(0, 2, 3, 1)[valid_mask].cpu().numpy().reshape(-1, 3)
            pred = pred_pts[valid_mask].reshape(-1, 3)
            gt = gt_pts[valid_mask].reshape(-1, 3)

            pred_cloud = o3d.geometry.PointCloud()
            pred_cloud.points = o3d.utility.Vector3dVector(pred)
            pred_cloud.colors = o3d.utility.Vector3dVector(colors)
            gt_cloud = o3d.geometry.PointCloud()
            gt_cloud.points = o3d.utility.Vector3dVector(gt)
            gt_cloud.colors = o3d.utility.Vector3dVector(colors)
            transform = o3d.pipelines.registration.registration_icp(
                pred_cloud,
                gt_cloud,
                0.1,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            ).transformation
            pred_cloud.transform(transform)
            pred_cloud.estimate_normals()
            gt_cloud.estimate_normals()
            pred_normal = np.asarray(pred_cloud.normals)
            gt_normal = np.asarray(gt_cloud.normals)
            acc, acc_med, nc1, nc1_med = accuracy(gt_cloud.points, pred_cloud.points, gt_normal, pred_normal)
            comp, comp_med, nc2, nc2_med = completion(gt_cloud.points, pred_cloud.points, gt_normal, pred_normal)
            writer.writerow({
                "seq": sequence_name,
                "Acc-mean": acc,
                "Acc-med": acc_med,
                "Comp-mean": comp,
                "Comp-med": comp_med,
                "NC1-mean": nc1,
                "NC1-med": nc1_med,
                "NC2-mean": nc2,
                "NC2-med": nc2_med,
            })
            csv_file.flush()
            print(f"[{sequence_name}] Acc={acc:.6f} Comp={comp:.6f} NC={(nc1 + nc2) / 2:.6f}", flush=True)


if __name__ == "__main__":
    main()
