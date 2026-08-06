#!/usr/bin/env python3
"""
Export VGGT (or Aggregator only) to ONNX for TensorRT / ORT workflows.

This does NOT embed PTQ JSON intervals; use TensorRT QAT/DQ or a separate
scale-injection pipeline for INT4 (see mv_recon/tensorrt/README.md).

Examples (from Pi3-evaluation repo root):
  python mv_recon/export_vggt_onnx.py --target aggregator --out /tmp/vggt_agg.onnx
  python mv_recon/export_vggt_onnx.py --target world_points --out /tmp/vggt_pts.onnx

Env: VGGT_MODEL_PATH
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from PTQ.vggt.models.vggt import VGGT


class AggregatorLastOut(nn.Module):
    """ONNX-friendly wrapper: last fused token tensor + patch_start_idx as tensor."""

    def __init__(self, aggregator: nn.Module):
        super().__init__()
        self.aggregator = aggregator

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_list, patch_start_idx = self.aggregator(images)
        last = out_list[-1]
        idx = torch.tensor([patch_start_idx], device=images.device, dtype=torch.int64)
        return last, idx


class VGGTWorldPoints(nn.Module):
    """Export subset: world_points head input path is inside forward; returns 5D tensor."""

    def __init__(self, vggt: VGGT):
        super().__init__()
        self.vggt = vggt

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pred = self.vggt(images)
        return pred["world_points"]


def main():
    parser = argparse.ArgumentParser(description="Export VGGT subgraph to ONNX")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local HF snapshot (default: VGGT_MODEL_PATH or hub cache)",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=("aggregator", "world_points"),
        default="aggregator",
        help="aggregator: patch_embed+backbone last tokens; world_points: full VGGT world_points output",
    )
    parser.add_argument("--out", type=str, required=True, help="Output .onnx path")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-views", type=int, default=2, help="S in [B,S,3,H,W]; use 2 for smaller trace")
    parser.add_argument("--img-h", type=int, default=518)
    parser.add_argument("--img-w", type=int, default=518)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="Try torch.onnx.export(dynamo=True); falls back to legacy exporter on failure",
    )
    args = parser.parse_args()

    default_path = os.environ.get(
        "VGGT_MODEL_PATH",
        "/root/autodl-tmp/hf_hub/models--facebook--VGGT-1B",
    )
    model_path = args.model_path or default_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vggt = VGGT.from_pretrained(model_path).to(device).eval()

    b, s, h, w = args.batch, args.num_views, args.img_h, args.img_w
    dummy = torch.randn(b, s, 3, h, w, device=device, dtype=torch.float32)

    if args.target == "aggregator":
        wrapper = AggregatorLastOut(vggt.aggregator).to(device).eval()
        in_names = ["images"]
        out_names = ["tokens_last", "patch_start_idx"]
    else:
        wrapper = VGGTWorldPoints(vggt).to(device).eval()
        in_names = ["images"]
        out_names = ["world_points"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _export_legacy() -> None:
        torch.onnx.export(
            wrapper,
            dummy,
            str(out_path),
            input_names=in_names,
            output_names=out_names,
            opset_version=args.opset,
            do_constant_folding=True,
        )

    try:
        if args.dynamo:
            torch.onnx.export(
                wrapper,
                dummy,
                str(out_path),
                input_names=in_names,
                output_names=out_names,
                opset_version=args.opset,
                do_constant_folding=True,
                dynamo=True,
            )
        else:
            _export_legacy()
    except Exception as e:
        if args.dynamo:
            print(f"dynamo export failed ({e}); retrying legacy exporter...", file=sys.stderr)
            _export_legacy()
        else:
            raise

    print(f"Wrote ONNX: {out_path.resolve()}")
    print(f"  target={args.target} input_shape={tuple(dummy.shape)} opset={args.opset}")


if __name__ == "__main__":
    main()
