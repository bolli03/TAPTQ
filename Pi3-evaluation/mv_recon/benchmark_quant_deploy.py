#!/usr/bin/env python3
"""
Benchmark W/A quantized VGGT for deployment-oriented metrics on GPU.

Measures (with current codebase):
  - Latency: forward pass with torch.cuda.synchronize (warmup + mean/std/min).
  - Memory: torch.cuda.max_memory_allocated / reserved peak around forward.
  - Size: on-disk state_dict bytes + in-RAM parameter bytes; optional W4-theory for Linear weights.

IMPORTANT — what "W4A8 deployment" means here:
  Quant layers use fake quantization: weights/activations are still stored and computed
  mainly in FP16/BF32 (see quant_forward), not packed INT4 kernels. Latency/memory will
  reflect THIS path. True INT4 TensorCore/CUTLASS/TensorRT deployment needs separate export.

Usage (from Pi3-evaluation repo root):
  python mv_recon/benchmark_quant_deploy.py \\
    --quant-json param/vggt/channelwise_8scan_w4a8.json \\
    --w-bit 4 --a-bit 8 \\
    --linear-channelwise \\
    --num-views 8 --img-h 518 --img-w 518

  # Only backbone (matches quant-on-aggregator): add --aggregator-only

  # FP baseline (no wrap / no json): omit --quant-json and add --no-quant

  Model directory can be overridden with env VGGT_MODEL_PATH.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import torch

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def _load_benchmark_common():
    path = Path(__file__).resolve().parent / "benchmark_common.py"
    spec = importlib.util.spec_from_file_location("benchmark_common", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description="VGGT quant deploy benchmark (latency / VRAM / size)")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="HF local snapshot dir (default: VGGT_MODEL_PATH env or hub cache path in code)",
    )
    parser.add_argument("--quant-json", type=str, default=None, help="JSON from ptq save (quant modules + intervals)")
    parser.add_argument(
        "--quant-checkpoint",
        type=str,
        default=None,
        help="TAPTQ quant checkpoint (.txt/.json/.pt); supports legacy text records and is preferred over --quant-json",
    )
    parser.add_argument("--config-name", type=str, default="PTQ4ViT")
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument("--a-bit", type=int, default=8)
    parser.add_argument("--linear-channelwise", action="store_true")
    parser.add_argument("--metric", type=str, default="hessian")
    parser.add_argument("--no-quant", action="store_true", help="Skip wrap/load; FP baseline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional state_dict .pt after QwT training; loaded with strict=False after quant json",
    )
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--img-h", type=int, default=518)
    parser.add_argument("--img-w", type=int, default=518)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--amp", action="store_true", help="Use autocast bf16/fp16 like infer_mv_pointclouds")
    parser.add_argument(
        "--aggregator-only",
        action="store_true",
        help="Benchmark only model.aggregator(imgs) instead of full VGGT forward",
    )
    parser.add_argument("--save-json", type=str, default=None, help="Write metrics to this path")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required for meaningful deploy benchmark.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    logger = logging.getLogger("benchmark_quant")
    logging.basicConfig(level=logging.INFO)

    bc = _load_benchmark_common()
    model_path = args.model_path or bc.default_model_path()
    ptq = bc.load_ptq_module()
    params = bc.DeployBenchmarkParams(
        model_path=model_path,
        config_name=args.config_name,
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        linear_channelwise=args.linear_channelwise,
        metric=args.metric,
        quant_json=args.quant_json,
        quant_checkpoint=args.quant_checkpoint,
        no_quant=args.no_quant,
        checkpoint=args.checkpoint,
        num_views=args.num_views,
        img_h=args.img_h,
        img_w=args.img_w,
        warmup=args.warmup,
        iters=args.iters,
        amp=args.amp,
        aggregator_only=args.aggregator_only,
    )
    out = bc.run_deploy_benchmark(ptq, params, device, logger)

    print(json.dumps(out, indent=2))
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
