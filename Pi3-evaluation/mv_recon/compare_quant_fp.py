#!/usr/bin/env python3
"""
Compare FP32 VGGT baseline vs W4A8 (JSON + PyTorch fake-quant) on the same GPU input.

Outputs a text table and optional merged JSON (fp_baseline + w4a8_pytorch).

  python mv_recon/compare_quant_fp.py \\
    --quant-json param/vggt/channelwise_8scan_w4a8.json \\
    --config-name PTQ4ViT_channelwise \\
    --w-bit 4 --a-bit 8 \\
    --amp --warmup 10 --iters 50 \\
    --save-json outputs/compare_w4a8_fp.json

Env: VGGT_MODEL_PATH overrides default HF snapshot directory.
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


def _print_table(fp: dict, w4: dict) -> None:
    def row(label, fpv, w4v, unit=""):
        print(f"{label:42} {fpv:>14} {w4v:>14} {unit}")

    fpl = fp["latency_ms"]
    w4l = w4["latency_ms"]
    fpm = fp["memory_bytes"]
    w4m = w4["memory_bytes"]
    fps = fp["size_bytes"]
    w4s = w4["size_bytes"]

    print()
    print("Metric                              FP baseline       W4A8 (JSON)")
    print("-" * 72)
    row("latency mean (ms)", f"{fpl['mean_ms']:.3f}", f"{w4l['mean_ms']:.3f}")
    row("latency stdev (ms)", f"{fpl['stdev_ms']:.3f}", f"{w4l['stdev_ms']:.3f}")
    row("latency min (ms)", f"{fpl['min_ms']:.3f}", f"{w4l['min_ms']:.3f}")
    row("peak VRAM allocated (MiB)", f"{fpm['peak_allocated_mib']:.1f}", f"{w4m['peak_allocated_mib']:.1f}")
    row("peak VRAM reserved (MiB)", f"{fpm['peak_reserved_mib']:.1f}", f"{w4m['peak_reserved_mib']:.1f}")
    row("state_dict RAM (MiB)", f"{fps['state_dict_ram_after_quant'] / 2**20:.1f}", f"{w4s['state_dict_ram_after_quant'] / 2**20:.1f}")
    row("theoretical W4 packed + scales est. (MiB)", "-", f"{(w4s['theoretical_linear_w4_weight_packed'] + w4s['theoretical_linear_w4_scales_bias_fp32_est']) / 2**20:.1f}")
    print("-" * 72)
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare FP vs W4A8 (PyTorch fake-quant) on VGGT")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--quant-json",
        type=str,
        default="param/vggt/channelwise_8scan_w4a8.json",
        help="Channel-wise W4A8 calibration JSON",
    )
    parser.add_argument("--config-name", type=str, default="PTQ4ViT_channelwise")
    parser.add_argument("--w-bit", type=int, default=4)
    parser.add_argument("--a-bit", type=int, default=8)
    parser.add_argument(
        "--linear-channelwise",
        action="store_true",
        help="Force linear_channelwise on quant cfg (not needed if config-name is PTQ4ViT_channelwise)",
    )
    parser.add_argument("--metric", type=str, default="hessian")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional QwT state_dict; applied only to the W4A8 run after JSON load",
    )
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--img-h", type=int, default=518)
    parser.add_argument("--img-w", type=int, default=518)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--aggregator-only",
        action="store_true",
        help="Benchmark model.aggregator only (FP and W4A8 runs)",
    )
    parser.add_argument("--save-json", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(1)

    bc = _load_benchmark_common()
    device = torch.device("cuda")
    logger = logging.getLogger("compare_quant_fp")
    logging.basicConfig(level=logging.INFO)

    model_path = args.model_path or bc.default_model_path()
    lc = args.linear_channelwise or (args.config_name == "PTQ4ViT_channelwise")

    ptq = bc.load_ptq_module()

    fp_params = bc.DeployBenchmarkParams(
        model_path=model_path,
        no_quant=True,
        num_views=args.num_views,
        img_h=args.img_h,
        img_w=args.img_w,
        warmup=args.warmup,
        iters=args.iters,
        amp=args.amp,
        aggregator_only=args.aggregator_only,
    )
    logger.info("Running FP baseline...")
    fp_result = bc.run_deploy_benchmark(ptq, fp_params, device, logger)

    w4_params = bc.DeployBenchmarkParams(
        model_path=model_path,
        config_name=args.config_name,
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        linear_channelwise=lc,
        metric=args.metric,
        quant_json=args.quant_json,
        no_quant=False,
        checkpoint=args.checkpoint,
        num_views=args.num_views,
        img_h=args.img_h,
        img_w=args.img_w,
        warmup=args.warmup,
        iters=args.iters,
        amp=args.amp,
        aggregator_only=args.aggregator_only,
    )
    logger.info("Running W4A8 (json)...")
    w4_result = bc.run_deploy_benchmark(ptq, w4_params, device, logger)

    merged = {
        "fp_baseline": fp_result,
        "w4a8_pytorch": w4_result,
        "comparison_note": (
            "W4A8 path uses PyTorch fake quantization from JSON intervals only; "
            "QwT tensors require --checkpoint. TensorRT INT4 is separate; see mv_recon/tensorrt/README.md."
        ),
    }

    _print_table(fp_result, w4_result)

    print(json.dumps(merged, indent=2))
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        logger.info("Wrote %s", args.save_json)


if __name__ == "__main__":
    main()
