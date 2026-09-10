#!/usr/bin/env python3
"""Validate deployed INT-GEMM VGGT W4A8 output against TAPTQ fake quant."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from PTQ.vggt.models.vggt import VGGT  # noqa: E402
from deployment.w4a8_int8_deploy import build_input, load_w4a8_model, resolve_dtype  # noqa: E402
from mv_recon import benchmark_common  # noqa: E402
from mv_recon.taptq import load_checkpoint  # noqa: E402


def stage_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference = reference.float()
    actual = actual.float()
    difference = actual - reference
    return {
        "mean_abs": float(difference.abs().mean()),
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(difference.norm() / reference.norm().clamp_min(1e-12)),
        "cosine": float(torch.nn.functional.cosine_similarity(reference.flatten(), actual.flatten(), dim=0)),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-views", type=int, default=1)
    parser.add_argument(
        "--post-gelu-backend",
        choices=("dual_int8", "hybrid_fp16"),
        default="hybrid_fp16",
    )
    parser.add_argument(
        "--non-quant-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--linear-backend",
        choices=("int8", "fp8_scaled_mm"),
        default="fp8_scaled_mm",
    )
    parser.add_argument("--height", type=int, default=518)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--min-last-cosine", type=float, default=0.99)
    parser.add_argument("--max-last-relative-l2", type=float, default=0.15)
    parser.add_argument("--output")
    args = parser.parse_args()

    device = torch.device(args.device)
    inputs = build_input(1, args.num_views, args.height, args.width, device)
    ptq = benchmark_common.load_ptq_module()
    fake_model = VGGT.from_pretrained(args.model_path).to(device).eval()
    quant_cfg = ptq.cfg_modifier(
        linear_ptq_setting=(1, 1, 1),
        metric="hessian",
        bit_setting=(4, 8),
        linear_channelwise=True,
    )(ptq.init_config("PTQ4ViT"))
    quant_cfg.linear_channelwise = True
    ptq.wrap_modules_in_net(fake_model, quant_cfg, quantize_aggregator=True)
    load_checkpoint(fake_model, args.checkpoint)
    ptq.enable_quant(fake_model)
    fake_stages, fake_patch_start = fake_model.aggregator(inputs)
    fake_stages_cpu = [tensor.detach().cpu() for tensor in fake_stages]
    del fake_model, fake_stages
    gc.collect()
    torch.cuda.empty_cache()

    deploy_dtype = resolve_dtype(args.non_quant_dtype)
    deploy_model, metadata = load_w4a8_model(
        args.model_path,
        args.artifact,
        device,
        post_gelu_backend=args.post_gelu_backend,
        non_quant_dtype=deploy_dtype,
        linear_backend=args.linear_backend,
    )
    int_stages, int_patch_start = deploy_model.aggregator(inputs.to(deploy_dtype))
    if int(fake_patch_start) != int(int_patch_start):
        raise RuntimeError(f"patch_start_idx mismatch: {fake_patch_start} != {int_patch_start}")
    if len(fake_stages_cpu) != len(int_stages):
        raise RuntimeError(f"stage count mismatch: {len(fake_stages_cpu)} != {len(int_stages)}")
    for index, (reference, actual) in enumerate(zip(fake_stages_cpu, int_stages)):
        if tuple(reference.shape) != tuple(actual.shape):
            raise RuntimeError(
                f"stage {index} shape mismatch: {tuple(reference.shape)} != {tuple(actual.shape)}"
            )
    if metadata["num_replaced_linears"] != 288:
        raise RuntimeError(f"Expected 288 replaced Linear layers, got {metadata['num_replaced_linears']}")
    metrics = [
        stage_metrics(reference, actual.detach().cpu())
        for reference, actual in zip(fake_stages_cpu, int_stages)
    ]
    report = {
        "input_shape": list(inputs.shape),
        "patch_start_idx_match": True,
        "stages": metrics,
        "deployment": metadata,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    last = metrics[-1]
    if last["cosine"] < args.min_last_cosine:
        raise RuntimeError(
            f"Last-stage cosine {last['cosine']:.6f} < required {args.min_last_cosine:.6f}"
        )
    if last["relative_l2"] > args.max_last_relative_l2:
        raise RuntimeError(
            f"Last-stage relative L2 {last['relative_l2']:.6f} > allowed {args.max_last_relative_l2:.6f}"
        )


if __name__ == "__main__":
    main()
