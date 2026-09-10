#!/usr/bin/env python3
"""VGGT TAPTQ W8A8 deployment with Triton INT8 GEMM fused epilogues."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import rootutils
import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from PTQ.vggt.models.vggt import VGGT  # noqa: E402
from deployment.w4a8_int8_deploy import (
    _expand_weight_scale,
    _get_submodule,
    _set_submodule,
    benchmark_model,
    build_input,
    resolve_dtype,
)  # noqa: E402

POST_GELU_NEGATIVE_BOUND = 0.16997124254703522


@triton.jit
def _quantize_kernel(x_ptr, q_ptr, scale_ptr, size: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr)
    q = libdevice.rint(x / scale)
    q = tl.maximum(-128.0, tl.minimum(127.0, q)).to(tl.int8)
    tl.store(q_ptr + offsets, q, mask=mask)


@triton.jit
def _quantize_dual_kernel(
    x_ptr, q_pos_ptr, q_neg_ptr, pos_scale_ptr, neg_scale_ptr,
    size: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    pos_scale = tl.load(pos_scale_ptr)
    neg_scale = tl.load(neg_scale_ptr)
    q_pos = libdevice.rint(x / pos_scale)
    q_neg = libdevice.rint(x / neg_scale)
    q_pos = tl.maximum(0.0, tl.minimum(127.0, q_pos)).to(tl.int8)
    q_neg = tl.maximum(-128.0, tl.minimum(0.0, q_neg)).to(tl.int8)
    tl.store(q_pos_ptr + offsets, q_pos, mask=mask)
    tl.store(q_neg_ptr + offsets, q_neg, mask=mask)

@triton.jit
def _gelu_quantize_kernel(
    x_ptr, q_ptr, q_neg_ptr, scale_ptr, neg_scale_ptr,
    size: tl.constexpr, DUAL: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = 0.5 * x * (1.0 + libdevice.erf(x * 0.7071067811865476))
    scale = tl.load(scale_ptr)
    q = libdevice.rint(x / scale)
    if DUAL:
        q = tl.maximum(0.0, tl.minimum(127.0, q)).to(tl.int8)
        neg_scale = tl.load(neg_scale_ptr)
        q_neg = libdevice.rint(x / neg_scale)
        q_neg = tl.maximum(-128.0, tl.minimum(0.0, q_neg)).to(tl.int8)
        tl.store(q_neg_ptr + offsets, q_neg, mask=mask)
    else:
        q = tl.maximum(-128.0, tl.minimum(127.0, q)).to(tl.int8)
    tl.store(q_ptr + offsets, q, mask=mask)



_CONFIGS = [
    triton.Config({"BM": 64, "BN": 64, "BK": 32}, num_warps=4, num_stages=4),
    triton.Config({"BM": 64, "BN": 128, "BK": 32}, num_warps=4, num_stages=4),
    triton.Config({"BM": 128, "BN": 64, "BK": 32}, num_warps=4, num_stages=4),
    triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=8, num_stages=4),
    triton.Config({"BM": 64, "BN": 128, "BK": 64}, num_warps=8, num_stages=3),
    triton.Config({"BM": 64, "BN": 256, "BK": 64}, num_warps=8, num_stages=3),
    triton.Config({"BM": 128, "BN": 256, "BK": 32}, num_warps=8, num_stages=3),
    triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_warps=8, num_stages=3),
    triton.Config({"BM": 256, "BN": 128, "BK": 32}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K", "DUAL"])
@triton.jit
def _gemm_epilogue_kernel(
    a_ptr, a_neg_ptr, w_ptr, out_ptr, w_scale_ptr,
    a_scale_ptr, a_neg_scale_ptr, bias_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    DUAL: tl.constexpr, HAS_BIAS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.int32)
    if DUAL:
        acc_neg = tl.zeros((BM, BN), tl.int32)
    for start in range(0, K, BK):
        k = start + offs_k
        mask_a = (offs_m[:, None] < M) & (k[None, :] < K)
        mask_w = (k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptr + offs_m[:, None] * K + k[None, :], mask=mask_a, other=0)
        w = tl.load(w_ptr + offs_n[None, :] * K + k[:, None], mask=mask_w, other=0)
        acc += tl.dot(a, w)
        if DUAL:
            a_neg = tl.load(a_neg_ptr + offs_m[:, None] * K + k[None, :], mask=mask_a, other=0)
            acc_neg += tl.dot(a_neg, w)
    w_scale = tl.load(w_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    a_scale = tl.load(a_scale_ptr)
    out = acc.to(tl.float32) * (a_scale * w_scale[None, :])
    if DUAL:
        a_neg_scale = tl.load(a_neg_scale_ptr)
        out += acc_neg.to(tl.float32) * (a_neg_scale * w_scale[None, :])
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        out += bias[None, :]
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], out,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def export_checkpoint(checkpoint: str, output: str) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or "state_dict" not in payload or "quant_records" not in payload:
        raise ValueError(f"Not a TAPTQ checkpoint: {checkpoint}")
    state = payload["state_dict"]
    compensation = [
        key
        for key in state
        if key.endswith("lora_weight")
        or key.endswith("lora_bias")
        or ".taptq_qwt." in key
    ]
    if compensation:
        raise ValueError(f"Only quant-only checkpoints are supported; found {len(compensation)} QwT tensors")
    entries = []
    post_gelu_count = 0
    weight_bytes = 0
    for record in payload["quant_records"]:
        name = str(record["name"])
        weight = state.get(f"{name}.weight")
        if weight is None or weight.ndim != 2:
            continue
        if int(record["w_qmax"]) != 128 or int(record["a_qmax"]) != 128:
            raise ValueError(f"{name}: checkpoint is not W8A8")
        weight = weight.detach().float()
        out_features, in_features = weight.shape
        raw_w_scale = torch.as_tensor(record["w_interval"])
        squeezed_w_scale = raw_w_scale.squeeze()
        if squeezed_w_scale.ndim > 1:
            raise ValueError(f"{name}: unsupported W8 scale layout {tuple(raw_w_scale.shape)}")
        if squeezed_w_scale.ndim == 1 and squeezed_w_scale.numel() not in (1, out_features) and raw_w_scale.shape[-2] != 1:
            raise ValueError(f"{name}: input-block W8 scales are not supported: {tuple(raw_w_scale.shape)}")
        w_scale = _expand_weight_scale(raw_w_scale, out_features)
        a_scale = torch.as_tensor(record["a_interval"]).detach().float().squeeze()
        if a_scale.numel() != 1:
            raise ValueError(f"{name}: expected static per-tensor A8, got {a_scale.shape}")
        qweight = torch.round(weight / w_scale[:, None]).clamp(-128, 127).to(torch.int8)
        is_post_gelu = str(record.get("module", "")).startswith("PostGelu")
        post_gelu_count += int(is_post_gelu)
        bias = state.get(f"{name}.bias")
        entries.append({
            "name": name,
            "shape": [out_features, in_features],
            "weight_int8": qweight.contiguous(),
            "w_scale": w_scale.float().contiguous(),
            "a_scale": a_scale.reshape(()).float(),
            "a_neg_scale": torch.tensor(POST_GELU_NEGATIVE_BOUND / 128) if is_post_gelu else None,
            "activation_scheme": "post_gelu_dual_scale" if is_post_gelu else "symmetric",
            "bias": bias.detach().float().contiguous() if isinstance(bias, torch.Tensor) else None,
        })
        weight_bytes += qweight.numel()
    if not entries:
        raise RuntimeError("No W8A8 Linear entries found")
    artifact = {
        "format": "taptq_vggt_w8a8_triton_v1",
        "source_checkpoint": checkpoint,
        "weight_storage": "signed int8 row-major [out,in]",
        "entries": entries,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
    result = {
        "output": str(path), "num_linears": len(entries),
        "post_gelu_linears": post_gelu_count, "weight_bytes": weight_bytes,
        "artifact_bytes": path.stat().st_size,
    }
    print(json.dumps(result, indent=2))
    return result


class FusedW8A8Linear(nn.Module):
    """A8 quant kernel plus INT8 GEMM with fused dequant/scale/bias epilogue."""

    def __init__(self, qweight, w_scale, a_scale, bias, a_neg_scale, post_gelu_mode):
        super().__init__()
        self.out_features, self.in_features = map(int, qweight.shape)
        self.register_buffer("weight_int8", qweight.to(torch.int8).contiguous())
        self.register_buffer("w_scale", w_scale.float().contiguous())
        self.register_buffer("a_scale", a_scale.float().reshape(()))
        if a_neg_scale is not None and post_gelu_mode == "dual":
            self.register_buffer("a_neg_scale", a_neg_scale.float().reshape(()))
        else:
            self.a_neg_scale = None
        if bias is None:
            self.register_buffer("bias", torch.zeros(1, dtype=torch.float32))
            self.has_bias = False
        else:
            self.register_buffer("bias", bias.float().contiguous())
            self.has_bias = True

    def forward_from_gelu(self, x: torch.Tensor) -> torch.Tensor:
        x_2d = x.reshape(-1, self.in_features)
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        m, k = x_2d.shape
        q = torch.empty_like(x_2d, dtype=torch.int8)
        dual = self.a_neg_scale is not None
        q_neg = torch.empty_like(q) if dual else q
        grid = (triton.cdiv(x_2d.numel(), 1024),)
        _gelu_quantize_kernel[grid](
            x_2d,
            q,
            q_neg,
            self.a_scale,
            self.a_neg_scale if dual else self.a_scale,
            x_2d.numel(),
            DUAL=dual,
            BLOCK=1024,
        )
        output = torch.empty((m, self.out_features), device=x.device, dtype=x.dtype)
        gemm_grid = lambda meta: (
            triton.cdiv(m, meta["BM"]), triton.cdiv(self.out_features, meta["BN"])
        )
        _gemm_epilogue_kernel[gemm_grid](
            q, q_neg, self.weight_int8, output, self.w_scale,
            self.a_scale, self.a_neg_scale if dual else self.a_scale, self.bias,
            m, self.out_features, k, DUAL=dual, HAS_BIAS=self.has_bias,
        )
        return output.reshape(*x.shape[:-1], self.out_features)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_2d = x.reshape(-1, self.in_features)
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        m, k = x_2d.shape
        q = torch.empty_like(x_2d, dtype=torch.int8)
        dual = self.a_neg_scale is not None
        grid = (triton.cdiv(x_2d.numel(), 1024),)
        if dual:
            q_neg = torch.empty_like(q)
            _quantize_dual_kernel[grid](
                x_2d, q, q_neg, self.a_scale, self.a_neg_scale,
                x_2d.numel(), BLOCK=1024,
            )
        else:
            q_neg = q
            _quantize_kernel[grid](
                x_2d, q, self.a_scale, x_2d.numel(), BLOCK=1024
            )
        output = torch.empty((m, self.out_features), device=x.device, dtype=x.dtype)
        gemm_grid = lambda meta: (
            triton.cdiv(m, meta["BM"]), triton.cdiv(self.out_features, meta["BN"])
        )
        _gemm_epilogue_kernel[gemm_grid](
            q, q_neg, self.weight_int8, output, self.w_scale,
            self.a_scale, self.a_neg_scale if dual else self.a_scale, self.bias,
            m, self.out_features, k, DUAL=dual, HAS_BIAS=self.has_bias,
        )
        return output.reshape(*x.shape[:-1], self.out_features)


def load_model(model_path, artifact_path, device, dtype, post_gelu_mode):
    started = time.perf_counter()
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False, mmap=True)
    if artifact.get("format") != "taptq_vggt_w8a8_triton_v1":
        raise ValueError(f"Unsupported artifact: {artifact.get('format')}")
    if post_gelu_mode not in {"dual", "single"}:
        raise ValueError(f"Unsupported post_gelu_mode: {post_gelu_mode}")
    entries = artifact.get("entries", [])
    if len(entries) != 288 or len({entry.get("name") for entry in entries}) != 288:
        raise ValueError("W8A8 artifact must contain 288 unique Linear entries")
    model = VGGT.from_pretrained(model_path).to(device=device, dtype=dtype).eval()
    if dtype != torch.float32:
        for name in ("camera_head", "depth_head", "point_head", "track_head"):
            head = getattr(model, name, None)
            if head is not None:
                head.float()
    replaced = 0
    post_gelu = 0
    weight_bytes = 0
    for entry in entries:
        qweight = entry["weight_int8"]
        w_scale = entry["w_scale"]
        a_scale = entry["a_scale"]
        bias = entry.get("bias")
        if tuple(qweight.shape) != tuple(entry["shape"]) or qweight.dtype != torch.int8:
            raise ValueError(f"{entry['name']}: invalid INT8 weight shape or dtype")
        if w_scale.numel() != entry["shape"][0] or not torch.isfinite(w_scale).all() or not (w_scale > 0).all():
            raise ValueError(f"{entry['name']}: invalid weight scale")
        if a_scale.numel() != 1 or not torch.isfinite(a_scale).all() or not (a_scale > 0).all():
            raise ValueError(f"{entry['name']}: invalid activation scale")
        if bias is not None and bias.numel() != entry["shape"][0]:
            raise ValueError(f"{entry['name']}: invalid bias shape")
        target = _get_submodule(model, entry["name"])
        if not isinstance(target, nn.Linear) or tuple(target.weight.shape) != tuple(entry["shape"]):
            raise ValueError(f"{entry['name']}: target mismatch")
        layer = FusedW8A8Linear(qweight, w_scale, a_scale, bias, entry.get("a_neg_scale"), post_gelu_mode).to(device)
        _set_submodule(model, entry["name"], layer)
        post_gelu += int(entry.get("a_neg_scale") is not None)
        weight_bytes += qweight.numel()
        replaced += 1
    if replaced != 288:
        raise RuntimeError(f"Expected 288 Linear replacements, got {replaced}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.cuda.synchronize(device)
    return model, {
        "load_seconds": time.perf_counter() - started,
        "num_replaced_linears": replaced, "post_gelu_linears": post_gelu,
        "artifact_bytes": Path(artifact_path).stat().st_size,
        "runtime_int8_weight_bytes": weight_bytes,
        "backend": "triton_fused", "post_gelu_mode": post_gelu_mode,
        "kernel_structure": "producer_A8_quant + fused_GELU_for_FC2 + INT8_GEMM_dequant_scale_bias",
    }


def benchmark_suite(args) -> dict[str, Any]:
    device = torch.device(args.device)
    dtype = resolve_dtype(args.non_quant_dtype)
    inputs = build_input(args.batch_size, args.num_views, args.height, args.width, device)
    report = {
        "device": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
        "triton_version": triton.__version__, "input_shape": list(inputs.shape),
        "target": args.target, "execution_mode": args.execution_mode,
        "warmup": args.warmup, "iterations": args.iterations,
        "backend": "triton_fused", "post_gelu_mode": args.post_gelu_mode,
    }
    fp = VGGT.from_pretrained(args.model_path).to(device=device, dtype=torch.float32).eval()
    report["fp32"] = benchmark_model(
        fp, inputs, args.target, args.warmup, args.iterations, args.execution_mode
    )
    del fp
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    low_inputs = inputs.to(dtype)
    low = VGGT.from_pretrained(args.model_path).to(device=device, dtype=dtype).eval()
    if args.target == "full" and dtype != torch.float32:
        for name in ("camera_head", "depth_head", "point_head", "track_head"):
            head = getattr(low, name, None)
            if head is not None:
                head.float()
    report[args.non_quant_dtype] = benchmark_model(
        low, low_inputs, args.target, args.warmup, args.iterations, args.execution_mode
    )
    del low, inputs
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    model, metadata = load_model(
        args.model_path, args.artifact, device, dtype, args.post_gelu_mode
    )
    report["w8a8"] = benchmark_model(
        model, low_inputs, args.target, args.warmup, args.iterations, args.execution_mode
    )
    report["w8a8"]["load"] = metadata
    fp_ms = report["fp32"]["latency"]["mean_ms"]
    low_ms = report[args.non_quant_dtype]["latency"]["mean_ms"]
    q_ms = report["w8a8"]["latency"]["mean_ms"]
    report["comparison"] = {
        "latency_speedup_vs_fp32": fp_ms / q_ms,
        f"latency_speedup_vs_{args.non_quant_dtype}": low_ms / q_ms,
        "model_storage_ratio": report["w8a8"]["model_storage_bytes"] / report["fp32"]["model_storage_bytes"],
        "peak_allocated_ratio": report["w8a8"]["cuda_peak_allocated_bytes"] / report["fp32"]["cuda_peak_allocated_bytes"],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    bench = commands.add_parser("benchmark")
    bench.add_argument("--model-path", required=True)
    bench.add_argument("--artifact", required=True)
    bench.add_argument("--device", default="cuda:0")
    bench.add_argument("--target", choices=("aggregator", "full"), default="aggregator")
    bench.add_argument("--post-gelu-mode", choices=("dual", "single"), default="dual")
    bench.add_argument("--non-quant-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    bench.add_argument("--execution-mode", choices=("eager", "cuda_graph"), default="cuda_graph")
    bench.add_argument("--batch-size", type=int, default=1)
    bench.add_argument("--num-views", type=int, default=8)
    bench.add_argument("--height", type=int, default=518)
    bench.add_argument("--width", type=int, default=518)
    bench.add_argument("--warmup", type=int, default=5)
    bench.add_argument("--iterations", type=int, default=20)
    bench.add_argument("--output")
    args = parser.parse_args()
    if args.command == "export":
        export_checkpoint(args.checkpoint, args.output)
        return
    result = benchmark_suite(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
