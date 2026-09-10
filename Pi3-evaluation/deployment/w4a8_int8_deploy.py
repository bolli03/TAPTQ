#!/usr/bin/env python3
"""VGGT TAPTQ W4A8 deployment on CUDA using INT8 Tensor Core GEMM.

Weights use calibrated 4-bit values and packed 4-bit storage on disk. At load time
weights are unpacked into int8 containers because PyTorch 2.5 exposes
``torch._int_mm`` (INT8 x INT8 -> INT32), but no native mixed INT4 x INT8 kernel.
Activations use the calibrated static A8 scale. This removes fake-quant weight QDQ
and FP32 Linear GEMMs while preserving TAPTQ W4A8 quantization semantics.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import rootutils
import torch
import torch.nn as nn

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from PTQ.vggt.models.vggt import VGGT  # noqa: E402


def _get_submodule(module: nn.Module, path: str) -> nn.Module:
    current: Any = module
    for part in path.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def _set_submodule(module: nn.Module, path: str, value: nn.Module) -> None:
    parts = path.split(".")
    parent = _get_submodule(module, ".".join(parts[:-1]))
    if parts[-1].isdigit():
        parent[int(parts[-1])] = value  # type: ignore[index]
    else:
        setattr(parent, parts[-1], value)


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack signed values in [-8, 7] into two two's-complement nibbles per byte."""
    flat = values.to(torch.int16).reshape(-1)
    codes = torch.bitwise_and(flat, 0xF).to(torch.uint8)
    if codes.numel() % 2:
        codes = torch.cat((codes, torch.zeros(1, dtype=torch.uint8)))
    return (codes[0::2] | (codes[1::2] << 4)).contiguous()


def unpack_int4(packed: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    count = math.prod(shape)
    packed = packed.to(torch.uint8).reshape(-1)
    codes = torch.empty(packed.numel() * 2, dtype=torch.uint8)
    codes[0::2] = packed & 0xF
    codes[1::2] = packed >> 4
    signed = codes[:count].to(torch.int8)
    signed[signed >= 8] -= 16
    return signed.reshape(shape).contiguous()


def _expand_weight_scale(scale: torch.Tensor, out_features: int) -> torch.Tensor:
    scale = scale.detach().float().squeeze()
    if scale.ndim == 0:
        return scale.expand(out_features).clone()
    if scale.numel() == out_features:
        return scale.reshape(out_features).contiguous()
    if scale.numel() < out_features and out_features % scale.numel() == 0:
        return scale.reshape(-1).repeat_interleave(out_features // scale.numel()).contiguous()
    raise ValueError(f"Cannot expand weight scale {tuple(scale.shape)} to {out_features}")


def export_taptq_checkpoint(checkpoint: str, output: str) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or "state_dict" not in payload or "quant_records" not in payload:
        raise ValueError(f"Not a TAPTQ checkpoint: {checkpoint}")
    state = payload["state_dict"]
    compensation_keys = [
        key for key in state if key.endswith("lora_weight") or key.endswith("lora_bias")
    ]
    if compensation_keys:
        raise ValueError(
            "This deployment exporter supports quant-only TAPTQ checkpoints; "
            f"found {len(compensation_keys)} QwT/compensation tensors"
        )
    entries: list[dict[str, Any]] = []
    logical_weight_bytes = 0
    packed_weight_bytes = 0

    for record in payload["quant_records"]:
        name = str(record["name"])
        weight_key = f"{name}.weight"
        if weight_key not in state:
            continue
        weight = state[weight_key].detach().float()
        if weight.ndim != 2:
            continue
        w_qmax = int(record["w_qmax"])
        a_qmax = int(record["a_qmax"])
        if w_qmax != 8 or a_qmax != 128:
            raise ValueError(
                f"{name}: checkpoint is not W4A8 (w_qmax={w_qmax}, a_qmax={a_qmax})"
            )
        out_features, in_features = weight.shape
        w_scale = _expand_weight_scale(record["w_interval"], out_features)
        a_scale_tensor = torch.as_tensor(record["a_interval"]).detach().float().squeeze()
        if a_scale_tensor.numel() != 1:
            raise ValueError(f"{name}: only static per-tensor A8 scale is supported, got {a_scale_tensor.shape}")
        qweight = torch.round(weight / w_scale[:, None]).clamp(-8, 7).to(torch.int8)
        packed = pack_int4(qweight)
        bias = state.get(f"{name}.bias")
        is_post_gelu = str(record.get("module", "")).startswith("PostGelu")
        entry = {
            "name": name,
            "shape": [out_features, in_features],
            "packed_weight": packed,
            "w_scale": w_scale.float(),
            "a_scale": a_scale_tensor.reshape(()).float(),
            "a_neg_scale": (
                torch.tensor(0.16997124254703522 / 128, dtype=torch.float32)
                if is_post_gelu
                else None
            ),
            "activation_scheme": "post_gelu_dual_scale" if is_post_gelu else "symmetric",
            "w_qmax": 8,
            "a_qmax": 128,
            "bias": bias.detach().float() if isinstance(bias, torch.Tensor) else None,
        }
        entries.append(entry)
        logical_weight_bytes += qweight.numel() // 2
        packed_weight_bytes += packed.numel()

    if not entries:
        raise RuntimeError("No quantized Linear entries were exported")
    artifact = {
        "format": "taptq_vggt_w4a8_int8_gemm_v1",
        "source_checkpoint": str(checkpoint),
        "backend": "torch._int_mm",
        "weight_storage": "packed signed int4 on disk; int8 container at runtime",
        "activation": "static symmetric int8",
        "entries": entries,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    result = {
        "output": str(output_path),
        "num_linears": len(entries),
        "packed_weight_bytes": packed_weight_bytes,
        "logical_weight_bytes": logical_weight_bytes,
        "artifact_bytes": output_path.stat().st_size,
    }
    print(json.dumps(result, indent=2))
    return result


class W4A8PostGeluLinear(nn.Module):
    """Latency-oriented fallback for dual-scale Post-GELU activations.

    It preserves calibrated W4/A8 values, but dequantizes the W4 values once so
    the 72 dual-scale FC2 layers use one Tensor Core GEMM rather than two INT8
    GEMMs. The remaining 216 Linear layers stay on the integer backend.
    """

    def __init__(
        self,
        qweight: torch.Tensor,
        w_scale: torch.Tensor,
        a_scale: torch.Tensor,
        a_neg_scale: torch.Tensor,
        bias: torch.Tensor | None,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        out_features, in_features = qweight.shape
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.compute_dtype = compute_dtype
        self.register_buffer(
            "weight", (qweight.float() * w_scale.float()[:, None]).to(compute_dtype).contiguous()
        )
        self.register_buffer("a_scale", a_scale.float().reshape(()))
        self.register_buffer("a_neg_scale", a_neg_scale.float().reshape(()))
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.to(compute_dtype).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        x_pos = torch.round(x_2d / self.a_scale).clamp_(0, 127).mul_(self.a_scale)
        x_neg = torch.round(x_2d / self.a_neg_scale).clamp_(-128, 0).mul_(self.a_neg_scale)
        quantized = (x_pos + x_neg).to(self.compute_dtype)
        output = torch.nn.functional.linear(quantized, self.weight, self.bias)
        return output.reshape(*original_shape[:-1], self.out_features).to(x.dtype)


class W4A8FP8Linear(nn.Module):
    """Calibrated W4/A8 values executed by Hopper FP8 Tensor Cores.

    The packed artifact and integer quantization grids are unchanged. Quantized
    integer values are represented in E4M3 only for the GEMM because PyTorch 2.5
    does not expose a native mixed INT4 x INT8 kernel.
    """

    def __init__(
        self,
        qweight: torch.Tensor,
        w_scale: torch.Tensor,
        a_scale: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        out_features, in_features = qweight.shape
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.padded_out_features = math.ceil(out_features / 32) * 32
        if self.padded_out_features != out_features:
            qweight = torch.nn.functional.pad(qweight, (0, 0, 0, self.padded_out_features - out_features))
        self.register_buffer("weight_fp8", qweight.to(torch.float8_e4m3fn).contiguous())
        self.register_buffer("output_scale", (w_scale.float() * a_scale.float()).contiguous())
        self.register_buffer("a_scale", a_scale.float().reshape(()))
        self.register_buffer("unit_scale", torch.ones(1, dtype=torch.float32))
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.float().contiguous())

    def _matmul(self, x_fp8: torch.Tensor) -> torch.Tensor:
        rows = x_fp8.shape[0]
        gemm_rows = 4096 if rows > 4096 else math.ceil(rows / 16) * 16
        chunks = []
        weight_t = self.weight_fp8.t()
        for start in range(0, rows, gemm_rows):
            chunk = x_fp8[start : start + gemm_rows]
            chunk_rows = chunk.shape[0]
            if chunk_rows != gemm_rows:
                chunk = torch.nn.functional.pad(chunk, (0, 0, 0, gemm_rows - chunk_rows))
            output = torch._scaled_mm(
                chunk.contiguous(),
                weight_t,
                scale_a=self.unit_scale,
                scale_b=self.unit_scale,
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
            )[:chunk_rows, : self.out_features]
            chunks.append(output)
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        x_fp8 = torch.round(x_2d / self.a_scale).clamp_(-128, 127).to(torch.float8_e4m3fn)
        output = self._matmul(x_fp8).float().mul_(self.output_scale)
        if self.bias is not None:
            output.add_(self.bias)
        return output.reshape(*original_shape[:-1], self.out_features).to(x.dtype)


class W4A8IntLinear(nn.Module):
    """Static A8, calibrated W4-value Linear backed by INT8 Tensor Core GEMM."""

    def __init__(
        self,
        qweight: torch.Tensor,
        w_scale: torch.Tensor,
        a_scale: torch.Tensor,
        bias: torch.Tensor | None,
        a_neg_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        out_features, in_features = qweight.shape
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.padded_out_features = math.ceil(out_features / 32) * 32
        if self.padded_out_features != out_features:
            qweight = torch.nn.functional.pad(qweight, (0, 0, 0, self.padded_out_features - out_features))
        self.register_buffer("weight_int_t", qweight.t().contiguous().to(torch.int8))
        self.register_buffer("output_scale", (w_scale.float() * a_scale.float()).contiguous())
        self.register_buffer("a_scale", a_scale.float().reshape(()))
        if a_neg_scale is None:
            self.a_neg_scale = None
            self.output_scale_neg = None
        else:
            self.register_buffer("a_neg_scale", a_neg_scale.float().reshape(()))
            self.register_buffer(
                "output_scale_neg", (w_scale.float() * a_neg_scale.float()).contiguous()
            )
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.float().contiguous())

    def _integer_matmul(self, x_int: torch.Tensor) -> torch.Tensor:
        rows = x_int.shape[0]
        gemm_rows = 4096 if rows > 4096 else math.ceil(rows / 16) * 16
        accum_chunks = []
        for start in range(0, rows, gemm_rows):
            chunk = x_int[start : start + gemm_rows]
            chunk_rows = chunk.shape[0]
            if chunk_rows != gemm_rows:
                chunk = torch.nn.functional.pad(chunk, (0, 0, 0, gemm_rows - chunk_rows))
            accum = torch._int_mm(chunk.contiguous(), self.weight_int_t)[:chunk_rows, : self.out_features]
            accum_chunks.append(accum)
        return accum_chunks[0] if len(accum_chunks) == 1 else torch.cat(accum_chunks, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        if self.a_neg_scale is None:
            x_int = torch.round(x_2d / self.a_scale).clamp_(-128, 127).to(torch.int8)
            output = self._integer_matmul(x_int).float().mul_(self.output_scale)
        else:
            x_pos = torch.round(x_2d / self.a_scale).clamp_(0, 127).to(torch.int8)
            x_neg = torch.round(x_2d / self.a_neg_scale).clamp_(-128, 0).to(torch.int8)
            output = self._integer_matmul(x_pos).float().mul_(self.output_scale)
            output.add_(self._integer_matmul(x_neg).float().mul_(self.output_scale_neg))
        if self.bias is not None:
            output.add_(self.bias)
        return output.reshape(*original_shape[:-1], self.out_features).to(x.dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, backend=torch._int_mm"


def load_w4a8_model(
    model_path: str,
    artifact_path: str,
    device: torch.device,
    post_gelu_backend: str = "dual_int8",
    non_quant_dtype: torch.dtype = torch.float32,
    linear_backend: str = "int8",
) -> tuple[VGGT, dict[str, Any]]:
    t0 = time.perf_counter()
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False, mmap=True)
    if linear_backend == "fp8_scaled_mm" and torch.cuda.get_device_capability(device)[0] < 9:
        raise RuntimeError("fp8_scaled_mm requires Hopper (SM90+) CUDA hardware")
    if artifact.get("format") != "taptq_vggt_w4a8_int8_gemm_v1":
        raise ValueError(f"Unsupported artifact format: {artifact.get('format')}")
    model = VGGT.from_pretrained(model_path).to(device=device, dtype=non_quant_dtype).eval()
    if non_quant_dtype != torch.float32:
        for head_name in ("camera_head", "depth_head", "point_head", "track_head"):
            head = getattr(model, head_name, None)
            if head is not None:
                head.float()
    replaced = 0
    runtime_int8_weight_bytes = 0
    runtime_float_weight_bytes = 0
    for entry in artifact["entries"]:
        shape = tuple(int(v) for v in entry["shape"])
        target = _get_submodule(model, entry["name"])
        if not isinstance(target, nn.Linear):
            raise TypeError(f"{entry['name']}: expected nn.Linear target, got {type(target).__name__}")
        if tuple(target.weight.shape) != shape:
            raise ValueError(
                f"{entry['name']}: target shape {tuple(target.weight.shape)} != artifact shape {shape}"
            )
        qweight = unpack_int4(entry["packed_weight"], shape)
        if entry.get("a_neg_scale") is not None and post_gelu_backend == "hybrid_fp16":
            layer = W4A8PostGeluLinear(
                qweight=qweight,
                w_scale=entry["w_scale"],
                a_scale=entry["a_scale"],
                a_neg_scale=entry["a_neg_scale"],
                bias=entry.get("bias"),
                compute_dtype=non_quant_dtype,
            )
        elif linear_backend == "fp8_scaled_mm" and entry.get("a_neg_scale") is None:
            layer = W4A8FP8Linear(
                qweight=qweight,
                w_scale=entry["w_scale"],
                a_scale=entry["a_scale"],
                bias=entry.get("bias"),
            )
        else:
            layer = W4A8IntLinear(
                qweight=qweight,
                w_scale=entry["w_scale"],
                a_scale=entry["a_scale"],
                a_neg_scale=entry.get("a_neg_scale"),
                bias=entry.get("bias"),
            )
        layer.to(device)
        _set_submodule(model, entry["name"], layer)
        installed = _get_submodule(model, entry["name"])
        if installed is not layer:
            raise RuntimeError(f"{entry['name']}: replacement was not installed")
        if isinstance(layer, W4A8IntLinear):
            runtime_int8_weight_bytes += layer.weight_int_t.numel() * layer.weight_int_t.element_size()
        elif isinstance(layer, W4A8FP8Linear):
            runtime_int8_weight_bytes += layer.weight_fp8.numel() * layer.weight_fp8.element_size()
        else:
            runtime_float_weight_bytes += layer.weight.numel() * layer.weight.element_size()
        replaced += 1
    if replaced != len(artifact["entries"]):
        raise RuntimeError(f"Replaced {replaced} of {len(artifact['entries'])} artifact Linear layers")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.cuda.synchronize(device)
    return model, {
        "load_seconds": time.perf_counter() - t0,
        "num_replaced_linears": replaced,
        "artifact_bytes": Path(artifact_path).stat().st_size,
        "runtime_int8_weight_bytes": runtime_int8_weight_bytes,
        "runtime_float_weight_bytes": runtime_float_weight_bytes,
        "runtime_deployed_weight_bytes": runtime_int8_weight_bytes + runtime_float_weight_bytes,
        "backend": linear_backend,
        "post_gelu_backend": post_gelu_backend,
        "non_quant_dtype": str(non_quant_dtype).removeprefix("torch."),
        "task_head_dtype": "float32",
        "int8_output_alignment": 32,
        "int8_max_gemm_rows": 4096,
    }


def build_input(batch_size: int, num_views: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    height = height // 14 * 14
    width = width // 14 * 14
    generator = torch.Generator(device=device).manual_seed(42)
    return torch.rand(batch_size, num_views, 3, height, width, device=device, generator=generator)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


@torch.inference_mode()
def benchmark_callable(
    fn: Callable[[], Any],
    warmup: int,
    iterations: int,
    scans: int,
    views: int,
    device: torch.device,
) -> dict[str, float]:
    with torch.cuda.device(device):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        timings: list[float] = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
    mean_ms = statistics.mean(timings)
    return {
        "mean_ms": mean_ms,
        "p50_ms": percentile(timings, 0.50),
        "p95_ms": percentile(timings, 0.95),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "scans_per_second": scans * 1000.0 / mean_ms,
        "views_per_second": views * 1000.0 / mean_ms,
    }


def _make_cuda_graph(fn: Callable[[], Any], device: torch.device) -> Callable[[], Any]:
    with torch.cuda.device(device):
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                static_output = fn()
        torch.cuda.current_stream(device).wait_stream(capture_stream)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = fn()

    def replay() -> Any:
        graph.replay()
        return static_output

    return replay


def _prepare_callable(
    model: nn.Module,
    inputs: torch.Tensor,
    target: str,
    execution_mode: str,
) -> Callable[[], Any]:
    if target == "aggregator":
        module = model.aggregator
    elif target == "full":
        module = model
    else:
        raise ValueError(target)

    if execution_mode == "compile":
        module = torch.compile(module, mode="reduce-overhead", fullgraph=False, dynamic=False)
    fn = lambda: module(inputs)
    if execution_mode == "cuda_graph":
        return _make_cuda_graph(fn, inputs.device)
    if execution_mode not in ("eager", "compile"):
        raise ValueError(execution_mode)
    return fn


def model_storage_bytes(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        pointer = tensor.untyped_storage().data_ptr()
        if pointer in seen:
            continue
        seen.add(pointer)
        total += tensor.untyped_storage().nbytes()
    return total


def benchmark_model(
    model: nn.Module,
    inputs: torch.Tensor,
    target: str,
    warmup: int,
    iterations: int,
    execution_mode: str,
) -> dict[str, Any]:
    device = inputs.device
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        fn = _prepare_callable(model, inputs, target, execution_mode)
        torch.cuda.synchronize(device)
        prepared_allocated = torch.cuda.memory_allocated(device)
        prepared_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        timing = benchmark_callable(
            fn,
            warmup,
            iterations,
            inputs.shape[0],
            inputs.shape[0] * inputs.shape[1],
            device,
        )
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    return {
        "latency": timing,
        "model_storage_bytes": model_storage_bytes(model),
        "cuda_prepared_allocated_bytes": prepared_allocated,
        "cuda_prepared_reserved_bytes": prepared_reserved,
        "cuda_replay_peak_allocated_bytes": peak_allocated,
        "cuda_replay_peak_reserved_bytes": peak_reserved,
        "cuda_idle_allocated_bytes": prepared_allocated,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "execution_mode": execution_mode,
    }


def release_model(model: nn.Module) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def benchmark_suite(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    non_quant_dtype = resolve_dtype(args.non_quant_dtype)
    inputs = build_input(args.batch_size, args.num_views, args.height, args.width, device)
    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "input_shape": list(inputs.shape),
        "target": args.target,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "execution_mode": args.execution_mode,
        "fp_baseline_dtype": "float32",
        "w4a8_non_quant_dtype": args.non_quant_dtype,
    }

    fp_model = VGGT.from_pretrained(args.model_path).to(device=device, dtype=torch.float32).eval()
    for parameter in fp_model.parameters():
        parameter.requires_grad_(False)
    report["fp32"] = benchmark_model(
        fp_model, inputs, args.target, args.warmup, args.iterations, args.execution_mode
    )
    del fp_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    quant_inputs = inputs.to(non_quant_dtype)
    if non_quant_dtype != torch.float32:
        low_precision_model = VGGT.from_pretrained(args.model_path).to(
            device=device, dtype=non_quant_dtype
        ).eval()
        for parameter in low_precision_model.parameters():
            parameter.requires_grad_(False)
        report[args.non_quant_dtype] = benchmark_model(
            low_precision_model,
            quant_inputs,
            args.target,
            args.warmup,
            args.iterations,
            args.execution_mode,
        )
        del low_precision_model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    del inputs
    quant_model, load_meta = load_w4a8_model(
        args.model_path,
        args.artifact,
        device,
        post_gelu_backend=args.post_gelu_backend,
        non_quant_dtype=non_quant_dtype,
        linear_backend=args.linear_backend,
    )
    report["w4a8"] = benchmark_model(
        quant_model,
        quant_inputs,
        args.target,
        args.warmup,
        args.iterations,
        args.execution_mode,
    )
    report["w4a8"]["load"] = load_meta
    del quant_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    fp_latency = report["fp32"]["latency"]["mean_ms"]
    quant_latency = report["w4a8"]["latency"]["mean_ms"]
    report["comparison"] = {
        "latency_speedup_vs_fp32": fp_latency / quant_latency,
        "throughput_speedup_vs_fp32": report["w4a8"]["latency"]["views_per_second"]
        / report["fp32"]["latency"]["views_per_second"],
        "latency_speedup": fp_latency / quant_latency,
        "throughput_speedup": report["w4a8"]["latency"]["views_per_second"]
        / report["fp32"]["latency"]["views_per_second"],
        "model_storage_ratio": report["w4a8"]["model_storage_bytes"]
        / report["fp32"]["model_storage_bytes"],
        "peak_allocated_ratio": report["w4a8"]["cuda_peak_allocated_bytes"]
        / report["fp32"]["cuda_peak_allocated_bytes"],
    }
    if args.non_quant_dtype in report:
        low_precision_latency = report[args.non_quant_dtype]["latency"]["mean_ms"]
        report["comparison"][f"latency_speedup_vs_{args.non_quant_dtype}"] = (
            low_precision_latency / quant_latency
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export packed W4A8 artifact from TAPTQ checkpoint")
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--output", required=True)

    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark FP32 and deployed W4A8")
    benchmark_parser.add_argument("--model-path", required=True)
    benchmark_parser.add_argument("--artifact", required=True)
    benchmark_parser.add_argument("--device", default="cuda:0")
    benchmark_parser.add_argument("--target", choices=("aggregator", "full"), default="aggregator")
    benchmark_parser.add_argument(
        "--post-gelu-backend",
        choices=("dual_int8", "hybrid_fp16"),
        default="hybrid_fp16",
    )
    benchmark_parser.add_argument(
        "--linear-backend",
        choices=("int8", "fp8_scaled_mm"),
        default="fp8_scaled_mm",
        help="Hopper FP8 carries calibrated W4/A8 grid values when native INT4xINT8 is unavailable",
    )
    benchmark_parser.add_argument(
        "--non-quant-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
        help="dtype for LayerNorm/attention/heads and the hybrid Post-GELU GEMM",
    )
    benchmark_parser.add_argument(
        "--execution-mode",
        choices=("eager", "cuda_graph", "compile"),
        default="cuda_graph",
        help="fixed-shape CUDA Graph removes launch overhead; compile requires Triton",
    )
    benchmark_parser.add_argument("--batch-size", type=int, default=1)
    benchmark_parser.add_argument("--num-views", type=int, default=4)
    benchmark_parser.add_argument("--height", type=int, default=518)
    benchmark_parser.add_argument("--width", type=int, default=518)
    benchmark_parser.add_argument("--warmup", type=int, default=5)
    benchmark_parser.add_argument("--iterations", type=int, default=20)
    benchmark_parser.add_argument("--output")

    args = parser.parse_args()
    if args.command == "export":
        export_taptq_checkpoint(args.checkpoint, args.output)
        return
    report = benchmark_suite(args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
