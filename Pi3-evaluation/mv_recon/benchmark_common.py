"""
Shared helpers for VGGT quant deploy benchmarks (latency, VRAM, size).

Used by benchmark_quant_deploy.py and compare_quant_fp.py.
"""
from __future__ import annotations

import importlib.util
import logging
import statistics
import tempfile
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from PTQ.vggt.models.vggt import VGGT


def load_ptq_module():
    path = Path(__file__).resolve().parent / "ptq.py"
    spec = importlib.util.spec_from_file_location("mv_recon_ptq", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def state_dict_bytes(model: torch.nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in model.state_dict().values())


def theoretical_linear_w4_bytes(model: torch.nn.Module) -> tuple[int, int]:
    """(packed int4 weight bytes, w_interval + bias fp32 est.) for modules with .weight and .w_interval."""
    w4 = 0
    extra = 0
    for m in model.modules():
        if hasattr(m, "weight") and hasattr(m, "w_interval") and getattr(m, "w_interval", None) is not None:
            n = m.weight.numel()
            w4 += (n + 1) // 2
            wi = m.w_interval
            if isinstance(wi, torch.Tensor):
                extra += wi.numel() * 4
            if getattr(m, "bias", None) is not None:
                extra += m.bias.numel() * 4
    return w4, extra


def build_dummy_batch(
    num_views: int, img_h: int, img_w: int, device: torch.device, align: int = 14
) -> torch.Tensor:
    h = (img_h // align) * align
    w = (img_w // align) * align
    return torch.randn(1, num_views, 3, h, w, device=device, dtype=torch.float32)


@torch.inference_mode()
def benchmark_forward(
    model: torch.nn.Module,
    imgs: torch.Tensor,
    warmup: int,
    iters: int,
    use_amp: bool,
    device: torch.device,
    aggregator_only: bool = False,
) -> dict[str, Any]:
    dtype = torch.bfloat16 if use_amp and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    def forward_once() -> None:
        if aggregator_only:
            _ = model.aggregator(imgs)
        else:
            _ = model(imgs)

    def one() -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if use_amp:
            with torch.amp.autocast(device.type, dtype=dtype, enabled=use_amp):
                forward_once()
        else:
            forward_once()
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    for _ in range(warmup):
        one()

    times = [one() for _ in range(iters)]
    return {
        "warmup": warmup,
        "iters": iters,
        "mean_ms": statistics.mean(times) * 1000,
        "stdev_ms": statistics.stdev(times) * 1000 if len(times) > 1 else 0.0,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
    }


@dataclass
class DeployBenchmarkParams:
    model_path: str
    config_name: str = "PTQ4ViT"
    w_bit: int = 4
    a_bit: int = 8
    linear_channelwise: bool = False
    metric: str = "hessian"
    quant_json: str | None = None
    quant_checkpoint: str | None = None
    no_quant: bool = False
    checkpoint: str | None = None
    num_views: int = 8
    img_h: int = 518
    img_w: int = 518
    warmup: int = 5
    iters: int = 20
    amp: bool = False
    # If True, time/memory only model.aggregator(imgs) (matches quant wrap on aggregator only).
    aggregator_only: bool = False


def default_model_path() -> str:
    return os.environ.get(
        "VGGT_MODEL_PATH",
        str(Path(__file__).resolve().parents[2] / "models" / "hf_hub" / "models--facebook--VGGT-1B"),
    )


def _torch_load_checkpoint(path: str | Path, map_location):
    load_kw = {"map_location": map_location}
    try:
        return torch.load(path, **load_kw, weights_only=False)
    except TypeError:
        return torch.load(path, **load_kw)


def run_deploy_benchmark(
    ptq_mod: Any,
    params: DeployBenchmarkParams,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """
    Load VGGT, optionally wrap/load quant + checkpoint, measure size and forward benchmark.
    """
    if logger is None:
        logger = logging.getLogger("benchmark_common")

    model = VGGT.from_pretrained(params.model_path).to(device).eval()

    ram_params_b = state_dict_bytes(model)
    with tempfile.NamedTemporaryFile(prefix="benchmark_vggt_", suffix=".pt", delete=False) as f:
        disk_path = Path(f.name)
    try:
        torch.save(model.state_dict(), disk_path)
        disk_full_b = disk_path.stat().st_size
    finally:
        disk_path.unlink(missing_ok=True)

    qwet_restored = False
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []

    linear_cw = params.linear_channelwise or params.config_name == "PTQ4ViT_channelwise"

    if not params.no_quant:
        quant_cfg = ptq_mod.init_config(params.config_name)
        mod = ptq_mod.cfg_modifier(
            linear_ptq_setting=(1, 1, 1),
            metric=params.metric,
            bit_setting=(params.w_bit, params.a_bit),
            linear_channelwise=linear_cw,
        )
        quant_cfg = mod(quant_cfg)
        if linear_cw:
            quant_cfg.linear_channelwise = True

        ptq_mod.wrap_modules_in_net(model, quant_cfg, quantize_aggregator=True)

        if params.quant_checkpoint:
            from mv_recon.taptq import load_checkpoint

            load_checkpoint(model, params.quant_checkpoint)
        elif params.quant_json:
            ptq_mod.model_load(model, params.quant_json, logger)
        else:
            logger.warning("No quant checkpoint: wrapped modules are NOT calibrated (random intervals).")

        ptq_mod.enable_quant(model)

        if params.checkpoint:
            ckpt = _torch_load_checkpoint(params.checkpoint, device)
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            incomp = model.load_state_dict(state, strict=False)
            missing_keys = list(incomp.missing_keys)
            unexpected_keys = list(incomp.unexpected_keys)
            qwet_restored = True
            if missing_keys:
                logger.warning("checkpoint missing_keys (first 10): %s", missing_keys[:10])
            if unexpected_keys:
                logger.warning("checkpoint unexpected_keys (first 10): %s", unexpected_keys[:10])

    ram_after_b = state_dict_bytes(model)
    w4_packed_b, w4_extra_b = theoretical_linear_w4_bytes(model)

    imgs = build_dummy_batch(params.num_views, params.img_h, params.img_w, device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    stats = benchmark_forward(
        model,
        imgs,
        params.warmup,
        params.iters,
        params.amp,
        device,
        aggregator_only=params.aggregator_only,
    )
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    out: dict[str, Any] = {
        "model_path": params.model_path,
        "forward_target": "aggregator" if params.aggregator_only else "full_model",
        "quant_json": params.quant_json,
        "quant_checkpoint": params.quant_checkpoint,
        "w_bit": params.w_bit,
        "a_bit": params.a_bit,
        "linear_channelwise": bool(linear_cw) if not params.no_quant else False,
        "config_name": params.config_name,
        "no_quant": params.no_quant,
        "input_shape": list(imgs.shape),
        "amp": params.amp,
        "latency_ms": stats,
        "memory_bytes": {
            "peak_allocated": peak_alloc,
            "peak_reserved": peak_reserved,
            "peak_allocated_mib": peak_alloc / (1024**2),
            "peak_reserved_mib": peak_reserved / (1024**2),
        },
        "size_bytes": {
            "state_dict_ram_initial": ram_params_b,
            "state_dict_ram_after_quant": ram_after_b,
            "theoretical_linear_w4_weight_packed": w4_packed_b,
            "theoretical_linear_w4_scales_bias_fp32_est": w4_extra_b,
            "disk_state_dict_torch_save_uncompressed": disk_full_b,
        },
        "qwet_checkpoint": params.checkpoint,
        "qwet_restored": qwet_restored,
        "checkpoint_missing_keys_count": len(missing_keys),
        "checkpoint_unexpected_keys_count": len(unexpected_keys),
        "note": (
            "PyTorch fake-quant (round + F.linear in float); not INT4 TensorCore/TensorRT. "
            "JSON does not include QwT A/B; use --checkpoint to restore full train-time state."
        ),
    }
    del model, imgs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out
