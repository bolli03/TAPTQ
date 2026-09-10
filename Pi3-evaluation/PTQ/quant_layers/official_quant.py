"""Quantization primitives ported from official RepQ-ViT and DeepCompressor."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QParams:
    scale: torch.Tensor
    zero: torch.Tensor
    bits: int
    axis: int | None = None


def _reshape(param: torch.Tensor, tensor: torch.Tensor, axis: int | None):
    if axis is None:
        return param
    shape = [1] * tensor.ndim
    shape[axis % tensor.ndim] = tensor.shape[axis % tensor.ndim]
    return param.reshape(shape)


def affine_fake_quant(tensor: torch.Tensor, params: QParams) -> torch.Tensor:
    scale = _reshape(params.scale.to(tensor.device, tensor.dtype), tensor, params.axis)
    zero = _reshape(params.zero.to(tensor.device, tensor.dtype), tensor, params.axis)
    qmax = 2**params.bits - 1
    return ((tensor / scale).round().add(zero).clamp_(0, qmax).sub(zero)).mul(scale)


def _percentile_channel_qparams(
    tensor: torch.Tensor,
    bits: int,
    percentiles: tuple[float, ...],
) -> QParams:
    """Search affine qparams independently for every row of a 2D weight matrix."""
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D weight matrix, got shape {tuple(tensor.shape)}")
    values = tensor.detach().float()
    best_error = torch.full((values.shape[0],), float("inf"), device=values.device)
    best_scale = torch.ones_like(best_error)
    best_zero = torch.zeros_like(best_error)
    qmax = 2**bits - 1
    for percentile in percentiles:
        high = torch.quantile(values, percentile, dim=1)
        low = torch.quantile(values, 1.0 - percentile, dim=1)
        scale = ((high - low) / qmax).clamp_min(1e-8)
        zero = (-low / scale).round()
        restored = (((values / scale[:, None]).round() + zero[:, None]).clamp(0, qmax) - zero[:, None]) * scale[:, None]
        error = (values - restored).square().mean(dim=1)
        improve = error < best_error
        best_error = torch.where(improve, error, best_error)
        best_scale = torch.where(improve, scale, best_scale)
        best_zero = torch.where(improve, zero, best_zero)
    return QParams(best_scale, best_zero, bits, axis=0)


def _masked_row_quantile(values: torch.Tensor, mask: torch.Tensor, percentile: float) -> torch.Tensor:
    """Vectorized row-wise quantile over a variable number of selected values."""
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise RuntimeError("ERQ two-part quantization requires both signs in every output row")
    ordered = values.masked_fill(~mask, float("inf")).sort(dim=1).values
    position = (counts - 1).float() * percentile
    lower = position.floor().long()
    upper = position.ceil().long()
    fraction = position - lower
    low_value = ordered.gather(1, lower[:, None]).squeeze(1)
    high_value = ordered.gather(1, upper[:, None]).squeeze(1)
    return low_value + (high_value - low_value) * fraction


@torch.no_grad()
def erq_two_part_qparams(weight: torch.Tensor, bits: int) -> tuple[QParams, torch.Tensor]:
    """Official ERQ qkv/fc1 two-part per-output-channel weight quantizer."""
    values = weight.detach().float()
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D weight matrix, got shape {tuple(values.shape)}")
    positive_threshold = _masked_row_quantile(values, values > 0, 0.99)
    negative_threshold = _masked_row_quantile(values, values < 0, 0.01)
    outlier_frequency = (
        (values > positive_threshold[:, None]) | (values < negative_threshold[:, None])
    ).sum(dim=0)
    top_count = min(values.shape[1] - 1, max(1, values.shape[0] // 20))
    top_indices = outlier_frequency.topk(top_count).indices.sort().values
    top_mask = torch.zeros(values.shape[1], dtype=torch.bool, device=values.device)
    top_mask[top_indices] = True
    other_indices = torch.nonzero(~top_mask, as_tuple=False).squeeze(1)
    percentiles = (0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1.0)
    top = _percentile_channel_qparams(values[:, top_indices], bits, percentiles)
    other = _percentile_channel_qparams(values[:, other_indices], bits, percentiles)
    scale = torch.empty_like(values)
    zero = torch.empty_like(values)
    scale[:, top_indices] = top.scale[:, None]
    zero[:, top_indices] = top.zero[:, None]
    scale[:, other_indices] = other.scale[:, None]
    zero[:, other_indices] = other.zero[:, None]
    return QParams(scale, zero, bits), top_indices


@torch.no_grad()
def repq_qparams(tensor: torch.Tensor, bits: int, axis: int | None = None) -> QParams:
    """RepQ-ViT official percentile-grid affine quantizer parameters."""
    values = tensor.detach().float()
    if axis is not None:
        axis %= values.ndim
        moved = values.movedim(axis, 0).reshape(values.shape[axis], -1)
        best_error = torch.full((moved.shape[0],), float("inf"), device=values.device)
        best_scale = torch.ones_like(best_error)
        best_zero = torch.zeros_like(best_error)
        qmax = 2**bits - 1
        for percentile in (0.999, 0.9999, 0.99999):
            high = torch.quantile(moved, percentile, dim=1)
            low = torch.quantile(moved, 1.0 - percentile, dim=1)
            scale = ((high - low) / qmax).clamp_min(1e-8)
            zero = (-low / scale).round()
            restored = (((moved / scale[:, None]).round() + zero[:, None]).clamp(0, qmax) - zero[:, None]) * scale[:, None]
            error = (moved - restored).square().mean(dim=1)
            improve = error < best_error
            best_error = torch.where(improve, error, best_error)
            best_scale = torch.where(improve, scale, best_scale)
            best_zero = torch.where(improve, zero, best_zero)
        return QParams(best_scale, best_zero, bits, axis)

    flat = values.reshape(-1)
    best_error = torch.tensor(float("inf"), device=values.device)
    best_scale = torch.ones((), device=values.device)
    best_zero = torch.zeros((), device=values.device)
    qmax = 2**bits - 1
    for percentile in (0.999, 0.9999, 0.99999):
        high = torch.quantile(flat, percentile)
        low = torch.quantile(flat, 1.0 - percentile)
        scale = ((high - low) / qmax).clamp_min(1e-8)
        zero = (-low / scale).round()
        quantized = ((values / scale).round() + zero).clamp(0, qmax)
        restored = (quantized - zero) * scale
        error = (values - restored).square().mean()
        if error < best_error:
            best_error, best_scale, best_zero = error, scale, zero
    return QParams(best_scale, best_zero, bits, None)


@torch.no_grad()
def symmetric_group_quant(tensor: torch.Tensor, bits: int, group_size: int, *, allow_unsigned: bool = False):
    """DeepCompressor-style signed symmetric group quantization along the last dim."""
    shape = tensor.shape
    width = shape[-1]
    padding = (-width) % group_size
    values = torch.nn.functional.pad(tensor.float(), (0, padding)) if padding else tensor.float()
    groups = values.reshape(*shape[:-1], -1, group_size)
    if allow_unsigned:
        nonnegative = groups.amin(dim=-1, keepdim=True) >= 0
    else:
        nonnegative = torch.zeros_like(groups[..., :1], dtype=torch.bool)
    signed_qmax = 2 ** (bits - 1) - 1
    unsigned_qmax = 2**bits - 1
    magnitude = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    signed_scale = magnitude / signed_qmax
    unsigned_scale = groups.amax(dim=-1, keepdim=True).clamp_min(1e-8) / unsigned_qmax
    scale = torch.where(nonnegative, unsigned_scale, signed_scale)
    signed = (groups / scale).round().clamp(-signed_qmax - 1, signed_qmax)
    unsigned = (groups / scale).round().clamp(0, unsigned_qmax)
    restored = torch.where(nonnegative, unsigned, signed) * scale
    restored = restored.reshape(*shape[:-1], -1)[..., :width]
    return restored.to(tensor.dtype), scale.squeeze(-1)
