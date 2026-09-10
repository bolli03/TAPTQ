"""Fake-quant layers shared by VGGT RTN, RepQ and ERQ baselines."""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AffineQParams:
    scale: torch.Tensor
    zero_point: torch.Tensor
    bits: int
    axis: Optional[int] = None


def _view(param, tensor, axis):
    if axis is None:
        return param
    shape = [1] * tensor.ndim
    shape[axis % tensor.ndim] = tensor.shape[axis % tensor.ndim]
    return param.reshape(shape)


def fake_quant(tensor, qparams, rounding="round"):
    scale = _view(qparams.scale.to(tensor.device, tensor.dtype), tensor, qparams.axis)
    zero = _view(qparams.zero_point.to(tensor.device, tensor.dtype), tensor, qparams.axis)
    value = tensor / scale + zero
    if rounding == "round":
        value = value.round()
    elif rounding == "floor":
        value = value.floor()
    elif rounding == "ceil":
        value = value.ceil()
    else:
        raise ValueError(f"Unknown rounding mode: {rounding}")
    return (value.clamp_(0, 2**qparams.bits - 1) - zero) * scale


@torch.no_grad()
def choose_qparams(tensor, bits, axis=None, candidates=20, min_ratio=0.5):
    """MSE grid search for unsigned affine layer/channel-wise qparams."""
    if not 2 <= bits <= 16 or candidates < 1:
        raise ValueError(f"Invalid bits/candidates: {bits}/{candidates}")
    values = tensor.detach().float()
    reduce = tuple(range(values.ndim)) if axis is None else tuple(
        dim for dim in range(values.ndim) if dim != axis % values.ndim
    )
    minimum = values.amin() if axis is None else values.amin(dim=reduce)
    maximum = values.amax() if axis is None else values.amax(dim=reduce)
    best_error = torch.full_like(minimum, float("inf"))
    best_scale = torch.ones_like(minimum)
    best_zero = torch.zeros_like(minimum)
    qmax = 2**bits - 1
    for ratio in torch.linspace(min_ratio, 1.0, candidates, device=values.device):
        origin = torch.zeros_like(minimum)
        low = torch.minimum(minimum * ratio, origin)
        high = torch.maximum(maximum * ratio, origin)
        scale = ((high - low) / qmax).clamp_min(1e-8)
        zero = (-low / scale).round().clamp(0, qmax)
        params = AffineQParams(scale, zero, bits, axis)
        errors = (values - fake_quant(values, params)).square()
        error = errors.mean() if axis is None else errors.mean(dim=reduce)
        improve = error < best_error
        best_error = torch.where(improve, error, best_error)
        best_scale = torch.where(improve, scale, best_scale)
        best_zero = torch.where(improve, zero, best_zero)
    return AffineQParams(best_scale, best_zero, bits, axis)


class BaselineQuantLinear(nn.Linear):
    """Linear with layer-wise activation fake quant and materialized QDQ weights."""
    def __init__(self, in_features, out_features, bias=True, a_bit=8):
        super().__init__(in_features, out_features, bias=bias)
        self.a_bit = int(a_bit)
        self.input_quant_enabled = False
        self.register_buffer("input_scale", torch.ones(()))
        self.register_buffer("input_zero_point", torch.zeros(()))
        self.register_buffer("input_calibrated", torch.tensor(False))

    @classmethod
    def from_float(cls, module, a_bit):
        result = cls(module.in_features, module.out_features, module.bias is not None, a_bit)
        result.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            result.bias.data.copy_(module.bias.data)
        return result.to(module.weight.device, module.weight.dtype)

    def set_input_qparams(self, qparams):
        if qparams.axis is not None or qparams.scale.numel() != 1:
            raise ValueError("Runtime activation qparams must be layer-wise")
        self.input_scale.copy_(qparams.scale.float().reshape(()))
        self.input_zero_point.copy_(qparams.zero_point.float().reshape(()))
        self.input_calibrated.fill_(True)

    def set_quant_state(self, enabled):
        self.input_quant_enabled = bool(enabled)

    def quantize_input(self, tensor):
        if not bool(self.input_calibrated):
            raise RuntimeError("Activation qparams are not calibrated")
        return fake_quant(tensor, AffineQParams(
            self.input_scale, self.input_zero_point, self.a_bit
        ))

    def forward(self, tensor):
        if self.input_quant_enabled:
            tensor = self.quantize_input(tensor)
        return F.linear(tensor, self.weight, self.bias)
