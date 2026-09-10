"""VGGT Linear adapter porting IST-DASLab GPTQ's official fasterquant algorithm.

Source: https://github.com/IST-DASLab/gptq, commit
2d65066eeb06a5c9ff5184d8cebdf33662c67faf.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .official_quant import QParams, affine_fake_quant, repq_qparams


def _weight_qparams(weight: torch.Tensor, bits: int):
    """Official Quantizer.find_params(weight=True, perchannel=True, sym=False)."""
    qmax = 2**bits - 1
    minimum = torch.minimum(weight.amin(dim=1), torch.zeros(weight.shape[0], device=weight.device))
    maximum = torch.maximum(weight.amax(dim=1), torch.zeros(weight.shape[0], device=weight.device))
    empty = (minimum == 0) & (maximum == 0)
    minimum[empty], maximum[empty] = -1, 1
    scale = ((maximum - minimum) / qmax).clamp_min(1e-8)
    zero = (-minimum / scale).round()
    return scale, zero


def _quantize_column(column: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int):
    qmax = 2**bits - 1
    return ((column / scale).round() + zero).clamp_(0, qmax).sub_(zero).mul_(scale)


class GPTQQuantLinear(nn.Module):
    """Official GPTQ weight update with RepQ activation quantizer for W4A8 fairness."""

    def __init__(self, module: nn.Linear, w_bit: int, a_bit: int):
        super().__init__()
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.w_bit = int(w_bit)
        self.a_bit = int(a_bit)
        self.register_buffer("qweight", module.weight.detach().clone())
        self.register_buffer("input_scale", torch.ones(()))
        self.register_buffer("input_zero_point", torch.zeros(()))
        self.register_buffer("input_calibrated", torch.tensor(False))
        self.register_buffer("weight_calibrated", torch.tensor(False))
        self.input_quant_enabled = False
        self.weight_quant_enabled = False
        self.bias = None if module.bias is None else nn.Parameter(module.bias.detach().clone())

    @classmethod
    def from_float(cls, module: nn.Linear, w_bit: int, a_bit: int):
        return cls(module, w_bit, a_bit).to(module.weight.device, module.weight.dtype)

    def set_quant_state(self, enabled: bool):
        self.input_quant_enabled = bool(enabled)
        self.weight_quant_enabled = bool(enabled)

    @torch.no_grad()
    def calibrate(
        self,
        inputs: torch.Tensor,
        _candidates: int = 20,
        damp_percent: float = 0.01,
        block_size: int = 128,
        group_size: int = -1,
        act_order: bool = False,
    ):
        x = inputs.reshape(-1, self.in_features).float().to(self.qweight.device)
        self._set_activation_qparams(repq_qparams(x, self.a_bit))
        weight = self.qweight.float().clone()
        hessian = (2.0 / max(1, x.shape[0])) * (x.transpose(0, 1) @ x)
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weight[:, dead] = 0

        if act_order:
            permutation = torch.argsort(torch.diag(hessian), descending=True)
            inverse = torch.argsort(permutation)
            weight = weight[:, permutation]
            hessian = hessian[permutation][:, permutation]
        else:
            permutation = inverse = None

        damp = float(damp_percent) * torch.mean(torch.diag(hessian))
        diagonal = torch.arange(self.in_features, device=hessian.device)
        hessian[diagonal, diagonal] += damp
        hinv = torch.linalg.cholesky(hessian)
        hinv = torch.cholesky_inverse(hinv)
        hinv = torch.linalg.cholesky(hinv, upper=True)
        quantized = torch.zeros_like(weight)

        for start in range(0, self.in_features, block_size):
            end = min(start + block_size, self.in_features)
            work = weight[:, start:end].clone()
            qblock = torch.zeros_like(work)
            errors = torch.zeros_like(work)
            local_hinv = hinv[start:end, start:end]
            for offset in range(end - start):
                column_index = start + offset
                column = work[:, offset]
                divisor = local_hinv[offset, offset]
                if group_size > 0:
                    group_start = (column_index // group_size) * group_size
                    group_end = min(group_start + group_size, self.in_features)
                    source_indices = torch.arange(group_start, group_end, device=weight.device)
                    if act_order:
                        source_indices = inverse[source_indices]
                    scale, zero = _weight_qparams(weight[:, source_indices], self.w_bit)
                else:
                    scale, zero = _weight_qparams(weight, self.w_bit)
                rounded = _quantize_column(column, scale, zero, self.w_bit)
                qblock[:, offset] = rounded
                error = (column - rounded) / divisor
                work[:, offset:].sub_(error.unsqueeze(1) @ local_hinv[offset, offset:].unsqueeze(0))
                errors[:, offset] = error
            quantized[:, start:end] = qblock
            if end < self.in_features:
                weight[:, end:].sub_(errors @ hinv[start:end, end:])

        if act_order:
            quantized = quantized[:, inverse]
        self.qweight.copy_(quantized.to(self.qweight.dtype))
        self.weight_calibrated.fill_(True)
        self.set_quant_state(True)

    def _set_activation_qparams(self, params: QParams):
        if params.axis is not None:
            raise ValueError("GPTQ fair W4A8 activation quantization must be layer-wise")
        self.input_scale.copy_(params.scale.reshape(()).float())
        self.input_zero_point.copy_(params.zero.reshape(()).float())
        self.input_calibrated.fill_(True)

    def forward(self, tensor: torch.Tensor):
        if not self.weight_quant_enabled:
            return F.linear(tensor, self.qweight, self.bias)
        if not bool(self.weight_calibrated):
            raise RuntimeError("GPTQ layer was used before calibration")
        if self.input_quant_enabled:
            if not bool(self.input_calibrated):
                raise RuntimeError("GPTQ activation quantizer was not calibrated")
            tensor = affine_fake_quant(tensor, QParams(
                self.input_scale, self.input_zero_point, self.a_bit
            ))
        return F.linear(tensor, self.qweight, self.bias)
