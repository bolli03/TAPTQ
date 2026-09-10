"""VGGT SVDQuant adapter aligned with DeepCompressor's SVDQuant configuration.

Reference: nunchaku-ai/deepcompressor commit
69f3473f5e1c1504bae35cc50c7858ef900a9b17.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .official_quant import symmetric_group_quant


class SVDQuantLinear(nn.Module):
    """Smooth + low-rank-first + signed group-quantized Linear."""

    def __init__(
        self,
        module: nn.Linear,
        w_bit: int,
        a_bit: int,
        rank: int,
        group_size: int = 64,
        num_grids: int = 20,
    ):
        super().__init__()
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.w_bit = int(w_bit)
        self.a_bit = int(a_bit)
        self.rank = int(rank)
        self.group_size = int(group_size)
        self.num_grids = int(num_grids)
        self.register_buffer("qweight", torch.empty_like(module.weight))
        self.register_buffer("low_rank_a", torch.empty(module.in_features, 0, device=module.weight.device, dtype=module.weight.dtype))
        self.register_buffer("low_rank_b", torch.empty(0, module.out_features, device=module.weight.device, dtype=module.weight.dtype))
        self.register_buffer("smooth_scale", torch.ones(module.in_features, device=module.weight.device, dtype=torch.float32))
        self.register_buffer("calibrated", torch.tensor(False))
        self.bias = None if module.bias is None else nn.Parameter(module.bias.detach().clone())
        self._raw_weight = module.weight.detach().float().clone()

    @classmethod
    def from_float(cls, module, w_bit, a_bit, rank, alpha=0.5, group_size=64, num_grids=20):
        del alpha
        return cls(module, w_bit, a_bit, rank, group_size, num_grids).to(module.weight.device)

    def _quantized_output(self, inputs, weight, scale):
        migrated = inputs / scale
        quant_input, _ = symmetric_group_quant(
            migrated, self.a_bit, self.group_size, allow_unsigned=True
        )
        quant_weight, _ = symmetric_group_quant(
            weight * scale.unsqueeze(0), self.w_bit, self.group_size
        )
        return F.linear(quant_input, quant_weight)

    @torch.no_grad()
    def calibrate(self, inputs: torch.Tensor, candidates: int | None = None):
        x = inputs.reshape(-1, self.in_features).to(self.qweight.device, torch.float32)
        weight = self._raw_weight.to(x.device)
        reference = F.linear(x, weight)
        x_span = x.abs().amax(dim=0).clamp_min(1e-8)
        w_span = weight.abs().amax(dim=0).clamp_min(1e-8)
        grids = int(candidates or self.num_grids)
        choices = [index / grids for index in range(1, grids)]
        pairs = [(0.0, 0.0)] + [(alpha, 0.0) for alpha in choices] + [
            (alpha, 1.0 - alpha) for alpha in choices
        ]
        best_error = float("inf")
        best_scale = torch.ones_like(x_span)
        for alpha, beta in pairs:
            scale = x_span.pow(alpha).div(w_span.pow(beta)).clamp(1e-5, 1e5)
            output = self._quantized_output(x, weight, scale)
            error = F.mse_loss(output, reference).item()
            if error < best_error:
                best_error, best_scale = error, scale

        migrated_weight = weight * best_scale.unsqueeze(0)
        rank = min(self.rank, min(migrated_weight.shape))
        if rank > 0:
            u, singular, vh = torch.linalg.svd(migrated_weight.double(), full_matrices=False)
            low_b = (u[:, :rank] * singular[:rank]).transpose(0, 1).float()
            low_a = vh[:rank].transpose(0, 1).float()
            low_weight = low_b.transpose(0, 1) @ low_a.transpose(0, 1)
        else:
            low_a = migrated_weight.new_empty(self.in_features, 0)
            low_b = migrated_weight.new_empty(0, self.out_features)
            low_weight = torch.zeros_like(migrated_weight)
        residual = migrated_weight - low_weight
        qweight, _ = symmetric_group_quant(residual, self.w_bit, self.group_size)

        self.qweight.resize_as_(qweight).copy_(qweight.to(self.qweight.dtype))
        self.low_rank_a.resize_as_(low_a).copy_(low_a.to(self.low_rank_a.dtype))
        self.low_rank_b.resize_as_(low_b).copy_(low_b.to(self.low_rank_b.dtype))
        self.smooth_scale.copy_(best_scale)
        self.calibrated.fill_(True)
        del self._raw_weight

    def forward(self, tensor):
        if not bool(self.calibrated):
            weight = self._raw_weight.to(tensor.device, tensor.dtype)
            return F.linear(tensor, weight, self.bias)
        shape = tensor.shape
        migrated = tensor.reshape(-1, self.in_features) / self.smooth_scale.to(tensor.device, tensor.dtype)
        quantized, _ = symmetric_group_quant(
            migrated, self.a_bit, self.group_size, allow_unsigned=True
        )
        output = F.linear(quantized, self.qweight, self.bias)
        if self.low_rank_a.shape[1]:
            output = output + (migrated @ self.low_rank_a.to(migrated.dtype)) @ self.low_rank_b.to(migrated.dtype)
        return output.reshape(*shape[:-1], self.out_features)
