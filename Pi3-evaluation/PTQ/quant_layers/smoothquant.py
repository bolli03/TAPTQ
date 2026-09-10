"""SmoothQuant fake-quant Linear for the unified VGGT baseline adapter."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline import BaselineQuantLinear


@torch.no_grad()
def smooth_scale(activation_absmax, weight, alpha=0.5, eps=1e-5):
    """Compute the official SmoothQuant per-input-channel migration scale."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"SmoothQuant alpha must be in [0, 1], got {alpha}")
    if eps <= 0:
        raise ValueError(f"SmoothQuant eps must be positive, got {eps}")
    activation_absmax = activation_absmax.float()
    weight_absmax = weight.detach().float().abs().amax(dim=0)
    inactive = (activation_absmax <= eps) | (weight_absmax <= eps)
    scale = activation_absmax.clamp_min(eps).pow(alpha)
    scale.div_(weight_absmax.clamp_min(eps).pow(1.0 - alpha))
    scale[inactive] = 1.0
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise RuntimeError("SmoothQuant produced an invalid migration scale")
    return scale


class SmoothQuantLinear(BaselineQuantLinear):
    """Linear supporting either folded or runtime SmoothQuant migration."""
    def __init__(self, in_features, out_features, bias=True, a_bit=8, runtime_smoothing=False):
        super().__init__(in_features, out_features, bias=bias, a_bit=a_bit)
        self.runtime_smoothing = bool(runtime_smoothing)
        self.register_buffer("smooth_scale", torch.ones(in_features))

    @classmethod
    def from_float(cls, module, a_bit, runtime_smoothing=False):
        result = cls(
            module.in_features, module.out_features, module.bias is not None,
            a_bit, runtime_smoothing,
        )
        result.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            result.bias.data.copy_(module.bias.data)
        return result.to(module.weight.device, module.weight.dtype)

    @torch.no_grad()
    def set_smooth_scale(self, scale):
        if scale.numel() != self.in_features:
            raise ValueError(
                f"SmoothQuant scale width mismatch: {scale.numel()} != {self.in_features}"
            )
        self.smooth_scale.copy_(scale.to(self.smooth_scale.device, torch.float32))

    def forward(self, tensor):
        if self.runtime_smoothing:
            tensor = tensor / self.smooth_scale.to(tensor.device, tensor.dtype)
        if self.input_quant_enabled:
            tensor = self.quantize_input(tensor)
        return F.linear(tensor, self.weight, self.bias)
