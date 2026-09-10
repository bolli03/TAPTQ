"""Memory-bounded Hessian calibration for asymmetric pairwise 3D models."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from PTQ.quant_layers.conv import MinMaxQuantConv2d
from PTQ.quant_layers.linear import MinMaxQuantLinear
from PTQ.quant_layers.matmul import MinMaxQuantMatMul
from PTQ.utils.quant_calib import (
    conv2d_forward_hook,
    grad_hook,
    linear_forward_hook,
    matmul_forward_hook,
)


class PairHessianQuantCalibrator:
    """Calibrate pair models without materializing all anchor pairs at once."""

    def __init__(
        self,
        net,
        wrapped_modules,
        calib_loader,
        seq_id_map,
        device="cuda",
        logger=None,
        pair_chunk_size=3,
    ):
        self.net = net
        self.wrapped_modules = wrapped_modules
        self.calib_loader = calib_loader
        self.seq_id_map = seq_id_map
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.pair_chunk_size = max(1, int(pair_chunk_size))

    @staticmethod
    def _normalized_point_loss(pred, target, mask):
        mask = mask.bool()
        weight = mask[..., None].to(pred.dtype)
        count = weight.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1)
        pred_center = (pred * weight).sum(dim=(1, 2, 3), keepdim=True) / count
        target_center = (target * weight).sum(dim=(1, 2, 3), keepdim=True) / count
        pred = pred - pred_center
        target = target - target_center
        pred_scale = ((pred.square() * weight).sum(dim=(1, 2, 3), keepdim=True) / count).sqrt().clamp_min(1e-6)
        target_scale = ((target.square() * weight).sum(dim=(1, 2, 3), keepdim=True) / count).sqrt().clamp_min(1e-6)
        diff = pred / pred_scale - target / target_scale
        return (diff.square() * weight).sum() / weight.sum().clamp_min(1)

    @staticmethod
    def _hooks(module):
        hooks = []
        if isinstance(module, MinMaxQuantLinear):
            hooks.append(module.register_forward_hook(linear_forward_hook))
        elif isinstance(module, MinMaxQuantConv2d):
            hooks.append(module.register_forward_hook(conv2d_forward_hook))
        elif isinstance(module, MinMaxQuantMatMul):
            hooks.append(module.register_forward_hook(matmul_forward_hook))
        if hasattr(module, "metric"):
            hooks.append(module.register_full_backward_hook(grad_hook))
        return hooks

    def _iter_pair_chunks(self, data: dict[str, Any]):
        images = data["images"]
        points = torch.from_numpy(np.asarray(data["pointclouds"])).to(self.device)
        mask = torch.from_numpy(np.asarray(data["valid_mask"])).to(self.device)
        for start in range(1, len(images), self.pair_chunk_size):
            other_ids = list(range(start, min(len(images), start + self.pair_chunk_size)))
            ids = [0, *other_ids]
            yield images[ids].to(self.device), points[ids][None], mask[ids][None]

    @staticmethod
    def _finalize_cache(module):
        if isinstance(module, MinMaxQuantLinear):
            module.raw_input = torch.cat(module.raw_input, dim=0)
            module.raw_out = torch.cat(module.raw_out, dim=0)
        elif isinstance(module, MinMaxQuantConv2d):
            module.raw_input = torch.cat(module.raw_input, dim=0)
            module.raw_out = torch.cat(module.raw_out, dim=0)
        elif isinstance(module, MinMaxQuantMatMul):
            module.raw_input = [torch.cat(values, dim=0) for values in module.raw_input]
            module.raw_out = torch.cat(module.raw_out, dim=0)
        if hasattr(module, "metric"):
            module.raw_grad = torch.cat(module.raw_grad, dim=0)

    @staticmethod
    def _calibration_step2(module):
        with torch.no_grad():
            module.calibration_step2()
        for key in ("raw_input", "raw_out", "raw_grad"):
            if hasattr(module, key):
                setattr(module, key, None)
        torch.cuda.empty_cache()

    def batching_quant_calib(self):
        self.logger.info("Pair Hessian calibration over %d modules", len(self.wrapped_modules))
        for module_index, (name, module) in enumerate(self.wrapped_modules.items(), start=1):
            hooks = self._hooks(module)
            try:
                for seq_name, ids in self.seq_id_map.items():
                    data = self.calib_loader.get_data(sequence_name=seq_name, ids=ids)
                    for images, gt_points, valid_mask in self._iter_pair_chunks(data):
                        self.net.zero_grad(set_to_none=True)
                        prediction = self.net(images)
                        pred_points = prediction["world_points"]
                        if pred_points.shape != gt_points.shape:
                            raise RuntimeError(
                                f"Pair calibration shape mismatch: pred={pred_points.shape}, gt={gt_points.shape}"
                            )
                        loss = self._normalized_point_loss(pred_points, gt_points, valid_mask)
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Non-finite pair calibration loss for {name}")
                        loss.backward()
                        del prediction, pred_points, loss, images, gt_points, valid_mask
                        torch.cuda.empty_cache()
                self._finalize_cache(module)
            finally:
                for hook in hooks:
                    hook.remove()
            self._calibration_step2(module)
            module.mode = "raw"
            self.logger.info(
                "Pair Hessian calibrated [%d/%d] %s",
                module_index,
                len(self.wrapped_modules),
                name,
            )
        for module in self.wrapped_modules.values():
            module.mode = "quant_forward"
