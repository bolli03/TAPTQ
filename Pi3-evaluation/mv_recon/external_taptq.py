"""TAPTQ calibration and module-level QwT for DUSt3R/MASt3R backbones.

The quantized scope is intentionally restricted to Transformer block linear
layers and excludes patch embedding, decoder bridge, and task heads:

- ``enc_blocks.*.{attn.qkv,attn.proj,mlp.fc1,mlp.fc2}``
- ``dec_blocks.*`` and ``dec_blocks2.*`` self-attention, cross-attention,
  and MLP projections.

Module-level compensation fits a low-rank residual correction for every
selected Attention/CrossAttention/MLP branch on the frozen ``DTU_train_8``
calibration set.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from PTQ.quant_layers.conv import MinMaxQuantConv2d
from PTQ.quant_layers.linear import MinMaxQuantLinear
from PTQ.quant_layers.matmul import MinMaxQuantMatMul
from PTQ.utils.net_wrap import wrap_modules_in_net
from PTQ.utils.quant_calib import (
    HessianQuantCalibrator,
    conv2d_forward_hook,
    grad_hook,
    linear_forward_hook,
    matmul_forward_hook,
)
from datasets.dtu import DTU
from mv_recon.external_baselines import evaluate_dataset, load_external_model
from mv_recon.taptq import cfg_modifier, init_config

LOGGER = logging.getLogger("external_taptq")
_QUANT_TYPES = (MinMaxQuantLinear, MinMaxQuantConv2d, MinMaxQuantMatMul)
_BACKBONE_SCOPE = ("enc_blocks", "dec_blocks", "dec_blocks2")
_EXPECTED_SCOPE_COUNTS = {"enc_blocks": 96, "dec_blocks": 96, "dec_blocks2": 96}
_BRANCH_LEAVES = ("attn", "cross_attn", "mlp")
_QUANT_ATTRS = ("w_interval", "w_qmax", "a_interval", "a_qmax")


def set_quant_mode(model: nn.Module, mode: str) -> None:
    for module in model.modules():
        if isinstance(module, _QUANT_TYPES):
            module.mode = mode


def apply_quant_scope(
    model: nn.Module,
    scopes: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> tuple[int, int]:
    """Enable only selected top-level scopes and disable matching sensitive paths."""
    enabled = disabled = 0
    allowed = set(scopes or _BACKBONE_SCOPE)
    excluded = tuple(exclude_patterns or ())
    for name, module in model.named_modules():
        if not isinstance(module, _QUANT_TYPES):
            continue
        top_level = name.split(".", 1)[0]
        active = top_level in allowed and not any(pattern in name for pattern in excluded)
        module.mode = "quant_forward" if active else "raw"
        enabled += int(active)
        disabled += int(not active)
    return enabled, disabled


class PairPointmapAdapter(nn.Module):
    """Expose a multi-view tensor interface over an asymmetric pair model.

    The first view is the anchor. Every other view is paired with it so all
    returned point maps share the anchor coordinate system.
    """

    def __init__(self, backbone: nn.Module, image_size: int = 512):
        super().__init__()
        self.backbone = backbone
        self.image_size = int(image_size)

    @staticmethod
    def _resize_points(points: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if tuple(points.shape[1:3]) == size:
            return points
        return F.interpolate(
            points.permute(0, 3, 1, 2), size=size, mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)

    @staticmethod
    def _resize_conf(conf: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if tuple(conf.shape[1:3]) == size:
            return conf
        return F.interpolate(conf[:, None], size=size, mode="bilinear", align_corners=False)[:, 0]

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim == 5:
            if images.shape[0] != 1:
                raise ValueError(f"PairPointmapAdapter only supports batch=1, got {images.shape}")
            images = images[0]
        if images.ndim != 4 or images.shape[0] < 2:
            raise ValueError(f"Expected [views,3,H,W] with views>=2, got {images.shape}")

        output_size = tuple(int(x) for x in images.shape[-2:])
        target_w = self.image_size
        target_h = max(16, round(output_size[0] * target_w / output_size[1] / 16) * 16)
        native = F.interpolate(images, size=(target_h, target_w), mode="bilinear", align_corners=False, antialias=True)
        native = native.mul(2).sub(1)

        pair_count = native.shape[0] - 1
        anchor = native[:1].expand(pair_count, -1, -1, -1).contiguous()
        others = native[1:]
        true_shape = torch.tensor([target_h, target_w], device=native.device).repeat(pair_count, 1)
        view1 = {
            "img": anchor,
            "true_shape": true_shape,
            "idx": list(range(pair_count)),
            "instance": [f"anchor-{i}" for i in range(pair_count)],
        }
        view2 = {
            "img": others,
            "true_shape": true_shape.clone(),
            "idx": list(range(pair_count)),
            "instance": [f"other-{i}" for i in range(pair_count)],
        }
        pred1, pred2 = self.backbone(view1, view2)
        anchor_points = pred1["pts3d"][:1]
        other_points = pred2["pts3d_in_other_view"]
        points = torch.cat([anchor_points, other_points], dim=0)
        points = self._resize_points(points, output_size).unsqueeze(0)

        conf1 = pred1.get("conf")
        conf2 = pred2.get("conf")
        if conf1 is None or conf2 is None:
            conf = torch.ones(points.shape[:-1], device=points.device, dtype=points.dtype)
        else:
            conf = torch.cat([conf1[:1], conf2], dim=0)
            conf = self._resize_conf(conf, output_size).unsqueeze(0)
        return {"world_points": points, "world_points_conf": conf}


class LowRankCorrection(nn.Module):
    def __init__(self, A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.A.to(dtype=x.dtype)
        B = self.B.to(dtype=x.dtype)
        bias = self.bias.to(dtype=x.dtype)
        return x @ A @ B + bias


def _correction_hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
    correction = getattr(module, "taptq_qwt", None)
    if correction is None or not isinstance(output, torch.Tensor):
        return output
    return output + correction(inputs[0])


def install_correction(module: nn.Module, correction: LowRankCorrection) -> None:
    handle = getattr(module, "_taptq_qwt_handle", None)
    if handle is not None:
        handle.remove()
    module.add_module("taptq_qwt", correction)
    module._taptq_qwt_handle = module.register_forward_hook(_correction_hook)


def disable_qwt_corrections(model: nn.Module) -> int:
    """Remove QwT forward hooks while retaining calibrated quantization state."""
    disabled = 0
    for module in model.modules():
        handle = getattr(module, "_taptq_qwt_handle", None)
        if handle is not None:
            handle.remove()
            module._taptq_qwt_handle = None
            disabled += 1
    return disabled


def _sample_rows(tensor: torch.Tensor, max_rows: int, seed: int) -> torch.Tensor:
    rows = tensor.detach().float().reshape(-1, tensor.shape[-1]).cpu()
    if rows.shape[0] <= max_rows:
        return rows
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randperm(rows.shape[0], generator=generator)[:max_rows]
    return rows[ids]


def tail_relative_error(
    raw_out: torch.Tensor,
    quant_out: torch.Tensor,
    rho: float,
    eps: float = 1e-8,
) -> float:
    """VGGT-compatible TRE on the top-rho magnitude elements of the raw output."""
    if not 0 < rho <= 1:
        raise ValueError(f"rho must be in (0, 1], got {rho}")
    raw = raw_out.reshape(-1)
    quant = quant_out.reshape(-1)
    count = max(1, int(rho * raw.numel()))
    indices = torch.topk(raw.abs(), count, largest=True, sorted=False).indices
    numerator = (raw[indices] - quant[indices]).pow(2).sum()
    denominator = raw[indices].pow(2).sum().clamp_min(eps)
    return float((numerator / denominator).item())


def _ridge_low_rank(
    x: torch.Tensor,
    target: torch.Tensor,
    rank: int,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = target.mean(dim=0, keepdim=True)
    xc = x - x_mean
    yc = target - y_mean
    eye = torch.eye(xc.shape[1], dtype=xc.dtype)
    lhs = xc.T @ xc + ridge * eye
    rhs = xc.T @ yc
    try:
        weight = torch.linalg.solve(lhs, rhs)
    except torch.linalg.LinAlgError:
        weight = torch.linalg.lstsq(lhs, rhs).solution
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    use_rank = max(1, min(int(rank), len(S)))
    sqrt_s = S[:use_rank].clamp_min(0).sqrt()
    A = U[:, :use_rank] * sqrt_s[None]
    B = sqrt_s[:, None] * Vh[:use_rank]
    low_rank_weight = A @ B
    bias = (y_mean - x_mean @ low_rank_weight).squeeze(0)
    pred = x @ low_rank_weight + bias
    residual = (target - pred).pow(2).mean().sqrt()
    denom = target.pow(2).mean().sqrt().clamp_min(1e-12)
    fit_error = float((residual / denom).item())
    return A, B, bias, fit_error


def _branch_modules(backbone: nn.Module) -> list[tuple[str, nn.Module]]:
    selected = []
    for block_group in _BACKBONE_SCOPE:
        blocks = getattr(backbone, block_group)
        for index, block in enumerate(blocks):
            for leaf in _BRANCH_LEAVES:
                if hasattr(block, leaf):
                    selected.append((f"{block_group}.{index}.{leaf}", getattr(block, leaf)))
    return selected


def _capture_branch(
    adapter: PairPointmapAdapter,
    module: nn.Module,
    dataset: DTU,
    seq_id_map: dict[str, list[int]],
    quant_mode: str,
    max_rows: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    set_quant_mode(adapter, quant_mode)
    captured_inputs: list[torch.Tensor] = []
    captured_outputs: list[torch.Tensor] = []

    rows_per_sequence = max(1, max_rows // max(1, len(seq_id_map)))

    def hook(_module, inputs, output):
        sample_seed = seed + len(captured_inputs)
        captured_inputs.append(_sample_rows(inputs[0], rows_per_sequence, sample_seed))
        captured_outputs.append(_sample_rows(output, rows_per_sequence, sample_seed))

    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for seq_name, ids in seq_id_map.items():
                data = dataset.get_data(sequence_name=seq_name, ids=ids)
                adapter(data["images"].to(next(adapter.parameters()).device))
    finally:
        handle.remove()
    if not captured_inputs:
        raise RuntimeError(f"No branch tensors captured from {module.__class__.__name__}")
    return torch.cat(captured_inputs), torch.cat(captured_outputs)


def compensate_modulewise(
    adapter: PairPointmapAdapter,
    dataset: DTU,
    seq_id_map: dict[str, list[int]],
    rank: int = 16,
    ridge: float = 1e-4,
    rho: float = 0.01,
    tau: float = 0.007,
    fit_error_max: float = 0.8,
    max_rows: int = 32768,
    seed: int = 42,
    max_modules: int | None = None,
) -> list[dict[str, Any]]:
    """Fit gated low-rank corrections without contaminating later branch captures."""
    records = []
    pending: list[tuple[nn.Module, LowRankCorrection]] = []
    disable_qwt_corrections(adapter.backbone)
    branches = _branch_modules(adapter.backbone)
    if max_modules is not None:
        branches = branches[:max_modules]
    for index, (name, module) in enumerate(branches):
        sample_seed = seed + index
        _raw_x, raw_out = _capture_branch(
            adapter, module, dataset, seq_id_map, "raw", max_rows, sample_seed
        )
        quant_x, quant_out = _capture_branch(
            adapter, module, dataset, seq_id_map, "quant_forward", max_rows, sample_seed
        )
        rows = min(len(raw_out), len(quant_out), len(quant_x))
        x = quant_x[:rows]
        raw_out = raw_out[:rows]
        quant_out = quant_out[:rows]
        target = raw_out - quant_out
        score = tail_relative_error(raw_out, quant_out, rho)
        candidate = bool(np.isfinite(score) and score >= tau)
        record = {
            "name": name, "score": score, "rho": float(rho), "tau": float(tau),
            "selected": False, "rank": int(rank),
        }
        if candidate:
            A, B, bias, fit_error = _ridge_low_rank(x, target, rank, ridge)
            record["fit_error"] = fit_error
            record["selected"] = bool(np.isfinite(fit_error) and fit_error <= fit_error_max)
            if record["selected"]:
                correction = LowRankCorrection(
                    A.to(next(module.parameters()).device),
                    B.to(next(module.parameters()).device),
                    bias.to(next(module.parameters()).device),
                )
                pending.append((module, correction))
        records.append(record)
        LOGGER.info(
            "QwT %s TRE=%.6f rho=%.4f candidate=%s selected=%s fit_error=%s",
            name, score, rho, candidate, record["selected"], record.get("fit_error"),
        )
    for module, correction in pending:
        install_correction(module, correction)
    set_quant_mode(adapter, "quant_forward")
    return records


def _quant_records(model: nn.Module) -> list[dict[str, Any]]:
    records = []
    for name, module in model.named_modules():
        if not isinstance(module, _QUANT_TYPES):
            continue
        records.append({
            "name": name,
            "mode": module.mode,
            **{key: getattr(module, key, None) for key in _QUANT_ATTRS},
        })
    return records


def save_external_checkpoint(
    backbone: nn.Module,
    path: str,
    model_name: str,
    quant_cfg: Any,
    qwt_records: list[dict[str, Any]],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    qwt_meta = []
    for name, module in backbone.named_modules():
        correction = getattr(module, "taptq_qwt", None)
        if correction is not None:
            qwt_meta.append({"name": name, "rank": correction.A.shape[1]})
    torch.save({
        "version": 1,
        "model_name": model_name,
        "bit": list(quant_cfg.bit),
        "scope": list(_BACKBONE_SCOPE),
        "state_dict": backbone.state_dict(),
        "quant_records": _quant_records(backbone),
        "qwt_meta": qwt_meta,
        "qwt_records": qwt_records,
    }, path)
    LOGGER.info("Saved %s checkpoint to %s", model_name, path)


def _module_by_name(root_module: nn.Module, name: str) -> nn.Module:
    module = root_module
    for part in name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def load_external_checkpoint(backbone: nn.Module, path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for item in payload.get("qwt_meta", []):
        module = _module_by_name(backbone, item["name"])
        dim = next(module.parameters()).shape[1]
        rank = int(item["rank"])
        correction = LowRankCorrection(
            torch.zeros(dim, rank), torch.zeros(rank, dim), torch.zeros(dim)
        ).to(next(module.parameters()).device)
        install_correction(module, correction)
    missing, unexpected = backbone.load_state_dict(payload["state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:8]}")
    if missing:
        LOGGER.warning("Missing checkpoint keys: %s", missing[:8])
    modules = dict(backbone.named_modules())
    for item in payload.get("quant_records", []):
        module = modules[item["name"]]
        module.mode = item.get("mode", "quant_forward")
        for key in _QUANT_ATTRS:
            value = item.get(key)
            if value is not None:
                setattr(module, key, value.to(next(module.parameters()).device) if torch.is_tensor(value) else value)
        module.calibrated = True
    return payload


def build_quantized_model(args) -> tuple[nn.Module, PairPointmapAdapter, Any, dict[str, nn.Module]]:
    backbone = load_external_model(args.model, args.weights, args.device, args.repo_root)
    modifier = cfg_modifier(
        linear_ptq_setting=(1, 1, 1), metric=args.metric, bit_setting=tuple(args.bit)
    )
    quant_cfg = modifier(init_config(args.quant_config))
    quant_cfg.linear_channelwise = bool(args.channelwise)
    quant_cfg.ptqsl_linear_kwargs["search_mode"] = args.search_mode
    wrapped = wrap_modules_in_net(
        backbone,
        quant_cfg,
        quantize_scope=_BACKBONE_SCOPE,
    )
    counts = {scope: sum(name.startswith(scope + ".") for name in wrapped) for scope in _BACKBONE_SCOPE}
    if counts != _EXPECTED_SCOPE_COUNTS:
        raise RuntimeError(f"Unexpected {args.model} Transformer scope: {counts}")
    LOGGER.info("Wrapped external Transformer linear scope: %s", counts)
    return backbone, PairPointmapAdapter(backbone, args.image_size).to(args.device), quant_cfg, wrapped


def calibration_data(args) -> tuple[DTU, dict[str, list[int]]]:
    dataset = DTU(
        DTU_DIR=args.data_root,
        split=args.data_split,
        load_img_size=args.load_img_size,
        cache_file=args.cache_file,
    )
    with open(args.seq_id_map, "r", encoding="utf-8") as handle:
        seq_id_map = json.load(handle)
    return dataset, seq_id_map


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("dust3r", "mast3r"), required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--mode", choices=("smoke", "calib", "calib_compensate", "compensate_test", "test"), default="calib_compensate")
    p.add_argument("--checkpoint")
    p.add_argument("--save-checkpoint")
    p.add_argument("--device", default="cuda")
    p.add_argument("--bit", nargs=2, type=int, default=(4, 8), metavar=("W", "A"))
    p.add_argument("--search-mode", choices=("ternary", "exhaustive"), default="ternary")
    p.add_argument("--channelwise", action="store_true")
    p.add_argument("--metric", default="hessian")
    p.add_argument("--quant-config", default="PTQ4ViT")
    p.add_argument("--data-root", default="data/dtu_8")
    p.add_argument("--cache-file", default="data/dataset_cache/dtu_mv_recon_cache_8.npy")
    p.add_argument("--seq-id-map", default="datasets/seq-id-maps/DTUTrain_8_mv-recon_seq-id-map-kf5.json")
    p.add_argument("--load-img-size", type=int, default=518)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--rho", type=float, default=0.01)
    p.add_argument("--tau", type=float, default=0.007)
    p.add_argument("--fit-error-max", type=float, default=0.8)
    p.add_argument("--max-qwt-rows", type=int, default=32768)
    p.add_argument("--max-calib-modules", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--max-qwt-modules", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--disable-qwt", action="store_true", help="Evaluate calibrated quantization without QwT hooks")
    p.add_argument("--quant-scope", nargs="+", choices=_BACKBONE_SCOPE, default=None)
    p.add_argument("--exclude-quant-pattern", action="append", default=[])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-data-root")
    p.add_argument("--eval-seq-id-map")
    p.add_argument("--eval-output-dir")
    return p


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    backbone, adapter, quant_cfg, wrapped = build_quantized_model(args)
    dataset, seq_id_map = calibration_data(args)
    if args.mode == "smoke":
        first_sequence, ids = next(iter(seq_id_map.items()))
        set_quant_mode(adapter, "raw")
        with torch.no_grad():
            output = adapter(dataset.get_data(sequence_name=first_sequence, ids=ids)["images"].to(args.device))
        points = output["world_points"]
        if not torch.isfinite(points).all():
            raise RuntimeError("External TAPTQ smoke forward produced non-finite point maps")
        LOGGER.info("Smoke forward passed: model=%s shape=%s", args.model, tuple(points.shape))
        return
    if not args.checkpoint:
        raise ValueError(f"--checkpoint is required for mode={args.mode}")

    qwt_records: list[dict[str, Any]] = []
    if args.mode in ("calib", "calib_compensate"):
        calibration_modules = wrapped
        if args.max_calib_modules is not None:
            calibration_modules = dict(list(wrapped.items())[:args.max_calib_modules])
        calibrator = HessianQuantCalibrator(
            adapter, calibration_modules, dataset, seq_id_map,
            sequential=True, batch_size=1, device=args.device, logger=LOGGER,
        )
        calibrator.batching_quant_calib()
        set_quant_mode(adapter, "quant_forward")
        if args.mode == "calib_compensate":
            qwt_records = compensate_modulewise(
                adapter, dataset, seq_id_map, rank=args.rank, ridge=args.ridge,
                rho=args.rho, tau=args.tau, fit_error_max=args.fit_error_max,
                max_rows=args.max_qwt_rows, seed=args.seed,
                max_modules=args.max_qwt_modules,
            )
        save_external_checkpoint(backbone, args.checkpoint, args.model, quant_cfg, qwt_records)
    else:
        load_external_checkpoint(backbone, args.checkpoint)
        set_quant_mode(adapter, "quant_forward")
        if args.mode == "compensate_test":
            disabled = disable_qwt_corrections(backbone)
            LOGGER.info("Removed %d existing QwT hooks before rho/tau compensation", disabled)
            qwt_records = compensate_modulewise(
                adapter, dataset, seq_id_map, rank=args.rank, ridge=args.ridge,
                rho=args.rho, tau=args.tau, fit_error_max=args.fit_error_max,
                max_rows=args.max_qwt_rows, seed=args.seed,
                max_modules=args.max_qwt_modules,
            )
            save_path = args.save_checkpoint or args.checkpoint
            save_external_checkpoint(backbone, save_path, args.model, quant_cfg, qwt_records)
        if args.disable_qwt:
            disabled = disable_qwt_corrections(backbone)
            LOGGER.info("Disabled %d QwT correction hooks for quant-only evaluation", disabled)
        if args.quant_scope is not None or args.exclude_quant_pattern:
            enabled, disabled = apply_quant_scope(
                backbone, args.quant_scope, args.exclude_quant_pattern,
            )
            LOGGER.info("Quant scope diagnostic: enabled=%d disabled=%d", enabled, disabled)

    if args.eval_data_root and args.eval_seq_id_map and args.eval_output_dir:
        from datasets.sevenscenes import SevenScenes
        dataset = SevenScenes(
            SEVENSCENES_DIR=args.eval_data_root,
            split="test",
            load_img_size=args.load_img_size,
            cache_file=os.path.join(args.eval_output_dir, "7scenes_cache.npy"),
        )
        evaluate_dataset(
            dataset, args.eval_seq_id_map, backbone, args.device,
            "7scenes", args.eval_output_dir, image_size=args.image_size,
        )


if __name__ == "__main__":
    main()
