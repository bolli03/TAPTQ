"""TAPTQ adapters for VGGT-Omega and the community D4RT implementation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_THIRD_PARTY = _PROJECT_ROOT / "third_party"


def _prepend_import_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def _resize_pointmap(points: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(points.shape[-3:-1]) == size:
        return points
    batch, views = points.shape[:2]
    resized = F.interpolate(
        points.reshape(batch * views, *points.shape[2:]).permute(0, 3, 1, 2),
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.permute(0, 2, 3, 1).reshape(batch, views, *size, 3)


def _resize_confidence(confidence: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(confidence.shape[-2:]) == size:
        return confidence
    batch, views = confidence.shape[:2]
    resized = F.interpolate(
        confidence.reshape(batch * views, 1, *confidence.shape[-2:]),
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(batch, views, *size)


class VGGTOmegaPointmapAdapter(nn.Module):
    """Expose VGGT-Omega as the E1-compatible ``world_points`` interface."""

    def __init__(self, model: nn.Module, image_resolution: int = 512):
        super().__init__()
        self.aggregator = model.aggregator
        self.camera_head = model.camera_head
        self.dense_head = model.dense_head
        self.text_alignment_head = model.text_alignment_head
        self.image_resolution = int(image_resolution)

    @staticmethod
    def _backproject(depth: torch.Tensor, pose_encoding: torch.Tensor) -> torch.Tensor:
        from vggt_omega.utils.pose_enc import encoding_to_camera

        batch, views, height, width = depth.shape
        safe_pose = pose_encoding.clone()
        safe_pose[..., 7:9] = safe_pose[..., 7:9].abs().clamp(1e-3, torch.pi - 1e-3)
        extrinsics, intrinsics = encoding_to_camera(safe_pose, (height, width))
        ys, xs = torch.meshgrid(
            torch.arange(height, device=depth.device, dtype=depth.dtype),
            torch.arange(width, device=depth.device, dtype=depth.dtype),
            indexing="ij",
        )
        fx = intrinsics[..., 0, 0][..., None, None]
        fy = intrinsics[..., 1, 1][..., None, None]
        cx = intrinsics[..., 0, 2][..., None, None]
        cy = intrinsics[..., 1, 2][..., None, None]
        camera_points = torch.stack(
            ((xs - cx) * depth / fx, (ys - cy) * depth / fy, depth), dim=-1
        )
        rotation = extrinsics[..., :3, :3]
        translation = extrinsics[..., :3, 3]
        world_points = torch.einsum(
            "bvij,bvhwj->bvhwi", rotation.transpose(-1, -2), camera_points - translation[..., None, None, :]
        )
        return world_points

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if images.ndim != 5:
            raise ValueError(f"VGGT-Omega expects [B,V,3,H,W], got {tuple(images.shape)}")
        output_size = tuple(int(value) for value in images.shape[-2:])
        target_h = max(16, round(output_size[0] * self.image_resolution / output_size[1] / 16) * 16)
        target_w = self.image_resolution
        native = F.interpolate(
            images.flatten(0, 1),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).reshape(images.shape[0], images.shape[1], 3, target_h, target_w)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=native.device.type, dtype=amp_dtype, enabled=native.is_cuda):
            tokens, patch_start = self.aggregator(native)
        with torch.autocast(device_type=native.device.type, enabled=False):
            pose_encoding = self.camera_head(tokens, patch_token_start=patch_start)
            depth, confidence = self.dense_head(tokens, images=native, patch_token_start=patch_start)
        if depth.ndim == 5 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        points = self._backproject(depth.float(), pose_encoding.float())
        points = _resize_pointmap(points, output_size)
        confidence = _resize_confidence(confidence.float(), output_size)
        return {
            "world_points": points,
            "world_points_conf": confidence,
            "pose_enc": pose_encoding,
            "depth": depth,
        }


class ExplicitMultiheadAttention(nn.Module):
    """Multi-head attention with explicit Linear projections for TAPTQ wrapping."""

    def __init__(self, source: nn.MultiheadAttention, *, cross_attention: bool):
        super().__init__()
        if not source.batch_first or source.kdim != source.embed_dim or source.vdim != source.embed_dim:
            raise ValueError("OpenD4RT attention must use batch_first with equal q/k/v dimensions")
        self.embed_dim = source.embed_dim
        self.num_heads = source.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = float(source.dropout)
        self.cross_attention = bool(cross_attention)
        if self.cross_attention:
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=source.in_proj_bias is not None)
            self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=source.in_proj_bias is not None)
            self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=source.in_proj_bias is not None)
            with torch.no_grad():
                q_weight, k_weight, v_weight = source.in_proj_weight.chunk(3, dim=0)
                self.q_proj.weight.copy_(q_weight)
                self.k_proj.weight.copy_(k_weight)
                self.v_proj.weight.copy_(v_weight)
                if source.in_proj_bias is not None:
                    q_bias, k_bias, v_bias = source.in_proj_bias.chunk(3, dim=0)
                    self.q_proj.bias.copy_(q_bias)
                    self.k_proj.bias.copy_(k_bias)
                    self.v_proj.bias.copy_(v_bias)
        else:
            self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=source.in_proj_bias is not None)
            with torch.no_grad():
                self.qkv.weight.copy_(source.in_proj_weight)
                if source.in_proj_bias is not None:
                    self.qkv.bias.copy_(source.in_proj_bias)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=source.out_proj.bias is not None)
        with torch.no_grad():
            self.proj.weight.copy_(source.out_proj.weight)
            if source.out_proj.bias is not None:
                self.proj.bias.copy_(source.out_proj.bias)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = value.shape
        return value.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask=None,
        need_weights: bool = True,
        attn_mask=None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        if key_padding_mask is not None:
            raise NotImplementedError("OpenD4RT does not use key_padding_mask")
        if self.cross_attention:
            q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)
        else:
            if query is not key or key is not value:
                raise ValueError("Self-attention expects identical query/key/value tensors")
            q, k, v = self.qkv(query).chunk(3, dim=-1)
        dropout_p = self.dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(
            self._split_heads(q),
            self._split_heads(k),
            self._split_heads(v),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
        output = output.transpose(1, 2).contiguous().reshape(query.shape[0], query.shape[1], self.embed_dim)
        output = self.proj(output)
        if need_weights:
            return output, None
        return output, None


class ExplicitFeedForward(nn.Module):
    """Preserve OpenD4RT's MLP while exposing canonical fc1/fc2 names."""

    def __init__(self, source: nn.Sequential):
        super().__init__()
        self.fc1 = source[0]
        self.activation = source[1]
        self.dropout1 = source[2]
        self.fc2 = source[3]
        self.dropout2 = source[4]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dropout2(self.fc2(self.dropout1(self.activation(self.fc1(value)))))


def _make_open_d4rt_attention_explicit(model: nn.Module) -> None:
    for block in model.encoder.blocks:
        block.attn = ExplicitMultiheadAttention(block.attn, cross_attention=False)
        block.ff = ExplicitFeedForward(block.ff)
    for block in model.decoder.blocks:
        block.cross_attn = ExplicitMultiheadAttention(block.cross_attn, cross_attention=True)
        block.ff = ExplicitFeedForward(block.ff)


class OpenD4RTPointmapAdapter(nn.Module):
    """Expose the trained OpenD4RT checkpoint through the E1 point-map interface."""

    def __init__(self, model: nn.Module, image_size: int = 256, query_grid: int = 16, query_chunk: int = 4096):
        super().__init__()
        self.backbone = model
        self.image_size = int(image_size)
        self.query_grid = int(query_grid)
        self.query_chunk = int(query_chunk)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if images.ndim != 5:
            raise ValueError(f"OpenD4RT expects [B,T,3,H,W], got {tuple(images.shape)}")
        batch, frames = images.shape[:2]
        max_frames = int(self.backbone.query_embedder.max_frames)
        if frames > max_frames:
            raise ValueError(f"OpenD4RT checkpoint supports at most {max_frames} frames, got {frames}")
        output_size = tuple(int(value) for value in images.shape[-2:])
        native = F.interpolate(
            images.flatten(0, 1),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).reshape(batch, frames, 3, self.image_size, self.image_size)
        aspect_ratio = torch.full(
            (batch, 1), output_size[1] / max(output_size[0], 1), device=native.device, dtype=native.dtype
        )
        memory = self.backbone.encode_video(native, aspect_ratio=aspect_ratio)

        axis = torch.linspace(0, 1, self.query_grid, device=native.device, dtype=native.dtype)
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        uv_one = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
        uv = uv_one.repeat(frames, 1).unsqueeze(0).expand(batch, -1, -1)
        time = torch.arange(frames, device=native.device).repeat_interleave(self.query_grid**2)
        time = time.unsqueeze(0).expand(batch, -1)
        reference = torch.zeros_like(time)

        xyz_chunks, confidence_chunks = [], []
        for start in range(0, uv.shape[1], self.query_chunk):
            stop = min(start + self.query_chunk, uv.shape[1])
            query = {
                "u": uv[:, start:stop, 0],
                "v": uv[:, start:stop, 1],
                "t_src": time[:, start:stop],
                "t_tgt": time[:, start:stop],
                "t_cam": reference[:, start:stop],
            }
            output = self.backbone.decode_queries(native, query, memory)
            xyz_chunks.append(output["xyz_3d"])
            confidence_chunks.append(output["confidence"])
        points = torch.cat(xyz_chunks, dim=1).reshape(
            batch, frames, self.query_grid, self.query_grid, 3
        )
        confidence = torch.sigmoid(torch.cat(confidence_chunks, dim=1)).reshape(
            batch, frames, self.query_grid, self.query_grid
        )
        return {
            "world_points": _resize_pointmap(points, output_size),
            "world_points_conf": _resize_confidence(confidence, output_size),
        }


class D4RTPointmapAdapter(nn.Module):
    """Expose the legacy community D4RT query model as a compact point-map interface."""

    def __init__(self, model: nn.Module, image_size: int = 256, query_grid: int = 16):
        super().__init__()
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.image_size = int(image_size)
        self.query_grid = int(query_grid)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if images.ndim != 5:
            raise ValueError(f"D4RT expects [B,T,3,H,W], got {tuple(images.shape)}")
        batch, original_frames = images.shape[:2]
        expected_frames = int(getattr(self.encoder.patch_embed, "temporal_size", original_frames))
        if original_frames > expected_frames:
            raise ValueError(
                f"D4RT checkpoint supports at most {expected_frames} frames, got {original_frames}. "
                "Use a matching seq-id-map; sliding-window reconstruction is not implemented yet."
            )
        elif original_frames < expected_frames:
            padding = images[:, -1:].expand(-1, expected_frames - original_frames, -1, -1, -1)
            images = torch.cat((images, padding), dim=1)
        frames = images.shape[1]
        native = F.interpolate(
            images.flatten(0, 1),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).reshape(batch, frames, 3, self.image_size, self.image_size)
        video = native.permute(0, 1, 3, 4, 2).contiguous()
        features = self.encoder(video)
        output_frames = min(original_frames, expected_frames)
        axis = torch.linspace(0, 1, self.query_grid, device=video.device, dtype=video.dtype)
        grid_x, grid_y = torch.meshgrid(axis, axis, indexing="xy")
        coords = torch.stack((grid_x, grid_y), dim=-1).reshape(1, -1, 2)
        coords = coords.repeat(batch, output_frames, 1).reshape(batch, -1, 2)
        time = torch.arange(output_frames, device=video.device).repeat_interleave(self.query_grid**2)
        time = time.unsqueeze(0).expand(batch, -1)
        reference = torch.zeros_like(time)
        output = self.decoder(features, native, coords, time, time, reference)
        points = output["pos_3d"].reshape(batch, output_frames, self.query_grid, self.query_grid, 3)
        confidence = output["confidence"].reshape(
            batch, output_frames, self.query_grid, self.query_grid
        )
        output_size = tuple(int(value) for value in images.shape[-2:])
        points = _resize_pointmap(points, output_size)
        confidence = _resize_confidence(confidence, output_size)
        return {
            "world_points": points,
            "world_points_conf": confidence,
            **output,
        }


def load_vggt_omega(checkpoint: str, device: str | torch.device, image_resolution: int = 512):
    _prepend_import_path(_THIRD_PARTY / "vggt-omega")
    from vggt_omega.models import VGGTOmega

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"VGGT-Omega checkpoint not found: {checkpoint_path}. Request access at "
            "https://huggingface.co/facebook/VGGT-Omega"
        )
    model = VGGTOmega()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return VGGTOmegaPointmapAdapter(model, image_resolution=image_resolution).to(device).eval()


def load_open_d4rt(
    checkpoint: str,
    device: str | torch.device,
    *,
    image_size: int = 256,
    query_grid: int = 16,
):
    checkpoint_path = Path(checkpoint)
    config_path = Path(os.environ.get("D4RT_CONFIG_PATH", checkpoint_path.with_name("model.yaml")))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"OpenD4RT checkpoint not found: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"OpenD4RT model config not found: {config_path}")

    open_d4rt_root = _THIRD_PARTY / "open-d4rt"
    _prepend_import_path(open_d4rt_root)
    from src.core.config import load_yaml_config
    from src.model.builder import build_model

    cfg = load_yaml_config(config_path)
    model = build_model(cfg["model"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "module", "network", "net"):
            if isinstance(payload.get(key), dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict) or not payload or not all(torch.is_tensor(v) for v in payload.values()):
        raise ValueError(f"No OpenD4RT state_dict found in {checkpoint_path}")
    missing, unexpected = model.load_state_dict(payload, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"OpenD4RT checkpoint mismatch: {len(missing)} missing, {len(unexpected)} unexpected; "
            f"samples={missing[:3]} / {unexpected[:3]}"
        )
    _make_open_d4rt_attention_explicit(model)
    return OpenD4RTPointmapAdapter(
        model, image_size=image_size, query_grid=query_grid
    ).to(device).eval()


def load_d4rt(
    checkpoint: str | None,
    device: str | torch.device,
    *,
    variant: str = "base",
    image_size: int = 256,
    temporal_size: int = 8,
    query_grid: int = 16,
    allow_random_init: bool = False,
):
    if checkpoint and Path(checkpoint).name == "opend4rt.ckpt":
        return load_open_d4rt(
            checkpoint, device, image_size=image_size, query_grid=query_grid
        )

    _prepend_import_path(_THIRD_PARTY / "d4rt-pytorch")
    from models.d4rt import D4RT

    checkpoint_path = Path(checkpoint) if checkpoint else None
    checkpoint_payload = None
    checkpoint_args: dict = {}
    if checkpoint_path and checkpoint_path.is_file():
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_args = dict(checkpoint_payload.get("args", {})) if isinstance(checkpoint_payload, dict) else {}
    elif not allow_random_init:
        raise FileNotFoundError(
            "D4RT has no official public checkpoint. Pass a community checkpoint or set "
            "D4RT_ALLOW_RANDOM_INIT=1 for adapter/calibration smoke tests only."
        )

    state = checkpoint_payload.get("model_state_dict", checkpoint_payload) if checkpoint_payload is not None else None
    if state is not None and any(key.startswith("encoder.backbone.") for key in state):
        raise ValueError(
            "This TAPTQ adapter currently supports custom D4RTEncoder checkpoints only; "
            "VideoMAE-backed community checkpoints need a separate quantization scope."
        )

    model = D4RT(
        encoder_variant=str(checkpoint_args.get("encoder", variant)),
        img_size=int(checkpoint_args.get("img_size", image_size)),
        temporal_size=int(checkpoint_args.get("num_frames", temporal_size)),
        decoder_depth=int(checkpoint_args.get("decoder_depth", 6)),
        query_patch_size=int(checkpoint_args.get("patch_size", 9)),
        encoder_use_videomae=False,
        encoder_pretrained=False,
    )
    if state is not None:
        model.load_state_dict(state, strict=True)
    adapter = D4RTPointmapAdapter(model, image_size=image_size, query_grid=query_grid)
    adapter.random_init = checkpoint_payload is None
    return adapter.to(device).eval()


def load_foundation_model(name: str, checkpoint: str | None, device, **kwargs):
    if name == "vggt_omega":
        return load_vggt_omega(checkpoint or "", device, kwargs.get("image_resolution", 512))
    if name == "d4rt":
        allow_random = kwargs.get("allow_random_init", False) or os.environ.get("D4RT_ALLOW_RANDOM_INIT") == "1"
        return load_d4rt(
            checkpoint,
            device,
            variant=kwargs.get("variant", "base"),
            image_size=kwargs.get("image_resolution", 256),
            temporal_size=kwargs.get("temporal_size", 10),
            query_grid=kwargs.get("query_grid", 16),
            allow_random_init=allow_random,
        )
    raise ValueError(f"Unknown foundation model: {name}")
