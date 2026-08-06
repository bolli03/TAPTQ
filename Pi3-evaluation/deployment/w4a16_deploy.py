"""
W4A16 部署加载器 (VGGT + INT4 权重 + 可选 torchao 打包 + QwT 补偿).

依据 deployment/VGGT_W4A16_Deploy_Guide.md:
  1) 从 int4_weights.pt 反量化注入 Linear
  2) 可选: torchao Int4WeightOnlyConfig 打包
  3) 挂载 QwT 并对 ViT Block 做 forward monkey-patch
  4) 加载 non_quant_params.pt

仅依赖 Pi3-evaluation 根目录在 PYTHONPATH (与 build_and_save_quant_model 相同, 使用 rootutils).

用法:
  python deployment/w4a16_deploy.py --model ... --deploy-dir ... --dummy
  python deployment/w4a16_deploy.py --compare-fp --warmup 5 --iters 20 --save-json report.json
  python deployment/w4a16_deploy.py --compare-fp --num-views 50   # 默认只测主干 aggregator（延时/显存/参数体积）
  python deployment/w4a16_deploy.py --compare-fp --full-model      # 整网 forward + 各 head 输出对比
  （--compare-fp：默认仅 ``model.aggregator``；与全精度对比、测延时/峰值显存/模型大小）
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import tempfile
import time
import types
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn

try:
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    from torchao.quantization.quantize_.workflows import Int4PackingFormat
except ImportError:
    quantize_ = None  # type: ignore[misc, assignment]
    Int4WeightOnlyConfig = None  # type: ignore[misc, assignment]
    Int4PackingFormat = None  # type: ignore[misc, assignment]

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from PTQ.vggt.models.vggt import VGGT  # noqa: E402

# VGGT 中 ``self.aggregator`` 为 ViT 主干，各 *\_head 为任务头。
VGGT_BACKBONE_PREFIX = "aggregator"


def _is_backbone_state_key(key: str, prefix: str = VGGT_BACKBONE_PREFIX) -> bool:
    return key == prefix or key.startswith(prefix + ".")


def _required_deploy_files(deploy_dir: str, require_qwt: bool = True) -> Tuple[str, str, str, str]:
    p_int4 = os.path.join(deploy_dir, "int4_weights.pt")
    p_qwt = os.path.join(deploy_dir, "qwt_compensation.pt")
    p_nq = os.path.join(deploy_dir, "non_quant_params.pt")
    p_cfg = os.path.join(deploy_dir, "deploy_config.json")
    need = [p_int4, p_nq]
    if require_qwt:
        need.append(p_qwt)
    missing = [p for p in need if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "部署目录缺少权重文件:\n"
            + "\n".join(missing)
            + "\n请先用 build_and_save_quant_model.py 生成 int4_weights.pt / non_quant_params.pt"
            + (" / qwt_compensation.pt" if require_qwt else "")
        )
    return p_int4, p_qwt, p_nq, p_cfg


def saved_int4_name_to_linear_fqn(saved_name: str) -> str:
    """
    PTQ 保存的 named_modules 名可能带 ``.block``（如 ``...attn.qkv.block``），
    而 VGGT 原生 ``nn.Linear`` 在 ``...attn.qkv``；部署注入时统一到后者。
    """
    if saved_name.endswith(".block"):
        return saved_name[: -len(".block")]
    return saved_name


def broadcast_w_interval(int_w: torch.Tensor, w_interval: torch.Tensor) -> torch.Tensor:
    """与 build_and_save_quant_model.quantize_weight_to_int 对称的 scale 广播。"""
    w_interval = w_interval.float().squeeze()
    weight = int_w.float()
    if w_interval.dim() == 0:
        return weight * w_interval
    if w_interval.numel() == weight.shape[0]:
        w_b = w_interval.view(-1, *([1] * (weight.dim() - 1)))
        return weight * w_b
    if w_interval.numel() < weight.shape[0] and weight.shape[0] % w_interval.numel() == 0:
        repeat_factor = weight.shape[0] // w_interval.numel()
        w_b = w_interval.repeat_interleave(repeat_factor)
        w_b = w_b.view(-1, *([1] * (weight.dim() - 1)))
        return weight * w_b
    w_b = w_interval.view(-1, *([1] * (weight.dim() - 1)))
    return weight * w_b


def dequantize_entry_to_weight(entry: Dict[str, Any]) -> torch.Tensor:
    int_w = entry["int_weight"]
    return broadcast_w_interval(int_w, entry["w_interval"])


def get_submodule(model: nn.Module, path: str) -> Any:
    obj: Any = model
    for part in path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def set_submodule(model: nn.Module, path: str, value: Any) -> None:
    parts = path.split(".")
    parent = get_submodule(model, ".".join(parts[:-1]))
    setattr(parent, parts[-1], value)


def inject_int4_weights(model: nn.Module, int4_data: List[Dict[str, Any]], dtype: torch.dtype) -> int:
    int4_map: Dict[str, Dict[str, Any]] = {}
    for e in int4_data:
        int4_map[saved_int4_name_to_linear_fqn(e["name"])] = e
    n = 0
    for name, module in model.named_modules():
        if name not in int4_map:
            continue
        entry = int4_map[name]
        if not isinstance(module, nn.Linear):
            continue
        w_float = dequantize_entry_to_weight(entry).to(dtype=dtype)
        if module.weight.shape != w_float.shape:
            raise RuntimeError(f"形状不一致 {name}: module {module.weight.shape} vs int4 {w_float.shape}")
        module.weight.data.copy_(w_float)
        bias = entry.get("bias")
        if bias is not None:
            if module.bias is None:
                raise RuntimeError(f"{name} 保存了 bias 但当前 Linear 无 bias")
            module.bias.data.copy_(bias.to(dtype=dtype))
        n += 1
    if n != len(int4_data):
        raise RuntimeError(
            f"注入层数 {n} 与 int4 条目数 {len(int4_data)} 不一致 (检查命名、.block 映射或模块类型)"
        )
    return n


def aggregator_linear_filter(module: nn.Module, fqn: str) -> bool:
    return isinstance(module, nn.Linear) and "aggregator" in fqn


def _cast_filtered_linear_weights(model: nn.Module, dtype: torch.dtype, filter_fn: Callable[..., bool]) -> None:
    for name, m in model.named_modules():
        if filter_fn(m, name) and getattr(m, "weight", None) is not None:
            m.weight.data = m.weight.data.to(dtype)


def _cuda_tile_int4_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 8


TorchaoInt4Packing = Literal["auto", "plain", "tile"]


def apply_torchao_int4(
    model: nn.Module,
    group_size: int = 32,
    int4_packing: TorchaoInt4Packing = "auto",
) -> Tuple[bool, str]:
    """
    尝试 torchao INT4 weight-only（前向走 INT4 权重 GEMM，激活仍为 FP）。

    **为什么常见日志里先提 PLAIN 再提 TILE？**
    ``Int4PackingFormat.PLAIN`` 在 torchao 里往往走 ``mslk`` 提供的 CUDA kernel；未安装或与当前
    PyTorch/CUDA 主版本匹配的 ``mslk`` wheel 时，``quantize_`` 会报错。此时在 **auto** 模式下会再试
    ``TILE_PACKED_TO_4D``（tinygemm 等），**仍是对 INT4 打包权重做矩阵乘**，不是改回「全 FP 权重
    matmul」。将部分 Linear 的 ``weight`` 转为 ``bfloat16`` 仅发生在 **打包/量化阶段**，以满足
    该 packing 的 API；推理时仍是 INT4 权重核。

    Args:
        int4_packing:
            - ``auto``: 先 PLAIN，失败且在 CUDA 且 sm>=8 时再 TILE。
            - ``plain``: 仅 PLAIN；失败则抛出（需匹配环境的 ``mslk``）。
            - ``tile``: 仅 TILE_PACKED_TO_4D；需 CUDA sm>=8；失败则抛出。

    Returns:
        (success, packing_mode)  packing_mode 为 ``plain`` | ``tile_packed`` | ``none``
    """
    if int4_packing not in ("auto", "plain", "tile"):
        raise ValueError(f"int4_packing must be auto|plain|tile, got {int4_packing!r}")

    if quantize_ is None or Int4WeightOnlyConfig is None or Int4PackingFormat is None:
        print("Warning: torchao 未安装, 跳过 INT4 打包; 使用反量化权重推理 (无 INT4 GEMM 加速)。")
        return False, "none"

    e_plain: Optional[Exception] = None

    if int4_packing in ("auto", "plain"):
        cfg_plain = Int4WeightOnlyConfig(group_size=group_size)
        try:
            quantize_(model, cfg_plain, filter_fn=aggregator_linear_filter)
            print("torchao: 已使用 Int4PackingFormat.PLAIN（INT4 权重 GEMM，通常经 mslk）。")
            return True, "plain"
        except Exception as ex:
            e_plain = ex
            if int4_packing == "plain":
                raise RuntimeError(
                    "torchao PLAIN（Int4PackingFormat 默认）失败，且 int4_packing='plain' 禁止回退。\n"
                    "PLAIN 常依赖与当前 CUDA 版本匹配的 mslk；请安装对应 wheel，或改用 int4_packing='tile' "
                    "（Ampere sm>=8，tinygemm INT4）或 'auto'。\n"
                    f"原始错误: {type(ex).__name__}: {ex}"
                ) from ex
            err = str(ex)
            if "mslk" not in err.lower() and "int4_row_quantize" not in err.lower():
                print(f"Warning: torchao PLAIN quantize_ 失败 ({type(ex).__name__}: {ex})。")

    if int4_packing == "tile" or int4_packing == "auto":
        if not _cuda_tile_int4_supported():
            if int4_packing == "tile":
                raise RuntimeError(
                    "int4_packing='tile' 需要 CUDA 且 GPU 算力 >= 8.0（Ampere 及以上）以使用 "
                    "TILE_PACKED_TO_4D / tinygemm INT4。"
                )
            print(
                "Warning: 无法使用 TILE_PACKED_TO_4D 回退 (需要 CUDA 且 GPU 算力 >= 8.0)，"
                f"PLAIN 失败原因: {type(e_plain).__name__ if e_plain else 'N/A'}: {e_plain}"
            )
            return False, "none"

        try:
            _cast_filtered_linear_weights(model, torch.bfloat16, aggregator_linear_filter)
            cfg_tile = Int4WeightOnlyConfig(
                group_size=group_size,
                int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
            )
            quantize_(model, cfg_tile, filter_fn=aggregator_linear_filter)
            print(
                "torchao: 使用 Int4PackingFormat.TILE_PACKED_TO_4D（tinygemm 等 INT4 权重 GEMM）。"
                " bfloat16 仅用于打包阶段将权重送入 quantize_；前向仍为 INT4 权重矩阵乘，非 FP 权重 matmul。"
            )
            return True, "tile_packed"
        except Exception as e_tile:
            if int4_packing == "tile":
                raise RuntimeError(
                    f"int4_packing='tile' 下 TILE_PACKED_TO_4D 失败: {type(e_tile).__name__}: {e_tile}"
                ) from e_tile
            print(
                f"Warning: torchao TILE_PACKED_TO_4D 仍失败 ({type(e_tile).__name__}: {e_tile}), "
                "回退为反量化权重推理。"
            )
            return False, "none"

    raise AssertionError(f"unreachable: int4_packing={int4_packing!r}")


class QwTCompensation(nn.Module):
    """QwT 低秩补偿: 输入为 block 入口 x (与 PTQ AttnQwT/MlpQwT 一致)。"""

    def __init__(
        self,
        A: Optional[torch.Tensor] = None,
        B: Optional[torch.Tensor] = None,
        lora_bias: Optional[torch.Tensor] = None,
        qwt_enabled: bool = False,
    ):
        super().__init__()
        self.qwt_enabled = qwt_enabled
        if qwt_enabled and A is not None and B is not None:
            self.register_buffer("A", A.half().contiguous())
            self.register_buffer("B", B.half().contiguous())
            self.register_buffer("lora_bias", lora_bias.float().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.qwt_enabled:
            # 与 x 同 dtype 的标量，可与 [B, N, C] 广播，避免 float 补偿把后续 LayerNorm 弄成混精度
            return torch.zeros((), device=x.device, dtype=x.dtype)
        comp = (x.half() @ self.A @ self.B).float() + self.lora_bias
        return comp.to(dtype=x.dtype)


def build_qwt_map(qwt_data: List[Dict[str, Any]], device: torch.device) -> Dict[str, QwTCompensation]:
    out: Dict[str, QwTCompensation] = {}
    for entry in qwt_data:
        name = entry["name"]
        if entry.get("QwT_enabled"):
            comp = QwTCompensation(
                A=entry["A"],
                B=entry["B"],
                lora_bias=entry["lora_bias"],
                qwt_enabled=True,
            ).to(device)
        else:
            comp = QwTCompensation(qwt_enabled=False).to(device)
        out[name] = comp
    return out


def make_qwt_block_forward(
    block: nn.Module,
    attn_qwt: QwTCompensation,
    mlp_qwt: QwTCompensation,
) -> Callable[..., torch.Tensor]:
    original_attn = block.attn
    original_mlp = block.mlp
    norm1 = block.norm1
    norm2 = block.norm2
    ls1 = block.ls1
    ls2 = block.ls2

    def new_forward(self: nn.Module, x: torch.Tensor, pos: Any = None) -> torch.Tensor:
        if pos is not None:
            attn_out = ls1(original_attn(norm1(x), pos=pos))
        else:
            attn_out = ls1(original_attn(norm1(x)))
        attn_out = attn_out + attn_qwt(x)
        x = x + attn_out

        mlp_out = ls2(original_mlp(norm2(x)))
        mlp_out = mlp_out + mlp_qwt(x)
        x = x + mlp_out
        return x

    return new_forward


def iter_vggt_block_paths(model: VGGT) -> List[str]:
    pe = model.aggregator.patch_embed
    paths: List[str] = []
    for i in range(len(pe.blocks)):
        paths.append(f"aggregator.patch_embed.blocks.{i}")
    for i in range(len(model.aggregator.frame_blocks)):
        paths.append(f"aggregator.frame_blocks.{i}")
        paths.append(f"aggregator.global_blocks.{i}")
    return paths


def mount_qwt_and_patch_blocks(
    model: VGGT,
    qwt_map: Dict[str, QwTCompensation],
    device: torch.device,
) -> None:
    disabled = QwTCompensation(qwt_enabled=False).to(device)
    for block_path in iter_vggt_block_paths(model):
        block = get_submodule(model, block_path)
        attn_key = f"{block_path}.attn_QwT"
        mlp_key = f"{block_path}.mlp_QwT"
        attn_qwt = qwt_map.get(attn_key, disabled)
        mlp_qwt = qwt_map.get(mlp_key, disabled)
        set_submodule(model, attn_key, attn_qwt)
        set_submodule(model, mlp_key, mlp_qwt)
        new_fwd = make_qwt_block_forward(block, attn_qwt, mlp_qwt)
        block.forward = types.MethodType(new_fwd, block)  # type: ignore[method-assign]


def load_non_quant_params(model: nn.Module, non_quant: Dict[str, torch.Tensor], strict_warn: bool = True) -> int:
    loaded = 0
    for param_name, param_value in non_quant.items():
        try:
            parts = param_name.split(".")
            parent = model
            for p in parts[:-1]:
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)
            param = getattr(parent, parts[-1])
            if not isinstance(param, torch.Tensor):
                continue
            param.data.copy_(param_value.to(device=param.device, dtype=param.dtype))
            loaded += 1
        except Exception as e:
            if strict_warn:
                print(f"Warning: 跳过参数 {param_name}: {e}")
    return loaded


def load_w4a16_deployed_model(
    pretrained_model_path: str,
    deploy_dir: str,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    use_torchao: bool = True,
    torchao_group_size: int = 32,
    torchao_int4_packing: TorchaoInt4Packing = "auto",
    allow_fp_weight_fallback: bool = True,
    use_qwt: bool = True,
) -> Tuple[VGGT, Dict[str, Any]]:
    """
    构建可在 GPU 上推理的 W4A16(+QwT) 模型。

    默认 ``dtype=float32``：当前仓库内 VGGT 的 DPT head 在整网 float16 下会出现
    激活 float32 / 权重 float16 混用导致 conv 报错；与是否部署无关。
    若你确认环境上全 FP16 可跑通，可传入 ``dtype=torch.float16``。

    ``use_qwt=False``：不加载/不挂载 QwT，block 保持原始 forward（优先推理速度，精度下降）。

    ``torchao_int4_packing``：``auto`` | ``plain`` | ``tile``，见 ``apply_torchao_int4``。
    ``allow_fp_weight_fallback=False``：在 ``use_torchao=True`` 时若 INT4 打包失败则抛错，
    避免静默用反量化 FP 权重做 matmul。

    Returns:
        model, meta (deploy_config + 加载统计)
    """
    p_int4, p_qwt, p_nq, p_cfg = _required_deploy_files(deploy_dir, require_qwt=use_qwt)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type != "cuda":
        print("Warning: W4A16 部署建议在 CUDA 上运行; 当前为 CPU。")

    with open(p_cfg, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = VGGT.from_pretrained(pretrained_model_path)
    model.eval()
    model.to(dev, dtype=dtype)

    int4_data = torch.load(p_int4, map_location="cpu", weights_only=False)
    if use_qwt and os.path.isfile(p_qwt):
        qwt_data: List[Dict[str, Any]] = torch.load(p_qwt, map_location="cpu", weights_only=False)
    else:
        qwt_data = []
    non_quant = torch.load(p_nq, map_location="cpu", weights_only=False)

    n_inj = inject_int4_weights(model, int4_data, dtype=dtype)

    torchao_ok = False
    torchao_packing = "none"
    if use_torchao:
        torchao_ok, torchao_packing = apply_torchao_int4(
            model,
            group_size=torchao_group_size,
            int4_packing=torchao_int4_packing,
        )
        if not allow_fp_weight_fallback and not torchao_ok:
            raise RuntimeError(
                "已启用 torchao 且 allow_fp_weight_fallback=False，但 INT4 权重打包未成功 "
                f"(packing={torchao_packing!r})。请安装匹配 CUDA 的 mslk、换用 Ampere+ 试 "
                "torchao_int4_packing='tile'，或显式 allow_fp_weight_fallback=True。"
            )

    if use_qwt and qwt_data:
        qwt_map = build_qwt_map(qwt_data, dev)
        mount_qwt_and_patch_blocks(model, qwt_map, dev)

    n_nq = load_non_quant_params(model, non_quant, strict_warn=True)

    meta = {
        "deploy_config": cfg,
        "num_int4_entries": len(int4_data),
        "num_injected_linears": n_inj,
        "torchao_applied": torchao_ok,
        "torchao_packing": torchao_packing,
        "torchao_int4_packing": torchao_int4_packing,
        "allow_fp_weight_fallback": allow_fp_weight_fallback,
        "use_qwt": use_qwt,
        "num_qwt_modules": len(qwt_data),
        "num_non_quant_tensors_loaded": n_nq,
    }
    for p in model.parameters():
        p.requires_grad_(False)
    return model, meta


def forward_inference(
    model: nn.Module,
    imgs: torch.Tensor,
    aggregator_only: bool = False,
    use_amp: bool = False,
) -> Any:
    """
    推理前向：在 ``torch.inference_mode()`` 下执行，不建 autograd 图、不分配与梯度相关的中间状态
    （比 ``no_grad`` 更严格，适合纯推理）。
    """
    model.eval()
    with torch.inference_mode():
        device = imgs.device
        if use_amp and device.type == "cuda":
            autocast_dtype = (
                torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            )
            with torch.amp.autocast(device.type, dtype=autocast_dtype, enabled=True):
                if aggregator_only:
                    return model.aggregator(imgs)
                return model(imgs)
        if aggregator_only:
            return model.aggregator(imgs)
        return model(imgs)


def inference(model: VGGT, images: torch.Tensor) -> Dict[str, torch.Tensor]:
    """images: [B, S, 3, H, W], 数值范围 [0, 1]；dtype 与模型参数对齐；内部走 ``forward_inference``。"""
    p0 = next(model.parameters())
    images = images.to(device=p0.device, dtype=p0.dtype)
    return forward_inference(model, images, aggregator_only=False, use_amp=False)


# ---------------------------------------------------------------------------
# 评估 / benchmark（对齐 mv_recon/benchmark_common.py 的计时与显存统计）
# ---------------------------------------------------------------------------

COMPARE_OUTPUT_KEYS: Tuple[str, ...] = (
    "pose_enc",
    "depth",
    "depth_conf",
    "world_points",
    "world_points_conf",
)


def build_eval_batch(
    num_views: int,
    img_h: int,
    img_w: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    align: int = 14,
) -> torch.Tensor:
    """
    [B,S,3,H,W]，像素 [0,1]；H/W 按 patch 对齐（与 mv_recon/benchmark_common.build_dummy_batch 一致）。
    """
    torch.manual_seed(seed)
    h = (img_h // align) * align
    w = (img_w // align) * align
    return torch.rand(1, num_views, 3, h, w, device=device, dtype=dtype)


def state_dict_ram_bytes(model: nn.Module, backbone_only: bool = False) -> int:
    total = 0
    for k, t in model.state_dict().items():
        if not isinstance(t, torch.Tensor):
            continue
        if backbone_only and not _is_backbone_state_key(k):
            continue
        total += t.numel() * t.element_size()
    return int(total)


def _compare_save_temp_dir() -> Optional[str]:
    """
    torch.save 全量 FP state_dict 可达数 GiB，默认 /tmp 常爆满。
    优先顺序: PI3_COMPARE_TMPDIR -> TMPDIR -> /root/autodl-tmp -> 系统默认。
    """
    candidates: List[str] = []
    for env in ("PI3_COMPARE_TMPDIR", "TMPDIR"):
        v = os.environ.get(env)
        if v:
            candidates.append(os.path.expanduser(v))
    candidates.extend(["/root/autodl-tmp", "/tmp"])
    seen: set[str] = set()
    for d in candidates:
        if not d or d in seen:
            continue
        seen.add(d)
        try:
            if os.path.isdir(d) and os.access(d, os.W_OK):
                return d
        except OSError:
            continue
    return None


def torch_save_state_dict_disk_bytes(
    model: nn.Module, backbone_only: bool = False
) -> Tuple[int, bool]:
    """
    将 state_dict 写入临时 .pt 以度量 zip 落盘大小。

    Returns:
        (size_bytes, is_estimate)。若 ``torch.save`` 失败（多为临时目录满盘），
        返回基于张量体积的估算（约 +2%% 视作 zip/元数据）且 ``is_estimate=True``。
    """
    sd = model.state_dict()
    if backbone_only:
        sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor) and _is_backbone_state_key(k)}
    ram_est = int(sum(t.numel() * t.element_size() for t in sd.values()))
    tmp_dir = _compare_save_temp_dir()
    nt_kwargs: Dict[str, Any] = {"suffix": ".pt", "delete": False}
    if tmp_dir is not None:
        nt_kwargs["dir"] = tmp_dir
    path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(**nt_kwargs) as f:
            path = f.name
        torch.save(sd, path)
        return int(os.path.getsize(path)), False
    except (RuntimeError, OSError) as e:
        print(
            f"Warning: torch.save(state_dict) 失败 ({type(e).__name__}: {e})。"
            "多为临时目录空间不足（全精度 VGGT 约数 GiB）。"
            "已用张量体积估算 .pt 大小；可设置环境变量 PI3_COMPARE_TMPDIR 或 TMPDIR 指向大空间目录后重试。"
        )
        fudge = int(ram_est * 0.02) + 262144
        return ram_est + fudge, True
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def deploy_artifacts_disk_bytes(deploy_dir: str) -> Dict[str, Any]:
    names = ["int4_weights.pt", "qwt_compensation.pt", "non_quant_params.pt", "deploy_config.json"]
    parts: Dict[str, int] = {}
    total = 0
    for fn in names:
        p = os.path.join(deploy_dir, fn)
        if os.path.isfile(p):
            sz = int(os.path.getsize(p))
            parts[fn] = sz
            total += sz
    return {"files": parts, "total_bytes": total, "total_mib": total / (1024**2)}


def _tensor_storage_bytes(t: Any) -> int:
    return int(t.numel() * t.element_size()) if isinstance(t, torch.Tensor) else 0


def deploy_backbone_artifacts_tensor_bytes(
    deploy_dir: str, require_qwt: bool = True
) -> Dict[str, Any]:
    """
    按 **参数名前缀 ``aggregator``** 统计部署目录内张量的存储字节（非整文件 getsize）。
    与 ``state_dict_ram_bytes(..., backbone_only=True)`` 口径一致，便于对比主干体积。
    """
    p_int4, p_qwt, p_nq, p_cfg = _required_deploy_files(deploy_dir, require_qwt=require_qwt)
    int4_data: List[Dict[str, Any]] = torch.load(p_int4, map_location="cpu", weights_only=False)
    int4_b = 0
    for e in int4_data:
        fqn = saved_int4_name_to_linear_fqn(e["name"])
        if not _is_backbone_state_key(fqn):
            continue
        int4_b += _tensor_storage_bytes(e.get("int_weight"))
        int4_b += _tensor_storage_bytes(e.get("w_interval"))
        int4_b += _tensor_storage_bytes(e.get("bias"))

    qwt_b = 0
    if require_qwt and os.path.isfile(p_qwt):
        qwt_data: List[Dict[str, Any]] = torch.load(p_qwt, map_location="cpu", weights_only=False)
        for e in qwt_data:
            nm = str(e.get("name", ""))
            if not (nm == VGGT_BACKBONE_PREFIX or nm.startswith(VGGT_BACKBONE_PREFIX + ".")):
                continue
            qwt_b += _tensor_storage_bytes(e.get("A"))
            qwt_b += _tensor_storage_bytes(e.get("B"))
            qwt_b += _tensor_storage_bytes(e.get("lora_bias"))

    non_quant: Dict[str, Any] = torch.load(p_nq, map_location="cpu", weights_only=False)
    nq_b = 0
    for k, t in non_quant.items():
        if not _is_backbone_state_key(str(k)):
            continue
        nq_b += _tensor_storage_bytes(t)

    cfg_b = int(os.path.getsize(p_cfg)) if os.path.isfile(p_cfg) else 0
    total = int4_b + qwt_b + nq_b
    return {
        "int4_tensor_bytes": int4_b,
        "qwt_tensor_bytes": qwt_b,
        "non_quant_tensor_bytes": nq_b,
        "config_json_bytes": cfg_b,
        "total_param_tensor_bytes": total,
        "total_including_config_bytes": total + cfg_b,
        "total_mib": total / (1024**2),
        "scope": "backbone_aggregator_prefix_only",
    }


def _forward_once(
    model: nn.Module,
    imgs: torch.Tensor,
    aggregator_only: bool,
    use_amp: bool,
    device: torch.device,
) -> None:
    _ = forward_inference(model, imgs, aggregator_only=aggregator_only, use_amp=use_amp)


def benchmark_forward_timing(
    model: nn.Module,
    imgs: torch.Tensor,
    warmup: int,
    iters: int,
    device: torch.device,
    use_amp: bool,
    aggregator_only: bool,
) -> Dict[str, Any]:
    def one() -> float:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _forward_once(model, imgs, aggregator_only, use_amp, device)
        if device.type == "cuda":
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


def _collect_compare_tensors(pred: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k in COMPARE_OUTPUT_KEYS:
        if k not in pred:
            continue
        v = pred[k]
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().float().cpu()
    return out


def _tensor_diff(fp: torch.Tensor, dq: torch.Tensor) -> Dict[str, float]:
    if fp.shape != dq.shape:
        return {
            "error": f"shape mismatch fp={tuple(fp.shape)} dq={tuple(dq.shape)}",
            "max_abs": float("nan"),
            "mean_abs": float("nan"),
            "rel_mean": float("nan"),
        }
    d = (fp - dq).abs()
    denom = float(fp.abs().mean().item()) + 1e-8
    return {
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "rel_mean": float(d.mean().item() / denom),
    }


def _diff_aggregator_outputs(fp_out: Any, dq_out: Any) -> Dict[str, Any]:
    """``Aggregator.forward`` 返回 ``(List[Tensor], patch_start_idx)``。"""
    out: Dict[str, Any] = {}
    if not (
        isinstance(fp_out, tuple)
        and isinstance(dq_out, tuple)
        and len(fp_out) >= 2
        and len(dq_out) >= 2
    ):
        out["error"] = "aggregator 输出应为 tuple(token_list, patch_start_idx)"
        return out
    fp_list, fp_ps = fp_out[0], fp_out[1]
    dq_list, dq_ps = dq_out[0], dq_out[1]
    fp_i = int(fp_ps) if isinstance(fp_ps, torch.Tensor) else int(fp_ps)
    dq_i = int(dq_ps) if isinstance(dq_ps, torch.Tensor) else int(dq_ps)
    out["patch_start_idx_match"] = fp_i == dq_i
    if not isinstance(fp_list, list) or not isinstance(dq_list, list):
        out["error"] = "token 列表类型非 list"
        return out
    if len(fp_list) != len(dq_list):
        out["error"] = f"stage 数不一致 fp={len(fp_list)} dq={len(dq_list)}"
        return out
    for i, (a, b) in enumerate(zip(fp_list, dq_list)):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            out[f"stage_{i}"] = _tensor_diff(a.detach().float().cpu(), b.detach().float().cpu())
        else:
            out[f"stage_{i}"] = {"error": "non-tensor stage output"}
    return out


def run_fp_vs_deploy_compare(
    pretrained_model_path: str,
    deploy_dir: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    use_torchao: bool = True,
    torchao_group_size: int = 32,
    torchao_int4_packing: TorchaoInt4Packing = "auto",
    allow_fp_weight_fallback: bool = True,
    use_amp_benchmark: bool = False,
    use_qwt: bool = True,
    num_views: int = 4,
    img_h: int = 518,
    img_w: int = 518,
    seed: int = 42,
    warmup: int = 5,
    iters: int = 20,
    aggregator_only: bool = True,
) -> Dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("本评估需要 CUDA（与 benchmark_common / PTQ eval 一致）。")

    imgs_fp32 = build_eval_batch(num_views, img_h, img_w, device, torch.float32, seed)

    model_fp = VGGT.from_pretrained(pretrained_model_path).to(device).to(dtype).eval()
    for p in model_fp.parameters():
        p.requires_grad_(False)
    out_fp = forward_inference(model_fp, imgs_fp32, aggregator_only=aggregator_only)
    if not aggregator_only:
        tensors_fp = _collect_compare_tensors(out_fp)

    del model_fp
    gc.collect()
    torch.cuda.empty_cache()

    model_dq, deploy_meta = load_w4a16_deployed_model(
        pretrained_model_path,
        deploy_dir,
        device=str(device),
        dtype=dtype,
        use_torchao=use_torchao,
        torchao_group_size=torchao_group_size,
        torchao_int4_packing=torchao_int4_packing,
        allow_fp_weight_fallback=allow_fp_weight_fallback,
        use_qwt=use_qwt,
    )
    out_dq = forward_inference(model_dq, imgs_fp32, aggregator_only=aggregator_only)
    if aggregator_only:
        output_diff = _diff_aggregator_outputs(out_fp, out_dq)
    else:
        tensors_dq = _collect_compare_tensors(out_dq)
        output_diff = {}
        for k in tensors_fp:
            if k in tensors_dq:
                output_diff[k] = _tensor_diff(tensors_fp[k], tensors_dq[k])
            else:
                output_diff[k] = {"error": "missing in deploy output"}

    del out_fp, out_dq, model_dq
    if not aggregator_only:
        del tensors_fp
        del tensors_dq
    gc.collect()
    torch.cuda.empty_cache()

    model_fp = VGGT.from_pretrained(pretrained_model_path).to(device).to(dtype).eval()
    for p in model_fp.parameters():
        p.requires_grad_(False)
    bb = aggregator_only
    ram_fp = state_dict_ram_bytes(model_fp, backbone_only=bb)
    disk_fp, disk_fp_estimated = torch_save_state_dict_disk_bytes(model_fp, backbone_only=bb)
    torch.cuda.reset_peak_memory_stats(device)
    lat_fp = benchmark_forward_timing(
        model_fp, imgs_fp32, warmup, iters, device, use_amp_benchmark, aggregator_only
    )
    peak_fp_a = int(torch.cuda.max_memory_allocated(device))
    peak_fp_r = int(torch.cuda.max_memory_reserved(device))
    del model_fp
    gc.collect()
    torch.cuda.empty_cache()

    model_dq, _ = load_w4a16_deployed_model(
        pretrained_model_path,
        deploy_dir,
        device=str(device),
        dtype=dtype,
        use_torchao=use_torchao,
        torchao_group_size=torchao_group_size,
        torchao_int4_packing=torchao_int4_packing,
        allow_fp_weight_fallback=allow_fp_weight_fallback,
        use_qwt=use_qwt,
    )
    ram_dq = state_dict_ram_bytes(model_dq, backbone_only=bb)
    if bb:
        disk_dq_detail: Dict[str, Any] = deploy_backbone_artifacts_tensor_bytes(
            deploy_dir, require_qwt=use_qwt
        )
    else:
        disk_dq_detail = deploy_artifacts_disk_bytes(deploy_dir)
    torch.cuda.reset_peak_memory_stats(device)
    lat_dq = benchmark_forward_timing(
        model_dq, imgs_fp32, warmup, iters, device, use_amp_benchmark, aggregator_only
    )
    peak_dq_a = int(torch.cuda.max_memory_allocated(device))
    peak_dq_r = int(torch.cuda.max_memory_reserved(device))
    del model_dq
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "pretrained_model_path": pretrained_model_path,
        "deploy_dir": deploy_dir,
        "device": str(device),
        "model_dtype": str(dtype),
        "use_amp_benchmark": use_amp_benchmark,
        "use_qwt": use_qwt,
        "aggregator_only": aggregator_only,
        "input_shape": list(imgs_fp32.shape),
        "seed": seed,
        "deploy_load_meta": {k: v for k, v in deploy_meta.items() if k != "deploy_config"},
        "output_diff_fp32_forward": output_diff,
        "fp_baseline": {
            "latency_ms": lat_fp,
            "memory_bytes": {
                "peak_allocated": peak_fp_a,
                "peak_reserved": peak_fp_r,
                "peak_allocated_mib": peak_fp_a / (1024**2),
                "peak_reserved_mib": peak_fp_r / (1024**2),
            },
            "size_bytes": {
                "state_dict_ram": ram_fp,
                "torch_save_state_dict_disk": disk_fp,
                "torch_save_disk_estimated": disk_fp_estimated,
                "state_dict_ram_mib": ram_fp / (1024**2),
                "torch_save_disk_mib": disk_fp / (1024**2),
            },
        },
        "w4a16_deploy": {
            "latency_ms": lat_dq,
            "memory_bytes": {
                "peak_allocated": peak_dq_a,
                "peak_reserved": peak_dq_r,
                "peak_allocated_mib": peak_dq_a / (1024**2),
                "peak_reserved_mib": peak_dq_r / (1024**2),
            },
            "size_bytes": {
                "state_dict_ram": ram_dq,
                "state_dict_ram_mib": ram_dq / (1024**2),
                **(
                    {"deploy_backbone_tensor_bytes": disk_dq_detail}
                    if aggregator_only
                    else {"deploy_artifacts_on_disk": disk_dq_detail}
                ),
            },
        },
    }


def _print_compare_report(report: Dict[str, Any]) -> None:
    fp = report["fp_baseline"]
    dq = report["w4a16_deploy"]
    fpl, dql = fp["latency_ms"], dq["latency_ms"]
    fpm, dqm = fp["memory_bytes"], dq["memory_bytes"]
    fps, dqs = fp["size_bytes"], dq["size_bytes"]

    print()
    shp = report.get("input_shape", [])
    if len(shp) >= 2:
        print(f"(输入 B={shp[0]}, 视角数 S={shp[1]}, H=W≈{shp[-1]})")
    if report.get("aggregator_only"):
        print("=== 输出差异 (aggregator 主干 forward, 无 autocast) ===")
    else:
        print("=== 输出差异 (同输入 float32 全模型 forward, 无 autocast) ===")
    for k, v in report["output_diff_fp32_forward"].items():
        if isinstance(v, dict):
            if "error" in v:
                print(f"  {k}: {v}")
            elif "max_abs" in v:
                print(
                    f"  {k}: max_abs={v['max_abs']:.6e}  mean_abs={v['mean_abs']:.6e}  rel_mean={v['rel_mean']:.6e}"
                )
            else:
                print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    tgt = "aggregator" if report["aggregator_only"] else "full_model"
    amp = report["use_amp_benchmark"]
    print()
    print(f"=== 推理延时 (target={tgt}, amp={amp}) ===")
    print(f"  {'':42} {'FP':>14} {'W4A16 deploy':>14}")
    print("-" * 72)
    print(f"  {'latency mean (ms)':42} {fpl['mean_ms']:14.3f} {dql['mean_ms']:14.3f}")
    print(f"  {'latency stdev (ms)':42} {fpl['stdev_ms']:14.3f} {dql['stdev_ms']:14.3f}")
    print(f"  {'latency min (ms)':42} {fpl['min_ms']:14.3f} {dql['min_ms']:14.3f}")

    print()
    print("=== 峰值显存 (benchmark 段内 max_memory_*) ===")
    print(f"  {'peak VRAM allocated (MiB)':42} {fpm['peak_allocated_mib']:14.1f} {dqm['peak_allocated_mib']:14.1f}")
    print(f"  {'peak VRAM reserved (MiB)':42} {fpm['peak_reserved_mib']:14.1f} {dqm['peak_reserved_mib']:14.1f}")

    print()
    print("=== 模型大小 ===")
    ram_hdr = (
        "state_dict RAM (aggregator, MiB)"
        if report.get("aggregator_only")
        else "state_dict RAM (MiB)"
    )
    print(f"  {ram_hdr:42} {fps['state_dict_ram_mib']:14.1f} {dqs['state_dict_ram_mib']:14.1f}")
    fp_disk_hdr = (
        "torch.save FP backbone disk (MiB)"
        if report.get("aggregator_only")
        else "torch.save FP state_dict disk (MiB)"
    )
    if fps.get("torch_save_disk_estimated"):
        fp_disk_hdr = fp_disk_hdr.replace("disk (MiB)", "disk (MiB, 估算)")
    print(f"  {fp_disk_hdr:42} {fps['torch_save_disk_mib']:14.1f} {'—':>14}")
    bb_tb = dqs.get("deploy_backbone_tensor_bytes")
    if bb_tb is not None:
        deploy_mib = float(bb_tb.get("total_mib", 0))
        deploy_lbl = "deploy 主干张量 (aggregator, MiB)"
    else:
        art = dqs.get("deploy_artifacts_on_disk", {})
        deploy_mib = float(art.get("total_mib", 0))
        deploy_lbl = "deploy 目录权重文件合计 (MiB)"
    print(f"  {deploy_lbl:42} {'—':>14} {deploy_mib:14.1f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="加载 VGGT W4A16 部署权重并可选跑 dummy 推理")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("VGGT_MODEL_PATH", "/root/autodl-tmp/hf_hub/models--facebook--VGGT-1B"),
        help="VGGT HuggingFace 本地目录或 hub id",
    )
    parser.add_argument(
        "--deploy-dir",
        type=str,
        default="/root/Pi3-evaluation/outputs/w4a16_deploy",
        help="含 int4_weights.pt / qwt_compensation.pt / non_quant_params.pt 的目录",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="整网 float16（若 DPT head 报 dtype 混用，请去掉此开关使用默认 float32）",
    )
    parser.add_argument("--no-torchao", action="store_true", help="跳过 torchao, 仅用反量化 FP16")
    parser.add_argument(
        "--torchao-packing",
        type=str,
        choices=["auto", "plain", "tile"],
        default="auto",
        help="INT4 权重打包：auto=先 PLAIN(mslk) 再 TILE(tinygemm,sm>=8)；plain/tile=只走一路，失败可抛错",
    )
    parser.add_argument(
        "--strict-torchao",
        action="store_true",
        help="torchao INT4 打包失败时抛错，禁止静默回退为反量化 FP 权重 matmul",
    )
    parser.add_argument("--dummy", action="store_true", help="随机输入跑一次 forward")
    parser.add_argument(
        "--compare-fp",
        action="store_true",
        help="全精度 vs W4A16 部署：真实 forward 数值对比 + 延时/显存/模型大小（需 CUDA）",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=4,
        metavar="S",
        help="输入视角数 S，形状 [1,S,3,H,W]；例如 --num-views 50 模拟 50 视角以测显存/延时（与 PTQ「scan 张数」无关）",
    )
    parser.add_argument("--img-h", type=int, default=518, help="输入高，会按 14 对齐")
    parser.add_argument("--img-w", type=int, default=518, help="输入宽，会按 14 对齐")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="延时测试时使用 torch.amp.autocast（与 benchmark_common 一致）；输出对比仍为 float32",
    )
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="--compare-fp 时测整网 forward 与各 head 输出；默认仅主干 aggregator",
    )
    parser.add_argument("--save-json", type=str, default=None, help="保存完整报告 JSON")
    parser.add_argument(
        "--no-qwt",
        action="store_true",
        help="部署侧不加载 QwT（更快，与 PTQ+QwT 全精度输出差异会变大）",
    )
    args = parser.parse_args()

    if args.compare_fp:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA 不可用，无法运行 --compare-fp")
        dev = torch.device(args.device or "cuda")
        if args.fp16:
            print("Warning: --compare-fp 默认用 float32 做对比与计时；已忽略 --fp16。")
        report = run_fp_vs_deploy_compare(
            args.model,
            args.deploy_dir,
            dev,
            dtype=torch.float32,
            use_torchao=not args.no_torchao,
            torchao_int4_packing=args.torchao_packing,
            allow_fp_weight_fallback=not args.strict_torchao,
            use_amp_benchmark=args.amp,
            use_qwt=not args.no_qwt,
            num_views=args.num_views,
            img_h=args.img_h,
            img_w=args.img_w,
            seed=args.seed,
            warmup=args.warmup,
            iters=args.iters,
            aggregator_only=not args.full_model,
        )
        _print_compare_report(report)
        if args.save_json:
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"已写入 {args.save_json}")
        return

    model, meta = load_w4a16_deployed_model(
        args.model,
        args.deploy_dir,
        device=args.device,
        dtype=torch.float16 if args.fp16 else torch.float32,
        use_torchao=not args.no_torchao,
        torchao_int4_packing=args.torchao_packing,
        allow_fp_weight_fallback=not args.strict_torchao,
        use_qwt=not args.no_qwt,
    )
    print("加载完成:", json.dumps({k: v for k, v in meta.items() if k != "deploy_config"}, indent=2))

    if args.dummy:
        dev = next(model.parameters()).device
        dt = next(model.parameters()).dtype
        x = torch.rand(1, 2, 3, 518, 518, device=dev, dtype=dt)
        out = inference(model, x)
        print("Dummy 输出 keys:", list(out.keys()))
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")


if __name__ == "__main__":
    main()