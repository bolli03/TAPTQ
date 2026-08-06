"""
W4A4 部署：真正的 INT4×INT4 GEMM（via torch._int_mm INT8 Tensor Core）

与原版 w4a4_deploy.py 的关键区别：
  - 原版：torchao tinygemm = INT4 权重 dequant → FP16 matmul（假 INT4）
  - 本版：torch._int_mm = INT8 Tensor Core GEMM（INT4 值域的整数运算，真正的整数 GEMM）

流程：
  1) 激活：FP16 → 静态 INT4 量化（clamp+round, a_qmax=8）→ 存为 INT8
  2) 权重：INT4（已校准, 存为 INT8, 值域 [-8,7]）
  3) GEMM：INT8 × INT8 → INT32（Tensor Core, SM80+）
  4) 反量化：INT32 → FP16（乘 a_scale * w_scale）
  5) 非线性层（LayerNorm / GeLU / Softmax）：FP16

硬件要求：SM80+（A100 / RTX 3090 / RTX 4090 / Pro 6000 / ...）

用法：
    CUDA_VISIBLE_DEVICES=1 python deployment/w4a4_deploy_int4gemm.py --dummy
    CUDA_VISIBLE_DEVICES=1 python deployment/w4a4_deploy_int4gemm.py --compare-fp --num-views 50 --no-qwt
    CUDA_VISIBLE_DEVICES=1 python deployment/w4a4_deploy_int4gemm.py --profile-ops --num-views 4
"""

from __future__ import annotations

import argparse
from typing import Any
import gc
import json
import os
import time
import types

import torch
import torch.nn as nn

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from PTQ.vggt.models.vggt import VGGT  # noqa: E402


# ================================================================
# 1. W4A4 Linear: 真正的 INT GEMM
# ================================================================


def _expand_w_interval(w_interval: torch.Tensor, out_features: int) -> torch.Tensor:
    """将 w_interval 展开为 per-channel [out_features] 形状。

    处理:
      - [1,1,1,1] → scalar → expand to [N]
      - [3,1,1,1] → qkv repeat_interleave → [N]
      - [N,1,1,1] → squeeze → [N]
    """
    w_s = w_interval.float().squeeze()
    if w_s.dim() == 0:
        return w_s.expand(out_features)
    if w_s.numel() == out_features:
        return w_s.reshape(out_features)
    if w_s.numel() < out_features and out_features % w_s.numel() == 0:
        return w_s.repeat_interleave(out_features // w_s.numel())
    raise ValueError(
        f"Cannot expand w_interval ({w_s.shape}, numel={w_s.numel()}) to {out_features}"
    )


class W4A4Linear(nn.Module):
    """
    真正的 W4A4 量化线性层。

    计算流程:
      x (FP16) → INT4 量化 (clamp+round, 存为 int8)
      →  torch._int_mm(x_int8, w_int8_T)  → INT32 (Tensor Core)
      →  dequant: int32 * a_scale * w_scale → 与输入同 dtype

    对比 torchao tinygemm (W4A16):
      - tinygemm:   INT4 权重 → on-the-fly dequant → FP16 × FP16 matmul
      - W4A4Linear: INT4 act + INT4 weight → INT8 Tensor Core → INT32 → dequant
    """

    _MIN_M = 16  # torch._int_mm 要求 M >= 16

    def __init__(
        self,
        int_weight: torch.Tensor,
        w_scale: torch.Tensor,
        a_scale: float,
        a_qmax: int = 8,
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        N, K = int_weight.shape
        self.N, self.K = N, K
        self.a_qmax = a_qmax

        # 权重转置并存为连续 INT8: [K, N]
        self.register_buffer("w_int_t", int_weight.t().contiguous().to(torch.int8))
        # Per-channel 权重 scale: [N]
        self.register_buffer("w_scale", w_scale.float().contiguous())
        # Per-tensor 激活 scale: scalar
        self.register_buffer("a_scale", torch.tensor(a_scale, dtype=torch.float32))

        if bias is not None:
            self.register_buffer("bias", bias.float().contiguous())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.K)
        M = x_2d.shape[0]

        # ---- 激活 INT4 量化 (存为 INT8) ----
        a_s = self.a_scale
        x_int = torch.clamp(
            torch.round(x_2d.float() / a_s), -self.a_qmax, self.a_qmax
        ).to(torch.int8)

        # ---- INT8 Tensor Core GEMM ----
        if M < self._MIN_M:
            # torch._int_mm 要求 M >= 16; 小 M 时 pad
            pad = self._MIN_M - M
            x_int = torch.nn.functional.pad(x_int, (0, 0, 0, pad))
            out_i32 = torch._int_mm(x_int, self.w_int_t)  # [16, N]
            out_i32 = out_i32[:M]
        else:
            out_i32 = torch._int_mm(x_int, self.w_int_t)  # [M, N]

        # ---- 反量化 ----
        out = out_i32.float() * (a_s * self.w_scale)  # broadcast [N]

        if self.bias is not None:
            out = out + self.bias

        # 与输入 / 其余 Linear·LayerNorm 的 dtype 一致（默认加载为 float32 时不能强行 .half()）
        return out.reshape(*orig_shape[:-1], self.N).to(dtype=x.dtype)


# ================================================================
# 2. QwT 补偿模块
# ================================================================


class QwTCompensation(nn.Module):
    """QwT 补偿: 在 FP16 上算低秩项，再 cast 回 x.dtype，避免与残差 dtype 不一致。"""
    def __init__(self, A=None, B=None, lora_bias=None, qwt_enabled=False):
        super().__init__()
        self.qwt_enabled = qwt_enabled
        if qwt_enabled and A is not None:
            self.register_buffer("A", A.half())
            self.register_buffer("B", B.half())
            self.register_buffer("lora_bias", lora_bias.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor | int:
        if not self.qwt_enabled:
            return 0
        delta = (x.half() @ self.A @ self.B).float() + self.lora_bias
        return delta.to(dtype=x.dtype)


# ================================================================
# 3. 模型加载
# ================================================================


def _get_submodule(model: nn.Module, path: str) -> nn.Module:
    obj = model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _set_submodule(model: nn.Module, path: str, value: nn.Module):
    parts = path.split(".")
    parent = _get_submodule(model, ".".join(parts[:-1]))
    if parts[-1].isdigit():
        parent[int(parts[-1])] = value
    else:
        setattr(parent, parts[-1], value)


def load_w4a4_deployed_model(
    pretrained_model_path: str,
    deploy_dir: str | None = None,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
    use_qwt: bool = True,
):
    """
    加载 W4A4 模型，使用 torch._int_mm 做真正的 INT GEMM。

    Returns:
        (model, meta_dict)
    """

    if deploy_dir is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        deploy_dir = os.path.join(repo_root, "outputs", "w4a4_deploy")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ---- Step 1: 加载原始模型结构 ----
    print(f"[W4A4] 加载 VGGT 模型结构: {pretrained_model_path}")
    model = VGGT.from_pretrained(pretrained_model_path)
    model = model.to(dtype=dtype, device=dev).eval()

    # ---- Step 2: 加载 INT4 权重 → 替换为 W4A4Linear ----
    int4_path = os.path.join(deploy_dir, "int4_weights.pt")
    print(f"[W4A4] 加载 INT4 权重: {int4_path}")
    int4_data = torch.load(int4_path, map_location="cpu", weights_only=False)

    replaced = 0
    for entry in int4_data:
        name = entry["name"]
        if "int_weight" not in entry:
            continue

        int_w = entry["int_weight"].to(torch.int8)  # [N, K]
        N = int_w.shape[0]
        w_scale = _expand_w_interval(entry["w_interval"], N)  # [N]
        a_scale = entry["a_interval"].float().squeeze().item()
        a_qmax = int(entry["a_qmax"]) if "a_qmax" in entry else 8
        bias = entry.get("bias", None)

        w4a4 = W4A4Linear(
            int_weight=int_w,
            w_scale=w_scale,
            a_scale=a_scale,
            a_qmax=a_qmax,
            bias=bias,
        ).to(dev)

        _set_submodule(model, name, w4a4)
        replaced += 1

    print(f"[W4A4] 替换 {replaced} 个 Linear → W4A4Linear (torch._int_mm)")

    # ---- Step 3: 加载 QwT 补偿模块 ----
    qwt_path = os.path.join(deploy_dir, "qwt_compensation.pt")
    qwt_map: dict[str, QwTCompensation] = {}
    enabled_count = 0

    if use_qwt and os.path.exists(qwt_path):
        print(f"[W4A4] 加载 QwT 补偿: {qwt_path}")
        qwt_data = torch.load(qwt_path, map_location="cpu", weights_only=False)
        for entry in qwt_data:
            qname = entry["name"]
            if entry.get("QwT_enabled", False) and "A" in entry:
                comp = QwTCompensation(
                    A=entry["A"], B=entry["B"],
                    lora_bias=entry["lora_bias"],
                    qwt_enabled=True,
                ).to(dev)
                enabled_count += 1
            else:
                comp = QwTCompensation(qwt_enabled=False).to(dev)
            qwt_map[qname] = comp
        print(f"[W4A4] QwT 补偿: {enabled_count}/{len(qwt_data)} 个模块启用")

    # ---- Step 4: 挂载 QwT 到 block 并修改 forward ----
    if qwt_map:
        _patch_blocks_with_qwt(model, qwt_map)

    # ---- Step 5: 加载非量化参数 ----
    nq_path = os.path.join(deploy_dir, "non_quant_params.pt")
    if os.path.exists(nq_path):
        print(f"[W4A4] 加载非量化参数: {nq_path}")
        non_quant = torch.load(nq_path, map_location="cpu", weights_only=False)
        loaded = 0
        for pname, pval in non_quant.items():
            try:
                parts = pname.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                param = getattr(parent, parts[-1])
                param.data.copy_(pval.to(param.device, param.dtype))
                loaded += 1
            except Exception:
                pass
        print(f"[W4A4] 已加载 {loaded}/{len(non_quant)} 个非量化参数")

    # ---- 元信息 ----
    config_path = os.path.join(deploy_dir, "deploy_config.json")
    deploy_config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            deploy_config = json.load(f)

    meta = {
        "gemm_backend": "torch._int_mm (INT8 Tensor Core)",
        "w4a4_layers": replaced,
        "qwt_enabled": enabled_count,
        "qwt_total": len(qwt_map),
        "deploy_config": deploy_config,
    }
    model.eval()
    return model, meta


def _patch_blocks_with_qwt(model, qwt_map: dict[str, QwTCompensation]):
    """修改所有 block 的 forward, 加入 QwT 补偿。"""

    # 收集所有 block 路径
    block_groups = [
        ("aggregator.patch_embed.blocks", 24),
        ("aggregator.frame_blocks", 24),
        ("aggregator.global_blocks", 24),
    ]

    patched = 0
    for group_path, num_blocks in block_groups:
        for i in range(num_blocks):
            block_path = f"{group_path}.{i}"
            attn_qwt_name = f"{block_path}.attn_QwT"
            mlp_qwt_name = f"{block_path}.mlp_QwT"

            attn_qwt = qwt_map.get(attn_qwt_name, QwTCompensation(qwt_enabled=False))
            mlp_qwt = qwt_map.get(mlp_qwt_name, QwTCompensation(qwt_enabled=False))

            try:
                block = _get_submodule(model, block_path)
            except (AttributeError, IndexError):
                continue

            # 挂载 QwT 模块
            block.attn_QwT = attn_qwt
            block.mlp_QwT = mlp_qwt

            # Monkey-patch forward
            _orig_attn = block.attn
            _norm1 = block.norm1
            _ls1 = block.ls1
            _orig_mlp = block.mlp
            _norm2 = block.norm2
            _ls2 = block.ls2
            _attn_qwt = attn_qwt
            _mlp_qwt = mlp_qwt

            def _make_forward(attn, norm1, ls1, mlp, norm2, ls2, aq, mq):
                def _forward(self, x, pos=None):
                    # Attention + QwT
                    attn_in = norm1(x)
                    attn_out = ls1(attn(attn_in, pos=pos) if pos is not None else attn(attn_in))
                    attn_out = attn_out + aq(x)
                    x = x + attn_out

                    # MLP + QwT
                    mlp_out = ls2(mlp(norm2(x)))
                    mlp_out = mlp_out + mq(x)
                    x = x + mlp_out
                    return x
                return _forward

            block.forward = types.MethodType(
                _make_forward(_orig_attn, _norm1, _ls1, _orig_mlp, _norm2, _ls2, _attn_qwt, _mlp_qwt),
                block,
            )
            patched += 1

    print(f"[W4A4] 已 patch {patched} 个 block forward (QwT 补偿)")


# ================================================================
# 4. 推理 & 对比
# ================================================================


@torch.no_grad()
def inference(model: nn.Module, images: torch.Tensor) -> dict:
    model.eval()
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    return model(images.to(device=dev, dtype=dt))


@torch.no_grad()
def forward_aggregator_only(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """只跑 aggregator 主干, 不跑 DPT heads。返回 aggregator 输出 tensor。"""
    model.eval()
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    images = images.to(device=dev, dtype=dt)
    return model.aggregator(images)


@torch.no_grad()
def run_op_profile(
    model: nn.Module,
    images: torch.Tensor,
    *,
    full_model: bool = False,
    warmup: int = 3,
    row_limit: int = 50,
    chrome_trace_path: str | None = None,
    sort_by: str = "cuda_time_total",
) -> Any:
    """用 torch.profiler 统计单次 forward 内各 aten 算子的 CPU/CUDA 耗时。

    full_model=False 时只跑 aggregator，栈更短、更贴近 --compare-fp 默认行为。
    表格可在 Chrome 打开 trace：chrome://tracing 加载 --profile-chrome 生成的 json。
    """
    model.eval()
    fwd = inference if full_model else forward_aggregator_only
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    images = images.to(device=dev, dtype=dt)

    for _ in range(max(0, warmup)):
        _ = fwd(model, images)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU]
    use_cuda = dev.type == "cuda" and torch.cuda.is_available()
    if use_cuda:
        activities.insert(0, torch.profiler.ProfilerActivity.CUDA)

    sort_key = sort_by
    if not use_cuda and "cuda" in sort_key:
        sort_key = "cpu_time_total"

    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        out = fwd(model, images)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    print("\n" + "=" * 70)
    print(f"  torch.profiler 算子延时 (sort_by={sort_key}, row_limit={row_limit})")
    print("=" * 70)
    print(prof.key_averages().table(sort_by=sort_key, row_limit=row_limit))

    if chrome_trace_path:
        prof.export_chrome_trace(chrome_trace_path)
        print(f"Chrome trace: {chrome_trace_path}")

    return out


def compute_model_size(model: nn.Module, aggregator_only: bool = True) -> dict:
    """统计模型各部分大小 (MB)。

    W4A4Linear: 权重量按逻辑 INT4（4 bit/元素 → 0.5 字节/元素），scale/bias 按实际存储。
    QwT: A、B、lora_bias 按 FP16 计（2 字节/元素），与实现中存 half、偏置常视为补偿精度无关。
    其余参数与 buffer 按 element_size 计算。
    """
    target = model.aggregator if aggregator_only else model

    int4_bytes = 0.0  # 逻辑 INT4 权重: numel * 0.5
    scale_bytes = 0   # W4A4Linear 的 scale/bias
    qwt_bytes = 0     # QwT：按 FP16 计
    other_bytes = 0   # LayerNorm, LayerScale, pos_embed 等
    fp16_bpe = 2      # bytes per element，QwT 统一按 FP16

    for name, module in target.named_modules():
        if isinstance(module, W4A4Linear):
            int4_bytes += 0.5 * module.w_int_t.numel()
            scale_bytes += module.w_scale.numel() * module.w_scale.element_size()
            scale_bytes += module.a_scale.numel() * module.a_scale.element_size()
            if module.bias is not None:
                scale_bytes += module.bias.numel() * module.bias.element_size()
        elif isinstance(module, QwTCompensation) and module.qwt_enabled:
            qwt_bytes += module.A.numel() * fp16_bpe
            qwt_bytes += module.B.numel() * fp16_bpe
            qwt_bytes += module.lora_bias.numel() * fp16_bpe

    # 非量化参数: 排除 W4A4Linear 和 QwTCompensation 的子模块
    quant_param_ids = set()
    for module in target.modules():
        if isinstance(module, (W4A4Linear, QwTCompensation)):
            for p in module.parameters():
                quant_param_ids.add(id(p))
            for b in module.buffers():
                quant_param_ids.add(id(b))

    for p in target.parameters():
        if id(p) not in quant_param_ids:
            other_bytes += p.numel() * p.element_size()
    for b in target.buffers():
        if id(b) not in quant_param_ids:
            other_bytes += b.numel() * b.element_size()

    to_mb = 1 / (1024 * 1024)
    return {
        "int4_weight_MB": round(int4_bytes * to_mb, 2),
        "scale_MB": round(scale_bytes * to_mb, 2),
        "qwt_MB": round(qwt_bytes * to_mb, 2),
        "other_param_MB": round(other_bytes * to_mb, 2),
        "total_MB": round((int4_bytes + scale_bytes + qwt_bytes + other_bytes) * to_mb, 2),
    }


def _make_dummy_input(
    num_views: int = 4,
    img_h: int = 518,
    img_w: int = 518,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(1, num_views, 3, img_h, img_w, device=device, dtype=dtype)


@torch.no_grad()
def run_fp_vs_w4a4_compare(
    pretrained_model_path: str,
    deploy_dir: str,
    device: torch.device,
    *,
    use_qwt: bool = True,
    num_views: int = 4,
    img_h: int = 518,
    img_w: int = 518,
    seed: int = 42,
    warmup: int = 5,
    iters: int = 20,
    aggregator_only: bool = True,
):
    """FP32 全精度基准 vs W4A4 对比: 延时 + 显存 + 模型大小。

    aggregator_only=True (默认): 只测主干 aggregator, 不跑 DPT heads。
    """

    fp_dtype = torch.float32
    images = _make_dummy_input(num_views, img_h, img_w, device, fp_dtype, seed)
    fwd_fn = forward_aggregator_only if aggregator_only else inference
    scope_label = "aggregator" if aggregator_only else "full model"
    report: dict = {
        "scope": scope_label,
        "num_views": num_views,
        "img_h": img_h,
        "img_w": img_w,
    }

    # ---- FP32 全精度基准 ----
    print(f"\n=== 加载 FP32 基准模型 ({scope_label}) ===")
    torch.cuda.reset_peak_memory_stats(device)
    model_fp = (
        VGGT.from_pretrained(pretrained_model_path)
        .to(device=device, dtype=fp_dtype)
        .eval()
    )
    fp_model_mem = torch.cuda.max_memory_allocated(device) / 1e6

    # FP 模型大小 (aggregator 部分)
    fp_size = compute_model_size(model_fp, aggregator_only=aggregator_only)
    report["fp_model_size"] = fp_size

    torch.cuda.reset_peak_memory_stats(device)
    _ = fwd_fn(model_fp, images)
    fp_infer_mem = torch.cuda.max_memory_allocated(device) / 1e6

    report["fp_model_mem_MB"] = round(fp_model_mem, 1)
    report["fp_infer_peak_MB"] = round(fp_infer_mem, 1)

    # FP 延时
    for _ in range(warmup):
        fwd_fn(model_fp, images)
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        fwd_fn(model_fp, images)
    torch.cuda.synchronize(device)
    fp_lat = (time.perf_counter() - t0) / iters * 1000
    report["fp_latency_ms"] = round(fp_lat, 2)
    report["fp_baseline_dtype"] = str(fp_dtype).replace("torch.", "")

    del model_fp
    gc.collect()
    torch.cuda.empty_cache()

    # ---- W4A4 模型 ----
    print(f"\n=== 加载 W4A4 INT GEMM 模型 ({scope_label}) ===")
    torch.cuda.reset_peak_memory_stats(device)
    model_q, meta = load_w4a4_deployed_model(
        pretrained_model_path, deploy_dir, device=str(device), use_qwt=use_qwt,
    )
    q_model_mem = torch.cuda.max_memory_allocated(device) / 1e6

    # W4A4 模型大小 (aggregator 部分)
    q_size = compute_model_size(model_q, aggregator_only=aggregator_only)
    report["w4a4_model_size"] = q_size

    torch.cuda.reset_peak_memory_stats(device)
    _ = fwd_fn(model_q, images)
    q_infer_mem = torch.cuda.max_memory_allocated(device) / 1e6

    report["w4a4_model_mem_MB"] = round(q_model_mem, 1)
    report["w4a4_infer_peak_MB"] = round(q_infer_mem, 1)

    # W4A4 延时
    for _ in range(warmup):
        fwd_fn(model_q, images)
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        fwd_fn(model_q, images)
    torch.cuda.synchronize(device)
    q_lat = (time.perf_counter() - t0) / iters * 1000
    report["w4a4_latency_ms"] = round(q_lat, 2)
    report["speedup"] = round(fp_lat / q_lat, 2) if q_lat > 0 else 0
    report["meta"] = {k: v for k, v in meta.items() if k != "deploy_config"}

    del model_q
    gc.collect()
    torch.cuda.empty_cache()

    return report


def print_report(report: dict):
    scope = report.get("scope", "aggregator")
    print("\n" + "=" * 70)
    print(f"  W4A4 INT GEMM vs FP32 对比  [{scope}]")
    print(f"  输入: {report['num_views']} views × {report['img_h']}×{report['img_w']}")
    print("=" * 70)

    # ---- 模型大小 ----
    fp_size = report.get("fp_model_size", {})
    q_size = report.get("w4a4_model_size", {})
    if fp_size and q_size:
        fp_total = fp_size.get("total_MB", 0)
        q_total = q_size.get("total_MB", 0)
        print(f"\n--- 模型大小 ({scope}) ---")
        print(f"{'':>28} {'FP32':>10} {'W4A4':>10}")
        print(f"{'INT4 权重 (int8 存储)':>28} {'--':>10} {q_size.get('int4_weight_MB', 0):>9.2f}M")
        print(f"{'量化 scale':>28} {'--':>10} {q_size.get('scale_MB', 0):>9.2f}M")
        print(f"{'QwT 补偿':>28} {'--':>10} {q_size.get('qwt_MB', 0):>9.2f}M")
        print(f"{'其他参数 (LN/LS/pos/...)':>28} {fp_size.get('other_param_MB', fp_total):>9.2f}M {q_size.get('other_param_MB', 0):>9.2f}M")
        print(f"{'合计':>28} {fp_total:>9.2f}M {q_total:>9.2f}M  ({q_total/fp_total:.2f}x)" if fp_total > 0 else "")

    # ---- 显存 & 延时 ----
    print(f"\n--- 显存 & 延时 ---")
    print(f"{'指标':<30} {'FP32':>12} {'W4A4':>12} {'比值':>8}")
    print("-" * 62)

    fp_mem = report.get("fp_model_mem_MB", 0)
    q_mem = report.get("w4a4_model_mem_MB", 0)
    if fp_mem:
        print(f"{'模型加载显存 (MB)':<30} {fp_mem:>12.1f} {q_mem:>12.1f} {q_mem/fp_mem:>7.2f}x")

    fp_peak = report.get("fp_infer_peak_MB", 0)
    q_peak = report.get("w4a4_infer_peak_MB", 0)
    if fp_peak:
        print(f"{'推理峰值显存 (MB)':<30} {fp_peak:>12.1f} {q_peak:>12.1f} {q_peak/fp_peak:>7.2f}x")

    fp_lat = report.get("fp_latency_ms", 0)
    q_lat = report.get("w4a4_latency_ms", 0)
    print(f"{'推理延时 (ms)':<30} {fp_lat:>12.2f} {q_lat:>12.2f} {report.get('speedup', 0):>7.2f}x")

    meta = report.get("meta", {})
    if meta:
        print(f"\nGEMM backend: {meta.get('gemm_backend', 'N/A')}")
        print(f"W4A4 layers: {meta.get('w4a4_layers', 'N/A')}")
        print(f"QwT enabled: {meta.get('qwt_enabled', 'N/A')}/{meta.get('qwt_total', 'N/A')}")
    print("=" * 70)


# ================================================================
# 5. CLI
# ================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VGGT W4A4 部署 (真正的 INT GEMM via torch._int_mm)"
    )
    parser.add_argument(
        "--model", type=str,
        default=os.environ.get("VGGT_MODEL_PATH", "/root/autodl-tmp/hf_hub/models--facebook--VGGT-1B"),
    )
    parser.add_argument("--deploy-dir", type=str, default=None, help="默认: <repo>/outputs/w4a4_deploy")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-qwt", action="store_true", help="不挂载 QwT 补偿")

    parser.add_argument("--dummy", action="store_true", help="加载模型 + dummy 推理")
    parser.add_argument(
        "--compare-fp", action="store_true",
        help="FP32 全精度基准 vs W4A4 对比（延时 / 显存 / 模型大小）",
    )

    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--img-h", type=int, default=518)
    parser.add_argument("--img-w", type=int, default=518)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--full-model", action="store_true",
        help="测整网 (含 heads); 默认只测 aggregator 主干",
    )
    parser.add_argument("--save-json", type=str, default=None)

    parser.add_argument(
        "--profile-ops", action="store_true",
        help="加载后对一次 forward 做 torch.profiler，打印各 aten 算子延时",
    )
    parser.add_argument("--profile-warmup", type=int, default=3, help="profiler 前 warmup 次数")
    parser.add_argument("--profile-rows", type=int, default=50, help="汇总表打印行数上限")
    parser.add_argument(
        "--profile-chrome", type=str, default=None,
        help="将 trace 写入 json，用 chrome://tracing 打开",
    )
    parser.add_argument(
        "--profile-sort", type=str, default="cuda_time_total",
        help="key_averages 排序列，如 cuda_time_total、self_cuda_time_total、cpu_time_total",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    deploy_dir = args.deploy_dir or os.path.join(repo_root, "outputs", "w4a4_deploy")

    if args.compare_fp:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA 不可用")
        dev = torch.device(args.device or "cuda")

        agg_only = not args.full_model  # 默认 True: 只测主干
        print(f"[模式] {'仅 aggregator 主干' if agg_only else '整网 (含 heads)'}")

        report = run_fp_vs_w4a4_compare(
            args.model, deploy_dir, dev,
            use_qwt=not args.no_qwt,
            num_views=args.num_views,
            img_h=args.img_h, img_w=args.img_w,
            seed=args.seed, warmup=args.warmup, iters=args.iters,
            aggregator_only=agg_only,
        )
        print_report(report)

        if args.save_json:
            # 清理不可序列化的值
            def _clean(obj):
                if isinstance(obj, dict):
                    return {k: _clean(v) for k, v in obj.items()}
                if isinstance(obj, float) and (obj != obj):  # NaN
                    return None
                return obj

            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(_clean(report), f, indent=2, ensure_ascii=False)
            print(f"已写入 {args.save_json}")
        return

    # ---- 默认: 加载 + dummy 推理 ----
    model, meta = load_w4a4_deployed_model(
        args.model, deploy_dir,
        device=args.device, use_qwt=not args.no_qwt,
    )
    print("\nW4A4 加载完成:")
    print(json.dumps({k: v for k, v in meta.items() if k != "deploy_config"}, indent=2))

    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    x = torch.rand(1, args.num_views, 3, args.img_h, args.img_w, device=dev, dtype=dt)

    if args.profile_ops:
        run_op_profile(
            model,
            x,
            full_model=args.full_model,
            warmup=args.profile_warmup,
            row_limit=args.profile_rows,
            chrome_trace_path=args.profile_chrome,
            sort_by=args.profile_sort,
        )

    if args.dummy:
        out = inference(model, x)
        print("output keys:", list(out.keys()))
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape} {v.dtype}")


if __name__ == "__main__":
    main()

# # 默认只 profile aggregator（与 --compare-fp 默认一致）
# CUDA_VISIBLE_DEVICES=1 PYTHONPATH=PTQ python deployment/w4a4_deploy_int4gemm.py --profile-ops --num-views 4

# # 带 dummy 再跑一遍整网 forward（会多一次推理，用于打印 output keys）
# ... --profile-ops --dummy --num-views 4

# # 整网（含 heads）
# ... --profile-ops --full-model

# # 导出 Chrome trace（chrome://tracing 打开 json）
# ... --profile-ops --profile-chrome /tmp/w4a4_trace.json

# # 多看几行 / 换排序（默认按 cuda_time_total）
# ... --profile-ops --profile-rows 80 --profile-sort self_cuda_time_total