# TensorRT / ONNX path for VGGT

This folder documents how to move from **PyTorch fake-quant** (see `benchmark_quant_deploy.py` and `compare_quant_fp.py`) toward a **TensorRT** engine. It is **not** a one-click translation of `param/vggt/channelwise_8scan_w4a8.json` into INT4 Tensor Cores: that JSON stores custom block-structured `w_interval` / `a_interval` tensors that must be mapped (or approximated) to TensorRT-friendly scales and Q/DQ nodes.

## What you have today

- **PyTorch W4A8**: Rounding + `F.linear` in floating point; metrics from `compare_quant_fp.py` reflect this path.
- **ONNX export**: `mv_recon/export_vggt_onnx.py` writes either the **Aggregator** backbone (recommended first) or a **world_points** slice of full VGGT.

## Prerequisites

- NVIDIA GPU driver matching your CUDA toolkit.
- **TensorRT** tarball or `tensorrt` Python wheel aligned with your CUDA version.
- **ONNX** graph that matches the ops TensorRT supports on your SM (Blackwell / Ada / etc.).

## 1) Export ONNX

From the Pi3-evaluation repository root (set `VGGT_MODEL_PATH` if the default snapshot path differs):

```bash
# Smaller trace: 2 views (faster export); increase --num-views for production shapes
python mv_recon/export_vggt_onnx.py \
  --target aggregator \
  --out /tmp/vggt_aggregator_last.onnx \
  --num-views 2 \
  --img-h 518 --img-w 518 \
  --opset 17
```

Optional full-model tensor output (heavier):

```bash
python mv_recon/export_vggt_onnx.py \
  --target world_points \
  --out /tmp/vggt_world_points.onnx \
  --num-views 2
```

If the dynamo-based exporter fails on your PyTorch build, omit `--dynamo` (default) to use the legacy ONNX exporter.

## 2) Build a TensorRT engine (FP16 first)

Validate the toolchain with **FP16** before attempting INT4. Example using `trtexec` (paths vary by install):

```bash
trtexec \
  --onnx=/tmp/vggt_aggregator_last.onnx \
  --saveEngine=/tmp/vggt_agg_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:4096
```

Inspect supported tactics / layers if build fails (unsupported ops, dynamic axes, etc.).

## 3) INT4 and your PTQ JSON

TensorRT INT4 (where available) typically expects **quantized weights + explicit scales/zero-points** in a form compatible with `IQuantizeLayer` / Q/DQ ONNX patterns. Your PTQ code uses **per-block** intervals (`n_V`, `n_H`, `n_a`), not necessarily a single per-channel scale per GEMM.

Reasonable next steps:

1. **POC**: Export a **single** `PTQSLBatchingQuantLinear` (or a tiny MLP) with fixed scales derived from one row of `w_interval`, build INT4/INT8 TRT, and compare numerically to PyTorch `quant_forward`.
2. **Production**: Either approximate PTQ scales as per-channel TRT scales (accuracy trade-off) or implement a **custom TRT plugin** / fused CUDA kernel that matches `quant_forward` exactly.

The JSON file is huge; treat it as a **calibration artifact** to compute scales for a separate export pipeline, not as a TensorRT-native format.

## 4) Related scripts

| Script | Role |
|--------|------|
| `mv_recon/benchmark_common.py` | Shared load / benchmark / optional QwT checkpoint |
| `mv_recon/benchmark_quant_deploy.py` | Single-run deploy metrics |
| `mv_recon/compare_quant_fp.py` | FP vs W4A8 (JSON) side-by-side |
| `mv_recon/export_vggt_onnx.py` | ONNX export for TRT |

## QwT compensation

Training-time **QwT** (`AttnQwT` / `MlpQwT`) uses **FP16** for low-rank factors `A`, `B` in eval (see `mv_recon/ptq.py`). Those tensors are **not** in the calibration JSON; restore them with `--checkpoint` on the benchmark scripts if you need parity with a saved training run.
