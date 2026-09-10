# Multi-view Reconstruction and TAPTQ

从 `Pi3-evaluation` 根目录运行：

```bash
cd /data/workspace/TAPTQ/Pi3-evaluation
```

## 主流程

### 普通多视图评估

```bash
python mv_recon/eval.py
```

### TAPTQ 完整流程

`configs/evaluation/mv_recon.yaml` 默认使用固定的 `Pi3-evaluation/data/dtu_8`（8 scans × 10 frames）作为校准和补偿数据集，正式评估集仍由 `eval_datasets` 单独控制。

```bash
# 校准、量化评估、QwT 补偿、补偿后评估并保存
python mv_recon/taptq.py ++mode=e2e

# 只校准并保存量化参数；通常只需执行一次
python mv_recon/taptq.py \
  ++mode=calib \
  ++ckpt.path=param/vggt/20scan_w4a8.json \
  ++ckpt.fmt=json

# 复用已有校准参数，比较不同 QwT 补偿策略
python mv_recon/taptq.py \
  ++mode=compensate_eval \
  ++ckpt.path=param/vggt/20scan_w4a8.json \
  ++compensate.strategy=module \
  ++compensate.select_metric=hessian \
  ++compensate.keep_ratio=0.5 \
  ++save_suffix=hess_50
```

可用模式：

- `calib`：校准并保存，不评估、不补偿。
- `calib_compensate`：校准、QwT 补偿并保存，不评估。
- `compensate_eval`：加载已有校准参数，补偿后评估；适合重复对比实验。
- `test`：加载已有校准参数，直接评估。
- `e2e`：校准、量化评估、补偿、补偿后评估并保存。

### 7-Scenes 消融

```bash
bash mv_recon/run_ablation_7sd.sh base
bash mv_recon/run_ablation_7sd.sh tail_ratio
bash mv_recon/run_ablation_7sd.sh rank
```

脚本会自动把当前脚本所在的 `Pi3-evaluation` 目录作为项目根，也可以通过环境变量覆盖：

```bash
GPU=0 CKPT=param/vggt/20scan_w4a8.txt bash mv_recon/run_ablation_7sd.sh tail_ratio
```

## 采样

已有的 `datasets/seq-id-maps/` 包含默认采样结果，通常不需要重新生成。只有修改帧采样策略时才运行：

```bash
python mv_recon/sampling.py
```

## 部署与对比工具

这些脚本互相独立，不参与 TAPTQ 校准主流程：

```bash
# FP / PyTorch fake-quant 延迟、显存、模型大小
python mv_recon/benchmark_quant_deploy.py --help
python mv_recon/compare_quant_fp.py --help

# 导出 ONNX
python mv_recon/export_vggt_onnx.py --help

# 导出整型权重
python mv_recon/save_vggt_w4_int_weights.py --help
```

`benchmark_common.py` 是上述 benchmark 的共享后端，依赖 `ptq.py`，不能移走或删除。

## 目录约定

当前主目录只保留活动入口：

```text
mv_recon/
├── eval.py
├── sampling.py
├── taptq.py
├── ptq.py                    # benchmark 后端依赖
├── baseline_quant.py         # VGGT RTN / RepQ / ERQ unified baseline entry
├── run_ablation_7sd.sh
├── benchmark_common.py
├── benchmark_quant_deploy.py
├── compare_quant_fp.py
├── export_vggt_onnx.py
├── save_vggt_w4_int_weights.py
├── utils.py
└── legacy/                   # 历史 PTQ/GPTQ 分支和旧测试脚本
```

`legacy/` 中的脚本保留用于复现实验，但不再作为默认入口：

- `ptq4pi3.py`
- `ptq_old.py`
- `ptq_gptq.py`
- `gptq4pi3.py`
- `demo_int_deploy_skeleton.py`
- `test.sh`

## 输出

TAPTQ 评估通常写入：

```text
outputs/mv_recon/<dataset>/
outputs/mv_recon/tre_diff/all_metrics.csv
```

其中包含每个序列的 `pred.ply`、`gt.ply`、输入图像和 CSV 指标。模型权重、数据集、量化参数和运行产物不纳入 GitHub 源码仓库。
