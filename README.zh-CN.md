# TAPTQ — Tail-Aware Post-Training Quantization for 3D Reconstruction

[English](README.md) | 简体中文

TAPTQ 是一套面向 3D 几何模型（以 VGGT 为代表的多视图重建 Transformer）的
**尾部分布感知训练后量化（Post-Training Quantization）**框架，并配套一套统一的
多视图重建（multi-view reconstruction, MV-Recon）评测与对比基线。

核心特点：

- **统一量化层**：`PTQ/quant_layers` 同时支持 W4A8 / W6A6 / W8A8，并可在
  per-tensor / per-channel / **per-output-channel（channel-wise）** 多种量化粒度间切换。
- **QwT 动态补偿**：`mv_recon/taptq.py` 在冻结校准参数之上做模块级尾部补偿
  （tail-aware compensation），缓解低比特下的几何退化。
- **统一评测协议**：`mv_recon/eval.py` 对 7-Scenes（dense, `kf=40`）与 ETH3D
  （13 sequences, `kf=5`）使用一致的 frozen frame-map、Sim(3)+ICP 点云对齐，以及
  Acc. / Comp. / N.C. 计算，保证不同方法公平可比。
- **多基线对齐**：`mv_recon/external_baselines.py` 与 `quant_layers` 内集成了
  RTN、PTQ4ViT、RepQ-ViT、ERQ、GPTQ、SmoothQuant、SVDQuant 与 QuantVGGT 的
  同协议实现。

---

## 仓库结构

```
TAPTQ/
├── Pi3-evaluation/            # 主代码（统一评测 + TAPTQ 量化/补偿）
│   ├── mv_recon/              # 多视图重建评测与 TAPTQ 主流程
│   │   ├── taptq.py          # 校准 / 量化评估 / QwT 补偿 / e2e
│   │   ├── eval.py           # 统一 Acc./Comp./N.C. 评测
│   │   ├── ptq.py            # 量化模型构建
│   │   ├── baseline_quant.py # 各基线量化封装
│   │   └── external_*.py     # 外部基线 / 基础模型对接
│   ├── PTQ/                   # 量化层与校准工具
│   │   ├── quant_layers/     # linear/conv 与 baseline 量化器
│   │   ├── vggt/             # VGGT 模型接入
│   │   └── utils/            # Hessian 校准、baseline 校准
│   ├── configs/              # Hydra 配置（保持入库）
│   ├── datasets/             # 预处理脚本与 seq-id maps
│   └── deployment/           # W4A8(int8) / W8A8(triton) 部署与验证
├── projects/QuantVGGT/        # QuantVGGT 同协议评测对接
├── third_party/              # 外部基线源码（各自独立 git，默认不入库）
├── vendor/                   # 第三方数据/代码（默认不入库）
├── doc/                      # 实验计划、论文草稿、结果清单与 manifest
└── scripts/                  # 批量实验脚本
```

> 说明：`third_party/` 与 `vendor/` 内为各自带独立 git 仓库的外部基线，已在
> `.gitignore` 中排除，克隆后需自行放置；它们不是 TAPTQ 主体代码的一部分。

---

## 快速开始

```bash
cd Pi3-evaluation

# 1) 校准并保存量化参数（通常只需一次）
python mv_recon/taptq.py \
  ++mode=calib \
  ptq.bit=[8,8] ptq.search_mode=ternary ptq.linear_channelwise=true \
  'optim_datasets=[DTU_train_8]' \
  ckpt.fmt=pt ckpt.path=/path/to/ckpt.pt

# 2) 复用已有校准，做 QwT 补偿后评测（适合反复对比）
python mv_recon/taptq.py \
  ++mode=compensate_eval \
  'eval_datasets=[ETH3D]' \
  ptq.bit=[8,8] ptq.search_mode=ternary ptq.linear_channelwise=true \
  'optim_datasets=[DTU_train_8]' \
  ckpt.fmt=pt ckpt.path=/path/to/ckpt.pt \
  ++compensate.strategy=module ++compensate.tail_ratio=0.01 \
  ++compensate.tau_thr=0.005 ++compensate.rank=256

# 3) 仅加载量化参数做纯量化评估（补偿前对照）
python mv_recon/taptq.py ++mode=test 'eval_datasets=[ETH3D]' ...
```

可用 `mode`：

| mode | 行为 |
|---|---|
| `calib` | 校准并保存量化参数，不评估、不补偿 |
| `calib_compensate` | 校准 + QwT 补偿并保存，不评估 |
| `compensate_eval` | 加载已有参数，补偿后评估 |
| `test` | 加载已有参数，直接量化评估 |
| `e2e` | 校准 → 量化评估 → 补偿 → 补偿后评估并保存 |

---

## 量化粒度 / 超参约定

ETH3D 主表采用各 bit-width 的**选定配置**（超参列于论文附录，不在此展开粒度术语）：

| 位宽 | 参考配置 `(ρ, τ, r)` | 选定配置 `(ρ, τ, r)` |
|---|---|---|
| W4A8 | (0.1, 0.007, 256) | (0.01, 0.005, 128) |
| W6A6 | (0.1, 0.007, 16)  | (0.01, 0.005, 256) |
| W8A8 | (0.1, 0.007, 16)  | (0.01, 0.005, 256) |

> 历史非协议匹配的 W6A6 量化记录已废弃，当前 W6A6 / W8A8 channel-wise 结果均来自
> 与 W8A8 同一链路（`DTU_train_8 + ternary + channel-wise`）重新校准的 checkpoint。

---

## 数据集与产物

- 校准集：`DTU_train_8`（8 scans × 10 frames，4096-token 预算）。
- 评测集：7-Scenes dense（`kf=40`）、ETH3D（13 sequences，`kf=5`，Pi3 兼容冻结帧表）。
- 模型权重、大规模点云与 `.pth/.pt/.ply/.npy` 等数据文件已被 `.gitignore` 排除，
  评测产物与中间数据请放置于本地或外部存储（如 CephFS），不要入库。

---

## 结果

主实验结果（含 ETH3D 选定配置、各基线同协议对齐）记录在
`doc/TMM_EXPERIMENTS_RESTRUCTURE_DRAFT.tex` 与 `doc/tmm_*_manifest.json`。
每个实验的（配置 → 校准协议 → 结果产物）映射由 manifest 完整留存，便于复现。
