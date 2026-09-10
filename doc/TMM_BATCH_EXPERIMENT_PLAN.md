# TMM 转投版实验批次计划

> 本文档针对当前确定的一批实验，按论文写作逻辑组织实验表格、实验矩阵、复现状态和执行顺序。
>
> **汇报口径（当前执行优先级）**：正文和阶段性实验汇报以 `7Scenes-dense` 为主结果集；`DTU` 作为校准协议和补充验证集，`ETH3D`、`Co3Dv2` 仅在数据完整且协议统一后加入。所有主表先按 `7Scenes-dense` 排序，不能用 DTU 数字替代 7Scenes 主结果。
>
> 当前确定的任务与对象：
>
> - 点云重建测试集：`7Scenes-dense`、`ETH3D`、`DTU`（`/data/workspace/TAPTQ/data/dtu`）；
> - 相机参数预测测试集：`Co3Dv2`；
> - 基础模型：`VGGT`、`Pi3`、`Dust3R`、`MASt3R`；
> - 量化/基线方法：`RTN`、`PTQ4ViT`、`ERQ`、`RepQ`、`GPTQ`、`QuantVGGT`、`TAPTQ`；
> - 需要额外补充 channel-wise 结果的方法：`ERQ`、`RepQ`、`GPTQ`、`TAPTQ`。

---

## 1. 实验总目标

这批实验最终需要支撑 TMM 版本的四条结论：

1. TAPTQ 在 3D 几何基础模型上的低比特量化结果优于或至少具有竞争力；
2. TAPTQ 的收益不仅来自普通 PTQ，而来自 progressive calibration、ternary search 和 TRE-guided QwT compensation 的组合；
3. TAPTQ 对不同模型、数据集、位宽和量化粒度具有一定泛化能力；
4. 方法的精度收益与校准时间、显存、推理 latency、模型大小和补偿开销之间的关系是可量化的。

当前不应把所有模型和所有 baseline 机械地组成一个巨大矩阵。实验分为三层：

```text
第一层：所有模型的 FP baseline
        ↓
第二层：VGGT 上完整量化 baseline 对比
        ↓
第三层：Pi3、Co3Dv2、Dust3R、MASt3R 的跨模型/跨任务扩展
```

这样可以保证主文有清晰主线，同时把暂时无法公平复现的设置放入补充材料或单独标注。

---

## 2. 当前数据、代码和复现状态盘点

### 2.1 已确认存在的数据

| 数据 | 本地路径 | 当前状态 | 用途 |
|---|---|---|---|
| 7Scenes | `/data/workspace/TAPTQ/data/7scenes` | 已存在，约 24G | 点云重建测试 |
| DTU test | `/data/workspace/TAPTQ/data/dtu` | 已存在，约 8.6G | 点云重建测试 |
| DTU calibration/compensation set | `/data/workspace/TAPTQ/Pi3-evaluation/data/dtu_8` | 已存在，约 858M，8 scans | TAPTQ 校准与补偿 |
| ETH3D | 待确认 | 当前顶层未发现对应目录 | 点云重建测试 |
| Co3Dv2 | 待确认 | QuantVGGT 有 annotation 目录，但完整数据需确认 | 相机参数预测测试 |

ETH3D 和 Co3Dv2 在正式启动前必须生成数据 manifest；不能因为代码中存在 dataset class 就认为数据已经完整可用。

### 2.2 当前已发现的量化参数或代码

| 方法/模型 | 当前发现 | 初步状态 |
|---|---|---|
| TAPTQ/VGGT | `param/vggt/20scan_*`、`8scan_*` | 有旧 checkpoint，可用于 sanity check；需要重新校准正式 TMM 参数 |
| TAPTQ/Pi3 | `param/pi3/8scan_*` | 有部分旧参数，需核对模型版本和 evaluator |
| GPTQ | `param/gptq/`、`param/gptq4pi3/`、`mv_recon/legacy/*gptq*` | 有历史产物/旧代码，需确认是否可直接复现 |
| PTQ4ViT | `configs/PTQ4ViT.py`、`mv_recon/ptq.py` | 当前主线可复用 |
| QuantVGGT | `projects/QuantVGGT/evaluation/` | 有独立 evaluator 和 calibration cache，已完成部分 H800 sanity check |
| RTN | 当前未发现明确独立入口 | 需要核对是否已有历史脚本或重新实现 |
| ERQ | 当前代码搜索未发现明确入口 | 尚未确认/需要复现 |
| RepQ | 当前代码搜索未发现明确入口 | 尚未确认/需要复现 |
| Dust3R | 当前 workspace 未发现明确实现/权重入口 | 尚未接入 |
| MASt3R | 当前 workspace 未发现明确实现/权重入口 | 尚未接入 |

“有参数文件”不等于“可作为最终结果”。每个旧 checkpoint 必须记录生成代码 commit、模型版本、校准集、位宽和量化粒度。

---

## 3. 论文正文实验表的推荐顺序

正文建议按以下顺序排布，不按代码目录或方法名称排列。

| 表格 | 论文位置 | 表格主题 | 回答的问题 |
|---|---|---|---|
| Table 1 | Experimental Setup | 任务、模型、数据、指标和硬件协议 | 实验是否公平、可复现？ |
| Table 2 | **Primary FP Results (7Scenes-dense)** | VGGT、Pi3 及可复现模型的 FP 结果，主列为 7Scenes-dense | 基础模型在主结果集上的能力差异是什么？ |
| Table 3 | **Primary Quantization Results (7Scenes-dense)** | VGGT 上 RTN/PTQ4ViT/ERQ/RepQ/GPTQ/QuantVGGT/TAPTQ | TAPTQ 是否优于量化 baseline？ |
| Table 4 | Cross-Model Quantization | Pi3 上可复现方法与 TAPTQ，主列为 7Scenes-dense | 方法能否迁移到 Pi3？ |
| Table 5 | Component Ablation | calibration、ternary、TRE、QwT 的逐步加入，主列为 7Scenes-dense | TAPTQ 的收益来自哪些组件？ |
| Table 6 | Channel-Wise Quantization | ERQ、RepQ、GPTQ、TAPTQ 的额外 channel-wise 结果，主列为 7Scenes-dense | 结论是否依赖 per-tensor 量化？ |
| Table 7 | Calibration and Search Efficiency | dtu_8/kf5、exhaustive/ternary；精度以 7Scenes-dense 为主 | 校准效率和搜索方法的精度—时间权衡如何？ |
| Table 8 | Deployment Cost | H800 latency、memory、size、overhead | 量化结果是否具有部署价值？ |
| Table 9 | Co3Dv2 Camera Prediction | 相机参数预测结果 | 方法能否迁移到相机任务？ |
| Table 10 | Cross-Architecture Generalization | VGGT、Pi3、Dust3R、MASt3R 综合结果，优先报告 7Scenes-dense | 方法是否具有跨架构泛化能力？ |

如果 TMM 正文篇幅有限，`Table 4`、`Table 7` 和 `Table 10` 可以压缩到 Supplement，但 `Table 1–3、5、6、8、9` 应优先保留。

---

## 4. Table 1：Experimental Setup and Protocol

该表只交代实验边界，不填方法优劣结论。

| Task | Test dataset | Model | Input protocol | Output | Metrics | Hardware |
|---|---|---|---|---|---|---|
| Point cloud reconstruction | 7Scenes-dense | VGGT/Pi3/Dust3R/MASt3R | 固定视角数、分辨率、采样 | world point map / point cloud | Acc、Comp、NC | H800 |
| Point cloud reconstruction | ETH3D | VGGT/Pi3/Dust3R/MASt3R | 同一输入规范，记录模型原生适配 | point cloud | Acc、Comp、NC | H800 |
| Point cloud reconstruction | DTU | VGGT/Pi3/Dust3R/MASt3R | `/data/workspace/TAPTQ/data/dtu` | point cloud | Acc、Comp、NC | H800 |
| Camera prediction | Co3Dv2 | VGGT | Official VGGT 10-frame protocol | camera pose / intrinsics / extrinsics | relative rotation/translation error | H800 |

表注必须固定：

- 输入图像尺寸；
- 视角数和帧采样规则；
- 是否使用相机先验；
- 是否使用 Umeyama / Sim(3) / ICP；
- 是否使用模型自带后处理；
- QuantVGGT 原 evaluator 与统一 evaluator 的差异；
- Dust3R/MASt3R 的坐标系、尺度和输出转换方式。

---

## 5. Table 2：FP Baselines Across Models and Tasks

先完成所有基础模型的 FP 结果，作为后续量化和跨模型比较的参照。

### 5.1 点云重建 FP 表

| Model | **7Scenes-dense Acc/Comp/NC (primary)** | DTU Acc/Comp/NC (supplement) | ETH3D Acc/Comp/NC (if available) | Latency | Peak memory |
|---|---|---|---|---:|---:|
| VGGT |  |  |  |  |  |
| Pi3 |  |  |  |  |  |
| Dust3R |  |  |  |  |  |
| MASt3R |  |  |  |  |  |

### 5.2 Co3Dv2 相机参数 FP 表

| Model | Rotation error | Translation error | Camera / pose metric | Inference latency | Peak memory |
|---|---:|---:|---:|---:|---:|
| VGGT |  |  |  |  |  |
| Pi3 |  |  |  |  |  |
| Dust3R |  |  |  |  |  |
| MASt3R |  |  |  |  |  |

Dust3R 和 MASt3R 第一阶段只做 FP，不立即要求量化。它们的 FP 结果必须先完成统一输出转换和评估。

---

## 6. Table 3：VGGT Main Quantization Comparison

VGGT 是当前代码最完整、也是 TMM 主要量化主线。该表建议作为正文核心结果表。

固定：

- 模型：VGGT；
- 主量化设置：W4A8；
- 校准池：`Pi3-evaluation/data/dtu_8`；
- 正式校准 manifest：固定 8-scan 或论文最终选定的 progressive subset；
- 测试集：7Scenes-dense、ETH3D、DTU；
- 统一 evaluator。

| Method | W/A | Granularity | Compensation | **7Scenes-dense (primary)** | DTU (supplement) | ETH3D (if available) |
|---|---|---|---|---|---|---|
| FP VGGT | FP | – | – | Acc/Comp/NC | Acc/Comp/NC | Acc/Comp/NC |
| RTN | W4A8 | Per-tensor | – |  |  |  |
| PTQ4ViT | W4A8 | Per-tensor | – |  |  |  |
| ERQ | W4A8 | Per-tensor | – |  |  |  |
| RepQ | W4A8 | Per-tensor | – |  |  |  |
| GPTQ | W4A8 | Per-tensor | – |  |  |  |
| QuantVGGT | W4A8 | Official | Official |  |  |  |
| TAPTQ | W4A8 | Per-tensor | QwT |  |  |  |

建议同时保留：

- Acc-mean / Acc-med；
- Comp-mean / Comp-med；
- NC-mean / NC-med；
- 相对 FP 的变化；
- calibration time；
- model/checkpoint size。

如果某 baseline 只能在上游 evaluator 下运行，表格必须增加 `Evaluator` 列，不能和统一 evaluator 数字无标记混排。

---

## 7. Table 4：Pi3 Cross-Model Quantization

Pi3 作为第二个可量化 3D geometry model，验证 TAPTQ 是否依赖 VGGT 特定结构。

| Method | W/A | Granularity | 7Scenes-dense | ETH3D | DTU | Calibration time |
|---|---|---|---|---|---|---:|
| FP Pi3 | FP | – |  |  |  | – |
| RTN / PTQ4ViT | W4A8 | Per-tensor |  |  |  |  |
| GPTQ | W4A8 | Per-tensor |  |  |  |  |
| TAPTQ | W4A8 | Per-tensor |  |  |  |  |
| TAPTQ | W4A8 | Channel-wise |  |  |  |  |

Pi3 任务需要先确认：

- 当前 `param/pi3` 参数对应哪个 Pi3 代码版本；
- Pi3 的量化包装层和 VGGT 是否共用；
- point map 输出是否需要单独适配；
- Co3Dv2 camera head 是否可复用；
- 现有 `gptq4pi3` 是否是可运行版本。

---

## 8. Table 5：TAPTQ Component Ablation

按方法章节的贡献顺序做逐步消融，而不是只比较最终方法。

| Setting | Progressive calibration | Ternary search | TRE selection | QwT compensation | 7Scenes | ETH3D | DTU | Calibration time |
|---|---|---|---|---|---|---|---|---:|
| FP / PTQ baseline | – | – | – | – |  |  |  |  |
| + progressive calibration | Yes | – | – | – |  |  |  |  |
| + ternary search | Yes | Yes | – | – |  |  |  |  |
| + TRE selection | Yes | Yes | Yes | – |  |  |  |  |
| Full TAPTQ | Yes | Yes | Yes | Yes |  |  |  |  |

所有行必须固定：

- 模型；
- W/A；
- evaluator；
- 校准候选池；
- 最终校准样本数量；
- compensation budget。

---

## 9. Table 6：Per-Tensor vs Channel-Wise

用户明确要求 `ERQ`、`RepQ`、`GPTQ`、`TAPTQ` 额外补 channel-wise 结果。本表只放 W4A8 主设置，其他位宽放补充材料。

### 9.1 VGGT channel-wise 主表

| Method | Per-tensor W4A8 | Channel-wise W4A8 | Acc change | Comp change | NC change | Calibration time | Model/param overhead |
|---|---|---|---:|---:|---:|---:|---:|
| ERQ |  |  |  |  |  |  |  |
| RepQ |  |  |  |  |  |  |  |
| GPTQ |  |  |  |  |  |  |  |
| TAPTQ |  |  |  |  |  |  |  |

### 9.2 每种方法必须记录的附加信息

| Method | Channel-wise implementation | Weight granularity | Activation granularity | Checkpoint format | Shape validation |
|---|---|---|---|---|---|
| ERQ |  |  |  |  | pass/fail |
| RepQ |  |  |  |  | pass/fail |
| GPTQ |  |  |  |  | pass/fail |
| TAPTQ | `PTQ4ViT_channelwise` or final implementation |  |  | `.json/.pt` | pass/fail |

当前已有的 `channelwise_8scan_w4a8.json` 与量化包装层曾出现 interval shape mismatch，不能直接作为最终结果。必须先完成：

1. 明确 channel-wise 维度定义，是 per-output-channel 还是 block/channel group；
2. 使用当前代码重新生成参数；
3. 加载后逐层检查 `w_interval`、`a_interval` shape；
4. 对单层、单 batch、单序列做 forward 数值检查；
5. 再进行完整测试集评估。

---

## 10. Table 7：Calibration Set and Search Efficiency

该表把论文的校准集构造和 ternary search 放在同一逻辑链中。

| Calibration pool | Selected calibration set | Search | Search rounds / eq_n | Search evaluations | Calibration time | Acc | Comp |
|---|---|---|---|---:|---:|---:|---:|
| `dtu_8` | 8 scans × 10 frames | Exhaustive | Original |  |  |  |  |
| `dtu_8` | 8 scans × 10 frames | Ternary | `search_round=1/2/3/4` |  |  |  |  |
| `dtu_8` | 8 scans × 10 frames | Ternary | Recommended |  |  |  |  |

必须同时记录：

- forward count；
- calibration peak memory；
- 每模块平均搜索时间；
- ternary 最优 interval 与 exhaustive 最优 interval 的差距；
- 搜索失败或非单峰模块数量。

---

## 11. Table 8：H800 Deployment Cost

该表报告实际 H800 上的部署代价，并明确 fake-quant 与真实 INT kernel 的区别。

| Method | Runtime path | W/A | Target | Latency mean/std | Peak allocated/reserved | Weight/model size | QwT overhead | Note |
|---|---|---|---|---:|---:|---:|---:|---|
| VGGT | PyTorch FP | FP | Full |  |  |  | – | FP reference |
| TAPTQ | PyTorch fake-quant | W4A8 | Full |  |  |  |  | Not real INT4 |
| TAPTQ | PyTorch fake-quant | W4A8 | Aggregator |  |  |  |  | Quantized scope |
| TAPTQ | PyTorch fake-quant | W6A6/W8A8 | Full |  |  |  |  |  |
| ERQ/RepQ/GPTQ | PyTorch quant | W4A8 | Full |  |  |  |  |  |
| QuantVGGT | Official runtime | W4A8 | Full |  |  |  |  | Separate evaluator |
| TAPTQ | TensorRT/ONNX | INT8/INT4 | Full |  |  |  |  | Only if available |

部署实验必须固定：

- GPU 型号；
- 输入 shape；
- batch size；
- warmup / iterations；
- AMP dtype；
- full model 或 aggregator-only；
- 是否包含数据加载和后处理。

---

## 12. Table 9：Co3Dv2 Camera Prediction

该表单独处理相机参数预测，不和点云 Acc/Comp/NC 混在一起。

### 12.1 FP cross-model camera table

| Model | Rotation error | Translation error | Camera center error | Focal/intrinsic error | Latency | Peak memory |
|---|---:|---:|---:|---:|---:|---:|
| VGGT |  |  |  |  |  |  |
| Pi3 |  |  |  |  |  |  |
| Dust3R |  |  |  |  |  |  |
| MASt3R |  |  |  |  |  |  |

### 12.2 Quantized camera table

先在 VGGT 上完成完整量化矩阵，再扩展到 Pi3；Dust3R/MASt3R 暂不强制量化。

| Model | Method | W/A | Camera head quantized? | Rotation error | Translation error | Memory |
|---|---|---|---|---:|---:|---:|
| VGGT | FP | FP | – |  |  |  |
| VGGT | RTN/PTQ4ViT | W4A8 |  |  |  |  |
| VGGT | GPTQ/QuantVGGT | W4A8 |  |  |  |  |
| VGGT | TAPTQ | W4A8 |  |  |  |  |
| VGGT | TAPTQ channel-wise | W4A8 |  |  |  |  |

必须先确认不同方法是否量化 camera head；如果量化范围不一致，必须拆表或增加 `Quantized scope` 列。

---

## 13. Table 10：Cross-Architecture Generalization

该表作为最后的总结性扩展表，重点展示 FP 和少量可复现的量化结果。

| Model | Task | FP | W4A8 per-tensor | W4A8 channel-wise | Dataset | Main observation |
|---|---|---:|---:|---:|---|---|
| VGGT | Point cloud / camera |  |  |  | 7Scenes/ETH3D/DTU/Co3Dv2 | Main model |
| Pi3 | Point cloud / camera |  |  |  | Same applicable sets | Related model |
| Dust3R | Point cloud / camera |  | N/A or pending | N/A or pending | Same applicable sets | FP external baseline |
| MASt3R | Point cloud / camera |  | N/A or pending | N/A or pending | Same applicable sets | FP external baseline |

Dust3R/MASt3R 的官方设置、输入视角数、坐标系和后处理需要单独记录；不能把不一致的原始结果直接和 VGGT 数字比较。

---

## 14. Supplement 表格顺序

| 表格 | 内容 | 对应正文 |
|---|---|---|
| Table S1 | 完整环境、模型、数据和 evaluator 配置 | Table 1 |
| Table S2 | 各数据集完整结果 | Table 2/3 |
| Table S3 | 逐场景/逐序列结果 | Table 2/3 |
| Table S4 | RTN/PTQ4ViT/ERQ/RepQ/GPTQ 的完整位宽矩阵 | Table 3 |
| Table S5 | Pi3 的完整量化矩阵 | Table 4 |
| Table S6 | calibration 4/8/20 scan 与随机/固定/progressive 对比 | Table 5/7 |
| Table S7 | ternary `search_round`、`eq_n` 和 forward count | Table 7 |
| Table S8 | TRE/MSE/Hessian/Random 不同 `keep_ratio` | Table 6 |
| Table S9 | `tail_ratio`、`tau`、rank sensitivity | Table 6 |
| Table S10 | ERQ/RepQ/GPTQ/TAPTQ channel-wise 全部结果 | Table 6 |
| Table S11 | H800 load/calibration/forward/QwT/memory 详细拆分 | Table 8 |
| Table S12 | ONNX/TensorRT/真实 INT kernel 支持状态 | Table 8 |
| Table S13 | Co3Dv2 各类别和序列结果 | Table 9 |
| Table S14 | Dust3R/MASt3R 输出适配和 FP 详细结果 | Table 10 |
| Table S15 | Failure cases、OOM、NaN/Inf、ICP failure、QwT 退化 | 全部 |
| Table S16 | Git commit、模型 hash、数据 manifest、运行命令 | 全部 |

---

## 15. 实验矩阵与优先级

### 15.1 P0：必须完成的主批次

| 批次 | 模型 | 方法 | 数据 | 设置 | 结果 |
|---|---|---|---|---|---|
| P0-1 | VGGT | FP | 7Scenes/ETH3D/DTU | FP | Table 2 |
| P0-2 | VGGT | RTN/PTQ4ViT/ERQ/RepQ/GPTQ/QuantVGGT/TAPTQ | 7Scenes/ETH3D/DTU | W4A8 per-tensor | Table 3 |
| P0-3 | VGGT | ERQ/RepQ/GPTQ/TAPTQ | 7Scenes/ETH3D/DTU | W4A8 channel-wise | Table 6 |
| P0-4 | VGGT | TAPTQ | 7Scenes/ETH3D/DTU | W8A8/W6A6/W4A8 | Table 3/7 |
| P0-5 | VGGT | TAPTQ | `Pi3-evaluation/data/dtu_8` | fixed kf5 calibration + QwT | Table 5/7 |
| P0-6 | VGGT | FP | Co3Dv2 | Camera prediction | Table 9 |
| P0-7 | VGGT | TAPTQ/主要 baselines | Co3Dv2 | W4A8 | Table 9 |
| P0-8 | Pi3 | FP/TAPTQ | 7Scenes/ETH3D/DTU | FP/W4A8 | Table 4 |
| P0-9 | All four models | FP/TAPTQ | 7Scenes/ETH3D/DTU | E1 unified 8-view protocol | Table 2/4/10 |
| P0-10 | All relevant methods | FP/quant | H800 | latency/memory/size | Table 8 |

### 15.2 P1：强烈建议完成

- Pi3 的 channel-wise `TAPTQ` 和可复现 baseline；
- ERQ/RepQ/GPTQ 在 Pi3 上的复现；
- `tail_ratio`、`keep_ratio`、`tau`、rank sensitivity；
- ternary vs exhaustive；
- Dust3R/MASt3R 的 backbone/channel-wise 适配稳定性；
- 真实 ONNX/TensorRT FP16/INT8 路径。

### 15.3 P2：时间允许时完成

- Dust3R/MASt3R 的额外位宽和更大规模量化 sweep；
- 所有 baseline 在所有模型上的完整笛卡尔积；
- QAT 对照；
- 自定义 INT4 kernel；
- 更大规模 Co3Dv2 类别 sweep。

---

## 16. TAPTQ 正式重新校准计划

当前已有旧的 `8scan_*` 和 `20scan_*` 参数只能作为历史 sanity check。TMM 正式结果必须重新校准，并且不能覆盖旧文件。

### 16.1 固定校准协议

固定校准与补偿集：

```text
/data/workspace/TAPTQ/Pi3-evaluation/data/dtu_8
```

固定 scan：

```text
scan5, scan6, scan7, scan8,
scan18, scan22, scan56, scan123
```

每个 scan 使用 `DTUTrain_8_mv-recon_seq-id-map-kf5.json` 中的 10 帧：

```text
[0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
```

正式实验必须先生成并冻结：

```text
doc/tmm_calibration_manifest.json
```

manifest 至少包括：

```json
{
  "dataset": "Pi3-evaluation/data/dtu_8",
  "scan_ids": ["scan5", "scan6", "scan7", "scan8", "scan18", "scan22", "scan56", "scan123"],
  "frame_sampling": "DTUTrain_8_mv-recon_seq-id-map-kf5.json",
  "frames_per_scan": 10,
  "calibration_and_compensation": true,
  "image_size": 518,
  "seed": 42,
  "code_commit": "",
  "model_hash": ""
}
```

本批次不采用“20 个候选样本到 8 个最终样本”的 progressive calibration 设定。`dtu_8` 本身就是固定的 8-scan 校准与补偿集；旧的 `param/vggt/8scan_*` 只能作为 sanity check，不能代替新的 TMM checkpoint。

### 16.2 必须新生成的 TAPTQ 参数

建议输出到独立目录：

```text
Pi3-evaluation/param/tmm_h800/taptq/
├── vggt/
│   ├── w8a8_per_tensor.pt
│   ├── w6a6_per_tensor.pt
│   ├── w4a8_per_tensor.pt
│   ├── w4a8_channelwise.pt
│   ├── w8a8_per_tensor.json
│   ├── w6a6_per_tensor.json
│   ├── w4a8_per_tensor.json
│   └── w4a8_channelwise.json
└── pi3/
    ├── w4a8_per_tensor.pt
    └── w4a8_channelwise.pt
```

每个参数文件必须有对应 metadata：

```text
calibration_manifest.json
config.yaml
runtime.json
git_commit.txt
model_hash.txt
calibration.log
```

`.pt` 应作为包含完整 quantizer state 的主 checkpoint；`.json` 只用于可读性和参数审计。若包含 QwT compensation，必须保存能恢复 QwT 模块的完整 `.pt`，不能只保存 quantizer intervals。

### 16.3 TAPTQ 校准矩阵

先在 VGGT 上验证校准/reload 链路，再按相同协议扩展到 Pi3、Dust3R、MASt3R：

| Job | Model | W/A | Granularity | Calibration | Compensation | 输出 |
|---|---|---|---|---|---|---|
| TQ-1 | VGGT | W4A8 | Per-tensor | fixed dtu_8 + kf5 | None | calibrated quant-only `.pt/.json` |
| TQ-2 | VGGT | W4A8 | Channel-wise | fixed dtu_8 + kf5 | None | calibrated channel-wise `.pt/.json` |
| TQ-3 | VGGT | W4A8 | Per-tensor | same checkpoint | module + TRE threshold | compensated `.pt` |
| TQ-4 | VGGT | W4A8 | Channel-wise | same channel checkpoint | module + TRE threshold | compensated `.pt` |
| TQ-5 | Pi3 | W4A8 | Per-tensor | fixed dtu_8 + kf5 | module + TRE threshold | quant-only + compensated `.pt` |
| TQ-6 | Pi3 | W4A8 | Channel-wise | fixed dtu_8 + kf5 | module + TRE threshold | quant-only + compensated `.pt` |
| TQ-7 | Dust3R | W4A8 | Per-tensor | fixed dtu_8 + kf5 | module + TRE threshold | quant-only + compensated `.pt` |
| TQ-8 | Dust3R | W4A8 | Channel-wise | fixed dtu_8 + kf5 | module + TRE threshold | quant-only + compensated `.pt` |
| TQ-9 | MASt3R | W4A8 | Per-tensor | fixed dtu_8 + kf5 | module + TRE threshold | quant-only + compensated `.pt` |
| TQ-10 | MASt3R | W4A8 | Channel-wise | fixed dtu_8 + kf5 | module + TRE threshold | quant-only + compensated `.pt` |

原则：

- TQ-1 至 TQ-4 每种模型/位宽/粒度只校准一次；
- TQ-5 至 TQ-9 只加载对应 calibrated checkpoint，不重新做 interval calibration；
- TRE/MSE/Hessian/Random 必须使用同一 quantization checkpoint 和同一评估数据；
- `tail_ratio`、`keep_ratio`、`tau`、`rank` 单独记录，不能混用。

### 16.4 正式校准前检查

在长时间 H800 校准前先执行：

1. 单 scan、单序列、单 batch forward；
2. 所有被包装模块数量检查；
3. 每个模块 `w_interval/a_interval` 非空检查；
4. channel-wise 的 interval shape 检查；
5. FP 与 quant-only 单 batch 输出统计；
6. checkpoint 保存后重新加载并再次 forward；
7. 随机抽取模块比较保存前后的量化输出；
8. 确认 calibration 数据与测试集无重叠。

### 16.5 校准输出验收标准

一次正式校准只有同时满足以下条件才算成功：

- calibration log 完整结束；
- 无 NaN/Inf；
- 无 `interval=None`；
- checkpoint 可重新加载；
- reload 后单 batch forward 成功；
- quant-only 评估可生成完整 7Scenes/ETH3D/DTU 指标；
- metadata 中记录了代码 commit、模型 hash 和 manifest；
- 新文件没有覆盖 `param/vggt/` 下旧实验参数。

---

## 17. 8 卡 H800 执行顺序

### 阶段 A：数据和 FP baseline

| GPU | Job |
|---:|---|
| 0 | VGGT FP：7Scenes-dense |
| 1 | VGGT FP：ETH3D |
| 2 | VGGT FP：DTU |
| 3 | Pi3 FP：7Scenes/ETH3D/DTU |
| 4 | Dust3R FP：点云测试集 |
| 5 | MASt3R FP：点云测试集 |
| 6 | VGGT FP：Co3Dv2 camera |
| 7 | Co3Dv2 official VGGT evaluator：manifest 生成与复核 |

### 阶段 B：VGGT 主量化 baseline

| GPU | Job |
|---:|---|
| 0 | RTN W4A8 |
| 1 | PTQ4ViT W4A8 |
| 2 | ERQ W4A8 |
| 3 | RepQ W4A8 |
| 4 | GPTQ W4A8 |
| 5 | QuantVGGT W4A8 |
| 6 | TAPTQ W4A8 per-tensor |
| 7 | TAPTQ W4A8 channel-wise / checkpoint validation |

### 阶段 C：TAPTQ 正式校准与补偿

| GPU | Job |
|---:|---|
| 0 | TQ-1：VGGT W8A8 per-tensor calibration |
| 1 | TQ-2：VGGT W6A6 per-tensor calibration |
| 2 | TQ-3：VGGT W4A8 per-tensor calibration |
| 3 | TQ-4：VGGT W4A8 channel-wise calibration |
| 4 | TQ-5：TRE + QwT |
| 5 | TQ-6：MSE + QwT |
| 6 | TQ-7：Hessian + QwT |
| 7 | TQ-8：Random seeds 1/2/3 |

TQ-5 至 TQ-8 必须等待对应 calibration checkpoint 生成后再启动；如果 calibration 时间不同，应先完成 TQ-1 至 TQ-4，再重新分配 GPU。

### 阶段 D：channel-wise 和扩展任务

| GPU | Job |
|---:|---|
| 0 | ERQ channel-wise |
| 1 | RepQ channel-wise |
| 2 | GPTQ channel-wise |
| 3 | TAPTQ channel-wise + QwT |
| 4 | Pi3 TAPTQ per-tensor |
| 5 | Pi3 TAPTQ channel-wise |
| 6 | Co3Dv2 quantized camera prediction |
| 7 | H800 deployment matrix |

---

## 18. 当前批次的执行状态定义

每个实验 job 使用以下状态之一：

| 状态 | 含义 |
|---|---|
| `available` | 代码、数据和 checkpoint 已存在，可直接启动 |
| `sanity-only` | 已有结果或代码，但 evaluator/协议尚未统一，只能作为内部 sanity check |
| `needs-reproduction` | 只有旧脚本/旧参数，必须在当前 commit 和 H800 重跑 |
| `blocked-data` | 缺少数据集或数据 manifest |
| `blocked-code` | 缺少 baseline 实现或当前代码无法加载 |
| `blocked-channelwise` | per-tensor 可运行，但 channel-wise shape/实现未对齐 |
| `completed` | 已完成并通过 checkpoint/reload/指标验收 |

当前初始标记建议：

| 实验 | 初始状态 |
|---|---|
| VGGT FP 7Scenes/DTU | `completed` |
| VGGT FP ETH3D | `rebuilding-pi3-reference-protocol`，Pi3 官方 `prepare_eth3d.sh` 原始归档下载中；此前 sparse projection 结果仅作 provisional |
| VGGT RTN | `available`，统一入口 `mv_recon/baseline_quant.py`，待 H800 重现 |
| VGGT PTQ4ViT | `baseline candidate`，已有 7Scenes 结果但需统一重现 |
| VGGT ERQ | `available`，已适配 Aqer + Wqer + RepQ 初始化，待 H800 重现 |
| VGGT RepQ | `available`，已适配 LayerNorm/Linear scale reparameterization，待 H800 重现 |
| VGGT GPTQ | `baseline candidate`，已有历史/旧代码结果，需统一重现 |
| VGGT QuantVGGT | `baseline candidate`，已完成官方 evaluator 的 FP/W4A4/W4A8/W8A8 |
| VGGT TAPTQ 正式 checkpoint | `completed`，已完成多位宽、补偿和主要 7Scenes/DTU 评估 |
| TAPTQ channel-wise | `completed`，quant-only 和 compensated 均已生成并评估 |
| Pi3 FP | `completed`，已有 7Scenes-dense 参考结果 |
| Pi3 TAPTQ | `blocked-code`，当前主入口仍按 VGGT aggregator 结构适配 |
| Dust3R FP | `completed-h800-7scenes-dense`，统一 E1 adapter 已跑完 18 序列 |
| MASt3R FP | `completed-h800-7scenes-dense`，统一 E1 adapter 已跑完 18 序列 |
| Co3Dv2 camera | `rebuilding-pi3-vggt-reference-protocol`，Pi3 frozen map 有 2506 序列；官方 10 test categories 完整下载中 |

---

## 19. 交付物清单

### 数据和协议

- [x] `doc/tmm_calibration_manifest.json`；
- [x] `doc/tmm_test_manifest.json`；
- [x] `doc/tmm_eth3d_manifest.json`（blocked-data manifest）；
- [x] `doc/tmm_co3dv2_manifest.json`（blocked-data manifest）；
- [ ] 统一 evaluator 配置（TAPTQ 与 QuantVGGT upstream 仍分开）；
- [x] 模型输入和帧采样配置。

### 参数和日志

- [ ] 每个 baseline 的 config；
- [ ] 每个方法的 checkpoint/量化参数；
- [ ] `runtime.json`；
- [ ] `git_commit.txt`；
- [ ] `model_hash.txt`；
- [ ] `stdout.log` / `stderr.log`；
- [ ] calibration time 和 peak memory。

### 论文表格

- [x] Table 1：setup（协议已冻结）；
- [x] Table 2：FP baselines（VGGT/Pi3/QuantVGGT 部分完成）；
- [ ] Table 3：VGGT 主量化比较（RTN/ERQ/RepQ 已实现，待统一重现）；
- [ ] Table 4：Pi3 量化扩展；
- [x] Table 5：TAPTQ component ablation（7Scenes/DTU 部分完成）；
- [ ] Table 6：四种方法 channel-wise（仅 TAPTQ 完成）；
- [ ] Table 7：校准和搜索效率（ternary/exhaustive 已完成精度，forward-count 审计仍待补）；
- [x] Table 8：H800 deployment（fake-quant 路径完成）；
- [ ] Table 9：Co3Dv2 camera（blocked-data）；
- [ ] Table 10：cross-architecture（Pi3 FP、Dust3R/MASt3R FP unified E1 已完成；量化语义 adapter 仍 pending）；
- [ ] Table S1–S16：完整结果、敏感性、失败案例和复现信息。

---

## 20. 当前执行原则

1. 先完成数据 manifest 和 FP baseline，再启动量化矩阵；
2. 先以 VGGT 为主线复现所有 baseline，再按 T1 语义模块适配扩展 Pi3、Dust3R、MASt3R；
3. Dust3R/MASt3R 必须完成 FP + TAPTQ W4A8，先完成 per-tensor，再完成 channel-wise；
4. ERQ、RepQ、GPTQ、TAPTQ 的 channel-wise 结果必须使用各自当前版本重新生成参数；
5. TAPTQ 正式 TMM 结果必须重新 calibration，旧的 `param/vggt/8scan_*` 和 `20scan_*` 只作为 sanity check；
6. 每一种 `(model, method, W/A, granularity, calibration manifest)` 只做一次校准，后续评估重复使用；
7. 不能将不同 evaluator 产生的数字直接放进同一公平比较表；
8. 任何无法复现的 baseline 必须在表中标记 `pending`、`sanity-only` 或 `not applicable`，不能用空白数字掩盖；
9. 先生成可 review 的中间表，再决定哪些结果进入正文、哪些移到 Supplement；
10. 所有最终结果必须能由 H800、代码 commit、模型 hash 和数据 manifest 重现。

---

## 21. 已确认的最终实验协议（Authoritative）

本节覆盖前文可能残留的旧假设，后续执行以本节为准。

### 21.1 任务和模型范围

| Model | Point cloud reconstruction | Co3Dv2 camera prediction |
|---|---|---|
| VGGT | FP + 全 baseline + TAPTQ | FP + 全 baseline + TAPTQ |
| Pi3 | FP + TAPTQ | 不测 |
| Dust3R | FP + TAPTQ | 不测 |
| MASt3R | FP + TAPTQ | 不测 |

点云测试集固定为：

```text
7Scenes-dense
ETH3D
/data/workspace/TAPTQ/data/dtu
```

相机参数测试集固定为：

```text
Co3Dv2 official VGGT evaluation protocol
```

### 21.2 统一输入协议

点云任务采用 E1：

```text
所有四个 basemodel 使用相同 8 views
相同 sequence
相同 frame IDs
相同图像预处理和分辨率
相同 GT、坐标变换、Sim(3)/ICP 和指标计算
```

Co3Dv2 不使用 E1 的 8 views，而使用官方 VGGT evaluator：

```text
TEST_CATEGORIES
split=test
min_num_images=50
num_frames=10
seed=0
```

所有 VGGT Co3Dv2 方法复用同一个 `doc/tmm_co3dv2_manifest.json`。

### 21.3 量化和 baseline 范围

主 baseline 对比集中在 VGGT：

```text
RTN
PTQ4ViT
ERQ
RepQ
GPTQ
QuantVGGT
TAPTQ
```

主量化设置固定为：

```text
W4A8 per-tensor
```

四个 basemodel 的最低 TAPTQ 结果均为：

```text
W4A8 per-tensor
```

`ERQ`、`RepQ`、`GPTQ`、`TAPTQ` 需要补充 channel-wise 结果；`TAPTQ` 的 channel-wise 扩展覆盖四个 basemodel。

RTN 和 PTQ4ViT 不强制补 channel-wise，除非后续实现成本很低且作者 review 后决定加入。

### 21.4 TAPTQ 固定参数

正式 TAPTQ 主结果不做本地 validation tuning，直接采用论文/rebuttal 默认协议并使用 rho sweep 中更优的 TRE 参数：

```text
strategy = module
select_metric = null
rho / tail_ratio = 0.1
tau_thr = 0.007
rank = 16
keep_ratio = unset
```

这里的 `rho/tail_ratio=0.1` 只用于 TRE 计算；不等于 `keep_ratio`。

补偿逻辑为：

```text
每个模块计算 TRE
→ 使用 tau_thr=0.007 判断是否启用 QwT
→ 不使用固定 keep_ratio
```

### 21.5 TAPTQ calibration 和 compensation

统一使用：

```text
Calibration set:
/data/workspace/TAPTQ/Pi3-evaluation/data/dtu_8

Compensation set:
/data/workspace/TAPTQ/Pi3-evaluation/data/dtu_8

Scan IDs:
scan5, scan6, scan7, scan8, scan18, scan22, scan56, scan123

Frame map:
Pi3-evaluation/datasets/seq-id-maps/DTUTrain_8_mv-recon_seq-id-map-kf5.json

Frames per scan:
10
```

本批次不使用 `data/dtu_20`，也不使用 20→8 progressive calibration 作为默认协议。

### 21.6 TAPTQ 模型适配范围

采用 T1 语义模块级适配：

```text
Transformer block
├── Attention → AttnQwT
└── MLP       → MlpQwT
```

对四个模型统一保持：

```text
backbone quantized
task-specific heads FP
```

如果 Dust3R 或 MASt3R 无法完成语义模块映射，必须记录为 `TAPTQ adaptation blocked`，不能替换成黑盒量化后继续冒充 TAPTQ。

### 21.7 Baseline calibration policy

主表采用：

```text
所有可校准 baseline 尽量统一 dtu_8
官方默认超参数
不在测试集调参
```

Supplement 保留：

```text
official calibration
official checkpoint
official tuned configuration
```

每条结果必须标注：

```text
unified dtu_8
official calibration
not reproduced
pending
```

QuantVGGT 如果只能使用官方 calibration cache，主表必须显式标记其 calibration protocol，不得伪装成 dtu_8 校准。

### 21.8 Checkpoint and artifact policy

优先使用官方 checkpoint；若官方资源不可用，再使用本地 checkpoint并记录 source 和 SHA256。

每个 TAPTQ 设置同时保存：

```text
calibrated_quant_only.pt
compensated_tre.pt
calibrated_quant_only.json
compensated_tre.json
config.yaml
calibration_manifest.json
runtime.json
git_commit.txt
model_hash.txt
calibration.log
metrics.csv
```

channel-wise 设置使用独立目录和独立 checkpoint，不能覆盖旧的 `param/vggt/8scan_*` 或其他历史参数。

### 21.9 结果统计

点云重建主表报告：

```text
Acc-mean / Acc-med
Comp-mean / Comp-med
NC-mean / NC-med
```

Co3Dv2 按官方 VGGT evaluator 报告：

```text
relative rotation error
relative translation angle error
mean / median
```

Supplement 追加逐 sequence、逐 category、worst-case、failure case、latency、memory 和 calibration time。

### 21.10 Baseline reproduction policy

采用 H1 → H2：

1. 先查找本地、远端和已有 checkpoint；
2. 先复现现有实现；
3. 能在 H800 + E1/官方 Co3D protocol 下运行才进入主表；
4. 对主结论关键但缺失的 `RTN/ERQ/RepQ` 再按论文补实现；
5. 不把论文原始数字直接混入 H800 主表。

---

## 22. 当前阶段汇报主表：7Scenes-dense

当前阶段所有实验结果汇报优先使用下表；DTU 仅作为补充验证，不与主表并列抢占叙事位置。表中均为 18 个 7Scenes-dense 序列的 mean 指标，完整逐序列结果位于 `/mnt/cephfs4/josephyou_debug/tmm-dtu8/` 对应目录。

| Setting | Acc-mean ↓ | Comp-mean ↓ | NC-mean ↑ | Status |
|---|---:|---:|---:|---|
| FP VGGT | 0.020174 | 0.029582 | 0.679993 | completed |
| TAPTQ W4A8 quant-only | 0.031895 | 0.045382 | 0.680580 | completed |
| TAPTQ W4A8 compensated | 0.029494 | 0.040417 | 0.680707 | completed |
| GPTQ W8A8 | 0.019850 | 0.029038 | 0.681350 | baseline candidate |
| PTQ4ViT W4A8 | 0.033761 | 0.047767 | 0.680801 | baseline candidate |
| TAPTQ W8A8 quant-only | 0.019598 | 0.029006 | 0.680987 | completed |
| TAPTQ W8A8 compensated | 0.019603 | 0.028981 | 0.680953 | completed |
| TAPTQ W6A6 quant-only | 0.021161 | 0.030383 | 0.682346 | completed |
| TAPTQ W6A6 compensated | 0.020251 | 0.030469 | 0.680545 | completed |
| TAPTQ W4A8 channel-wise | 0.019406 | 0.029320 | 0.679974 | completed |
| TAPTQ W4A8 channel-wise compensated | 0.018932 | 0.033166 | 0.687987 | completed |
| QuantVGGT FP (official evaluator) | 0.020490 | 0.033834 | 0.676035 | baseline candidate; upstream evaluator |
| QuantVGGT W4A4 (official evaluator) | 0.020498 | 0.033834 | 0.676302 | baseline candidate; upstream evaluator |
| QuantVGGT W4A8 (official evaluator) | 0.020489 | 0.033833 | 0.676142 | baseline candidate; upstream evaluator |
| QuantVGGT W8A8 (official evaluator) | 0.020497 | 0.033838 | 0.676140 | baseline candidate; upstream evaluator |
| RTN-style W4A8 (official evaluator) | 0.020496 | 0.033840 | 0.676275 | baseline candidate; upstream evaluator |
| Pi3 FP | 0.021613 | 0.026447 | 0.670970 | cross-model reference |

部署成本基准已在同一 H800 上完成，输入为 8 views × 518 × 518，batch size 1，AMP 开启，5 次 warmup、20 次计时：FP VGGT `243.91 ms`，W4A8 per-tensor `389.88 ms`，W4A8 channel-wise `387.87 ms`，W8A8 `388.04 ms`。这些 fake-quant 数字不等同于真实 INT4/INT8 kernel 延迟，放入 deployment 表而不是精度主表。

主表进入规则：只有代码版本、checkpoint、校准 manifest、evaluator 和逐序列结果齐全的设置进入 `completed`；baseline 若 evaluator 或 calibration protocol 不完全一致，保留为 `baseline candidate` 并在表注中标明，不能伪装成统一公平比较。

## 23. 全部 TODO 执行台账（2026-08-10）

| 计划项 | 当前状态 | 证据/下一步 |
|---|---|---|
| 7Scenes-dense 主结果 | completed | VGGT/Pi3/TAPTQ/QuantVGGT 结果已进入本节主表 |
| DTU 补充验证 | completed | 22-scan 指标和多个 checkpoint 已持久化到 `/mnt/cephfs4/josephyou_debug/tmm-dtu8/` |
| 固定 `dtu_8` 校准与 QwT | completed | W4A8/W6A6/W8A8、channel-wise quant-only/compensated 已生成 |
| QuantVGGT FP/W4A4/W4A8/W8A8 | completed as upstream baseline | 结果使用 QuantVGGT upstream evaluator，不能与 unified evaluator 直接混排 |
| H800 fake-quant deployment | completed | FP/W4A8/channel-wise/W8A8 latency、显存和大小已记录 |
| RTN-style upstream baseline | completed as upstream baseline | QuantVGGT evaluator 的 `not_smooth + not_rot` W4A8 已运行，需标注协议 |
| PTQ4ViT/GPTQ unified reproduction | baseline candidate | 已有 7Scenes 结果，仍需统一当前 dtu_8、代码和 evaluator 后才能进公平主表 |
| ERQ/RepQ | blocked-code | 当前仓库无独立可复现入口，不能用论文数字替代 |
| Pi3 TAPTQ | blocked-code | 现有 `taptq.py` 的 wrapper 绑定 VGGT `aggregator`，需要独立 Pi3 adapter |
| Dust3R/MASt3R FP | completed on H800, 18 sequences each | unified E1 输出、坐标、尺度、ICP 和 Acc/Comp/NC 已完成；结果在 `tmm-results/eval/*_7scenes_dense/_all_samples.json` |
| Dust3R/MASt3R TAPTQ | adapter implemented, full W4A8 running | `external_taptq.py` 量化 enc_blocks/dec_blocks/dec_blocks2 的 288 个 linear，并对 120 个 attention/cross-attention/MLP branch 做低秩 module QwT；smoke、mini calibration、checkpoint reload 均通过 |
| ETH3D | sparse-point adapter completed, official dense protocol pending | `prepare_eth3d_official.py` 已将官方 COLMAP/稀疏点转换为 TMM per-frame depth layout；13 序列八个单卡 H800 run 已完成，结果见 `eth3d_fp_summary.json` |
| Co3Dv2 camera | completed subset, full release pending | run `700605702` 八 shard 全部 exit 0，48 valid sequences；AUC@30 mean 0.000201，不能代表 full Co3Dv2 |
| ternary vs exhaustive | H800 checkpoint/eval completed, audit pending | Linear PTQ search 已支持 `ptq.search_mode=exhaustive/ternary`；两种模式均完成固定 dtu_8 校准和 7Scenes dense 18 序列评估，forward-count 仍需从日志提取 |
| real ONNX/TensorRT/INT4 | pending-engineering | 当前 benchmark 是 PyTorch fake-quant，不应宣称真实 INT kernel 加速 |
| QAT、全笛卡尔积和大规模 Co3Dv2 sweep | P2 pending | 在 P0/P1 缺口解决前不优先执行 |

本台账中的 `blocked-*` 不是用空白掩盖结果，而是明确的交付边界：缺数据需通过 Ceph 补齐，缺统一代码需先完成 adapter/evaluator，缺真实 kernel 需单独工程实现。

### 23.2 H800 结果收口（2026-08-11）

- 第二轮八卡矩阵 `run=700551152` 成功完成：VGGT FP、TAPTQ W4A8 ternary/exhaustive、W4A8 channel-wise、W6A6、W8A8 均覆盖 7Scenes dense 18 序列；Dust3R/MASt3R unified E1 各完成 18 序列。
- 最新统一 evaluator 结果已落在个人 Ceph `pansicheng/tmm-results/eval`，manifest 已记录均值；Dust3R/MASt3R 结果分别为 Acc/Comp/NC = 0.019343/0.029319/0.680308 和 0.025488/0.031090/0.665829。
- ETH3D 官方三份归档已下载并 SHA256 校验，13 个序列已解压；八个单卡 H800 run `700679076`/`700679081`/`700679089`/`700679099`/`700679105`/`700679117`/`700679132`/`700679135` 完成 sparse COLMAP point projection adapter 评估，Acc/Comp/NC mean = `0.070958/0.167750/0.091757`。该结果不是官方 dense-depth protocol，官方 dense GT 转换仍 pending。
- Co3dv2 single-sequence subset 已下载约 17G，48 个 test annotation 已生成；Co3Dv2 camera evaluator 已提交八个类别 shard，结果待 run 完成后归档，不能代表完整 Co3Dv2。

### 23.3 Co3Dv2 subset 收口（2026-08-11）

- `run=700605702` 的八个 Co3dv2 camera shard 全部 exit 0，48 个 test annotation/valid sequence 均完成。
- subset AUC 均值为 AUC@30 `0.000201`、AUC@15/5/3 `0`；这是官方 single-sequence subset 结果，不能代表 full Co3Dv2，也不应直接进入 Table 9 主结果。

### 23.1 本轮工程收口（2026-08-10）

- 已将默认校准集从旧的 `DTU_20` 切换为 authoritative `DTU_train_8`，并修复 `run_calibrate`、`run_compensate_eval`、`run_e2e` 误从测试集取校准样本的问题。
- 已移除评估和 deployment benchmark 中的 H800 旧绝对模型路径，改为工作区路径并支持 `VGGT_MODEL_PATH`、`PI3_MODEL_PATH`、`TMM_DATA_ROOT`、`TMM_CACHE_ROOT` 覆盖。
- 已新增 `Pi3-evaluation/mv_recon/external_baselines.py`，统一 DUSt3R/MASt3R 的 E1 输入、native pair inference、global alignment、Sim(3)、ICP 和 Acc/Comp/NC 输出；实际逐序列 H800 运行仍需在有权限的 H800 run 中执行。
- `PTQSLBatchingQuantLinear` 已支持 `ptq.search_mode=exhaustive|ternary`，记录 weight/activation candidate forward count；Table 7 仍需用固定 `dtu_8` 在 H800 实测。
- 本轮 AST/py_compile 和 linter 检查通过；当前 CVM 无 PyTorch，且现有 H800 run 所属人为其他用户。新增代码归档已上传至个人 Ceph 的 `pansicheng/tmm-code-update.tar.zst` 并带 SHA256；新 run 预检又被本地 Gemini mount 配置缺少 `JFS_META` 阻断，因此无法从本会话直接执行远端 forward，不把未运行的结果标成 completed。
