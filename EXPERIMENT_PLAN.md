# TAPTQ 实验计划表

> 状态：计划阶段，尚未按本计划批量执行。
>
> 默认校准集：`/data/workspace/TAPTQ/data/dtu_20`
>
> 主入口：`Pi3-evaluation/mv_recon/taptq.py`

## 1. 实验约定

| 项目 | 约定 |
|---|---|
| 校准集 | `/data/workspace/TAPTQ/data/dtu_20`，20 个 DTU scan |
| 校准策略 | 只校准一次，生成量化参数后重复复用 |
| 默认量化 | W4A8 |
| 默认补偿 | module-level QwT |
| 默认设备 | 单节点 H800；正式部署实验需额外记录 GPU 型号和 CUDA 环境 |
| 默认输入 | `load_img_size=518`，固定帧采样和 AMP 策略 |
| 随机种子 | 主实验 `seed=42`；Random 额外使用 `1/2/3` |
| 正式评估集 | 第一批：`7scenes-sparse`、`7scenes-dense`、`DTU`；数据完整后加入 `NRGBD`、`ETH3D` |
| 指标 | `Acc`、`Comp`、`NC1`、`NC2`、均值、中位数 |
| 对齐 | 统一使用 Umeyama Sim(3) 对齐和 ICP 后处理 |
| 输出 | 每个实验使用独立 `save_suffix`，禁止覆盖历史结果 |

### 关键变量定义

- \(\rho\)：`compensate.keep_ratio`，实际启用 QwT 补偿的模块比例。
- \(\tau\)：`compensate.tau_thr`，legacy threshold 模式下的误差阈值。
- `rank`：QwT 低秩 SVD 补偿的 rank。
- `tail_ratio`：TRE 计算时选择的高幅值输出元素比例。

## 2. 总体任务表

| 阶段 | 任务 | 核心实验 | 主要变量 | 主要输出 | 优先级 |
|---|---|---|---|---|---|
| 0 | 基线与环境冻结 | 固定模型、数据、采样、指标和软件版本 | seed、分辨率、帧数、GPU | 配置快照、环境信息 | P0 |
| 1 | FP + QuantVGGT 正式对比 | FP VGGT、W8A8、W6A6、W4A8、W4A4 | 权重位宽、激活位宽 | 精度、延迟、显存、模型大小 | P0 |
| 2 | Channel-wise quantization | block-wise 与 channel-wise 对比 | `PTQ4ViT`、`PTQ4ViT_channelwise` | 精度、校准时间、参数规模 | P0 |
| 3 | 部署开销 | FP、fake-quant、QwT、ONNX/TRT | latency、memory、overhead | 部署性能表 | P0 |
| 4 | 补偿策略对比 | TRE、MSE、Hessian、Random | metric、\(\rho\) | 精度—补偿比例曲线 | P0 |
| 5 | 超参数敏感性 | \(\rho\)、\(\tau\)、rank | 补偿比例、阈值、低秩维度 | 敏感性曲线、推荐参数 | P1 |
| 6 | Ternary search trade-off | 搜索轮数与精度/时间权衡 | `search_round`、`eq_n` | 校准时间—精度曲线 | P1 |
| 7 | Failure cases 与 limitations | 失败数据、显存问题、异常点云 | 数据集、序列、量化设置 | 失败案例表、限制说明 | P1 |
| 8 | Dust3R / MASt3R 基线 | 统一评估协议下的 FP 结果 | basemodel、数据集 | 基线结果表 | P0 |

## 3. 阶段 0：基线与环境冻结

### 目标

确保所有后续结果可以复现，并且不同模型之间使用相同的评估协议。

### 固定内容

- 校准集固定为 `DTU_20`，避免每个实验重复校准。
- 正式评估第一批使用 `7scenes-sparse`、`7scenes-dense`、`DTU`。
- 固定 `load_img_size=518`、帧采样策略、模型 AMP 设置。
- 固定 `seed=42`。
- 记录 GPU、驱动、CUDA、PyTorch、依赖版本和 Git commit。
- 记录模型 checkpoint 的路径和 SHA256。

每个实验应保留：

```text
outputs/<experiment_name>/
├── config.yaml
├── runtime.json
├── logs/
├── metrics.csv
└── hydra/
```

## 4. 阶段 1：FP + QuantVGGT 正式对比

### 实验矩阵

| 实验名 | 模型 | 量化设置 | 目的 |
|---|---|---|---|
| `fp_vggt` | FP VGGT | FP16/BF16 | 精度和部署基线 |
| `qvggt_w8a8` | QuantVGGT | W8A8 | 低损失量化基线 |
| `qvggt_w6a6` | QuantVGGT | W6A6 | 中间压缩点 |
| `qvggt_w4a8` | QuantVGGT | W4A8 | 主实验设置 |
| `qvggt_w4a4` | QuantVGGT | W4A4 | 激进量化设置 |

### 必须比较

- `Acc-mean`、`Acc-med`
- `Comp-mean`、`Comp-med`
- `NC1-mean`、`NC2-mean`
- 每个序列的最差结果
- 平均推理延迟
- 峰值显存
- 参数大小、checkpoint 大小

FP 和 QuantVGGT 必须使用相同的图片、帧 ID、分辨率、对齐和后处理。

## 5. 阶段 2：Channel-wise quantization

### 对比设置

| 设置 | 配置 | 说明 |
|---|---|---|
| Block-wise | `PTQ4ViT` | 当前默认量化方式 |
| Channel-wise | `PTQ4ViT_channelwise` | Linear 按输出通道量化 |
| Channel-wise + QwT | Channel-wise + module QwT | 观察补偿恢复能力 |

### 固定设置

- 模型：VGGT
- 量化：W4A8
- 校准集：`DTU_20`
- 评估集：`7scenes-dense`、`DTU`
- 补偿策略：先固定 Hessian、\(\rho=0.5\)

### 输出

- 精度变化
- 校准时间
- 量化参数大小
- 峰值显存
- Linear 层数量和量化粒度
- 异常 interval 或量化输出

## 6. 阶段 3：部署 latency / memory / overhead

### 3.1 PyTorch fake-quant 路径

使用：

- `mv_recon/benchmark_quant_deploy.py`
- `mv_recon/compare_quant_fp.py`
- `mv_recon/benchmark_common.py`

对比：

| 设置 | 测量内容 |
|---|---|
| FP VGGT | latency、显存、state_dict 大小 |
| W4A8 fake quant | latency、显存、量化模型大小 |
| W4A8 + QwT | latency、显存、补偿参数开销 |
| Channel-wise W4A8 | 与 block-wise 对比 |

需要记录：

- warmup 次数和正式迭代次数
- latency mean/stdev/min
- peak allocated/reserved VRAM
- 初始模型大小
- 量化后模型大小
- QwT A/B 或 LoRA 参数大小
- 额外模型加载时间
- 额外校准时间

### 3.2 ONNX / TensorRT 路径

使用：

- `mv_recon/export_vggt_onnx.py`
- `mv_recon/tensorrt/`

建议顺序：

1. 导出 FP16 ONNX。
2. 构建 TensorRT FP16 engine。
3. 验证 ONNX/TensorRT 数值一致性。
4. 尝试 INT8/QDQ。
5. 再评估 INT4 或自定义 kernel。

必须明确：

```text
PyTorch fake quant 不等于真实 INT4 Tensor Core/TensorRT 性能。
```

## 7. 阶段 4：TRE vs MSE / Hessian / Random

### 固定设置

- 模型：W4A8
- 校准集：`DTU_20`
- 补偿粒度：`module`
- \(\rho\)：`0.1、0.25、0.5、0.75、1.0`
- 评估集：`7scenes-dense`、`DTU`

### 策略

| 方法 | 模块选择逻辑 |
|---|---|
| TRE | 按 tail relative error 排序 |
| MSE | 按普通均方误差排序 |
| Hessian | 按当前 Fisher/Hessian 近似分数排序 |
| Random | 随机选择相同数量的模块 |

### 输出图表

- 横轴：补偿模块比例 \(\rho\)
- 纵轴：`Acc`、`Comp`、`NC`
- 四条曲线：TRE、MSE、Hessian、Random
- Random 使用 3 个 seed，并显示均值和标准差

四种方法必须保证实际补偿模块数量一致。

## 8. 阶段 5：\(\rho\)、\(\tau\)、rank 敏感性

### 8.1 \(\rho\) 敏感性

```text
rho = 0.0, 0.1, 0.25, 0.5, 0.75, 1.0
```

分别测试 TRE、MSE、Hessian、Random，输出：

- 精度—补偿比例曲线
- 参数量—补偿比例曲线
- 延迟—补偿比例曲线

### 8.2 \(\tau\) 敏感性

当前代码中，设置 `select_metric` 后模块选择主要由 \(\rho\) 决定，\(\tau\) 不再主导筛选。因此 \(\tau\) 实验应使用 legacy threshold 模式：

- 不设置 `compensate.select_metric`
- `skip_p=0`

建议：

```text
tau = 0.000, 0.001, 0.003, 0.005,
      0.007, 0.010, 0.020, 0.050
```

记录：

- 实际启用的 QwT 模块数
- Acc/Comp/NC
- QwT 参数量
- 推理延迟

### 8.3 rank 敏感性

```text
rank = 8, 16, 32, 64, 128, 256
```

固定：

- 选择方法：Hessian
- \(\rho=0.5\)
- W4A8
- 校准集：`DTU_20`

记录：

- 精度
- 补偿参数量
- 峰值显存
- 推理延迟
- SVD/补偿阶段耗时

最终绘制精度—开销 Pareto 曲线。

## 9. 阶段 6：Ternary search 效率—精度 trade-off

当前 PTQ 配置中重点关注：

- `search_round`
- `eq_n`
- `eq_alpha`
- `eq_beta`

### 第一组实验

固定其他参数，只改变：

```text
search_round = 1, 2, 3, 4
```

### 第二组实验

```text
eq_n = 25, 50, 100, 200
```

### 记录

| 指标 | 说明 |
|---|---|
| Calibration time | 完整校准耗时 |
| Per-module time | 单模块平均搜索耗时 |
| Forward count | 搜索期间前向次数 |
| Quant error | 模块级量化误差 |
| Final Acc/Comp/NC | 正式评估指标 |
| Peak memory | 校准峰值显存 |

输出：

1. 搜索时间 vs 精度。
2. 前向次数 vs 精度。
3. 推荐的效率—精度平衡点。

## 10. 阶段 7：Failure cases 与 limitations

### Failure case 分类

| 类型 | 记录内容 |
|---|---|
| 数据问题 | 缺失图片、深度、mask、相机参数 |
| 采样问题 | seq-id-map 与数据目录不匹配 |
| 数值问题 | NaN、Inf、空 valid mask |
| 量化问题 | interval 为 None、量化输出爆炸 |
| 补偿问题 | R² 小于 0、补偿后精度下降 |
| 显存问题 | Hessian 校准 OOM、QwT SVD OOM |
| 评估问题 | ICP 不收敛、点云为空、法向估计失败 |
| 部署问题 | ONNX 不支持算子、TensorRT 构建失败 |
| 泛化问题 | DTU 校准后其他数据集精度明显下降 |

每个失败案例至少记录：

```text
dataset
sequence
model
quant_config
rho
tau
rank
error_type
log_path
repro_command
```

### Limitations

1. PyTorch benchmark 是 fake quant，不代表真实 INT4 硬件性能。
2. 当前 Hessian 分数是输出误差加权近似，不是完整二阶 Hessian。
3. QwT 补偿依赖校准数据分布，可能存在数据集偏差。
4. `DTU_20` 只用于校准，不能参与正式评估指标统计。
5. Dust3R、MASt3R 与 VGGT 可能存在输入接口和输出坐标系差异。
6. 不同模型的预训练数据、输入分辨率和后处理可能不同。
7. Sim(3) 对齐和 ICP 可能掩盖部分绝对尺度误差。

## 11. 阶段 8：Dust3R / MASt3R 基线

第一阶段只做 FP baseline，不立即进行量化。

### 统一流程

```text
Dust3R / MASt3R 推理
→ 输出深度或点云
→ 转换到统一坐标格式
→ 使用相同 GT、mask 和有效点规则
→ 使用相同 Umeyama + ICP
→ 计算 Acc / Comp / NC
```

### 基线矩阵

| 模型 | 量化 | 目标 |
|---|---|---|
| FP VGGT | 无 | 主基线 |
| QuantVGGT W4A8 | W4A8 | 量化主结果 |
| Dust3R | FP | 外部基线 |
| MASt3R | FP | 外部基线 |
| Dust3R / MASt3R | 可选量化 | 后续扩展 |

必须记录：

- checkpoint
- 输入图片数量
- 输入分辨率
- 是否使用相机位姿
- 是否使用全序列或采样帧
- 点云坐标系
- 是否进行尺度对齐
- 是否使用 ICP
- 推理耗时和显存

## 12. 推荐执行顺序

```text
1. 固定配置、环境、数据和评估协议
2. 生成 FP VGGT 正式结果
3. 使用 DTU_20 校准结果生成 QuantVGGT W4A8 结果
4. 完成 channel-wise 对比
5. 完成 TRE/MSE/Hessian/Random 对比
6. 完成 rho/tau/rank 敏感性
7. 完成 ternary search trade-off
8. 完成 latency/memory/overhead
9. 加入 Dust3R、MASt3R FP baseline
10. 汇总 failure cases 和 limitations
```

## 13. 最终汇总表

| Model | Quant | Channel-wise | Compensation | Dataset | Acc | Comp | NC | Latency | VRAM | Size |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| VGGT | FP | - | - | 7Scenes |  |  |  |  |  |  |
| QuantVGGT | W4A8 | No | None | 7Scenes |  |  |  |  |  |  |
| QuantVGGT | W4A8 | Yes | None | 7Scenes |  |  |  |  |  |  |
| QuantVGGT | W4A8 | Yes | TRE | 7Scenes |  |  |  |  |  |  |
| QuantVGGT | W4A8 | Yes | Hessian | 7Scenes |  |  |  |  |  |  |
| Dust3R | FP | - | - | 7Scenes |  |  |  |  |  |  |
| MASt3R | FP | - | - | 7Scenes |  |  |  |  |  |  |

## 14. 相关代码入口

- `Pi3-evaluation/mv_recon/taptq.py`：TAPTQ 主流程
- `Pi3-evaluation/mv_recon/eval.py`：FP 多视图评估
- `Pi3-evaluation/mv_recon/benchmark_quant_deploy.py`：部署 benchmark
- `Pi3-evaluation/mv_recon/compare_quant_fp.py`：FP 与 fake-quant 对比
- `Pi3-evaluation/mv_recon/run_ablation_7sd.sh`：7-Scenes 消融
- `Pi3-evaluation/configs/data/mv_recon.yaml`：数据集和默认校准集配置
- `Pi3-evaluation/configs/evaluation/mv_recon.yaml`：评估、校准和补偿数据集配置
- `Pi3-evaluation/mv_recon/legacy/`：历史 PTQ/GPTQ 分支

## 15. TAPTQ 目录布局

为减少根目录混杂内容，当前目录按用途归类：

```text
TAPTQ/
├── Pi3-evaluation/       主项目：TAPTQ、评估和部署
├── data/                 数据集和缓存
├── projects/             独立项目：QuantVGGT、Bohua 及其运行脚本
├── models/               HuggingFace 缓存和 Pi3 权重
├── miniconda3/           原远程环境，暂不移动以避免破坏 prefix
├── conda/                原远程 Conda 元数据
└── _snapshot/             远程机器系统目录、用户缓存和日志快照
```

`.codebuddy/`、`.codebuddy-server-cn/`、`.git/` 保持在根目录，不参与快照归类。`data/`、`Pi3-evaluation/` 和 `projects/` 的活动路径保持清晰；模型与系统快照不进入 Git 源码仓库。

QuantVGGT 的统一入口为：

```bash
cd /data/workspace/TAPTQ/projects/QuantVGGT
bash scripts/run_quant_eval.sh
```

也可以使用兼容的薄封装：`scripts/run_w4a8.sh`、`scripts/run_w8a8.sh`、`scripts/run_7s_quant.sh` 和 `scripts/run_nr_quant.sh`。
