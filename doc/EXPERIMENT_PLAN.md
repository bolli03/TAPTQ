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

- 论文中的 \(\rho\)：对应代码 `compensate.tail_ratio`，表示 TRE 计算时选取的高幅值元素比例，论文默认值为 `0.01`。
- 模块补偿比例：对应代码 `compensate.keep_ratio`，表示实际启用 QwT 补偿的模块比例，不能与论文中的 \(\rho\) 混淆。
- \(\tau\)：`compensate.tau_thr`，legacy threshold 模式下的 TRE 阈值，论文默认值为 `0.007`。
- `rank`：QwT 低秩 SVD 补偿的 rank；论文默认值为 `16`，当前代码默认值需要与论文实验统一。
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
- 补偿策略：先固定 Hessian、`keep_ratio=0.5`、`tail_ratio=0.01`

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
- `keep_ratio`：`0.25、0.50`
- `tail_ratio`：固定为 `0.01`
- 评估集：`7scenes-dense`、`DTU`

### 策略

| 方法 | 模块选择逻辑 |
|---|---|
| TRE | 按 tail relative error 排序 |
| MSE | 按普通均方误差排序 |
| Hessian | 按当前 Fisher/Hessian 近似分数排序 |
| Random | 随机选择相同数量的模块 |

### 输出图表

- 横轴：补偿模块比例 `keep_ratio`
- 纵轴：`Acc`、`Comp`、`NC`
- 四条曲线：TRE、MSE、Hessian、Random
- Random 使用 3 个 seed，并显示均值和标准差

四种方法必须保证实际补偿模块数量一致。

## 8. 阶段 5：\(\rho\)、`keep_ratio`、\(\tau\)、rank 敏感性

### 8.1 论文 \(\rho\) / `tail_ratio` 敏感性

论文中的 \(\rho\) 对应当前代码的 `compensate.tail_ratio`，不是 `keep_ratio`。

```text
tail_ratio = 0.001, 0.005, 0.010, 0.020,
              0.050, 0.100, 0.500, 1.000
```

固定 `keep_ratio=0.50`，输出：

- TRE 分布
- 被选择的模块
- 精度—`tail_ratio` 曲线
- 参数量和延迟变化

### 8.2 `keep_ratio` 模块比例敏感性

```text
keep_ratio = 0.0, 0.1, 0.25, 0.5, 0.75, 1.0
```

分别测试 TRE、MSE、Hessian、Random，输出：

- 精度—模块比例曲线
- 参数量—模块比例曲线
- 延迟—模块比例曲线

### 8.3 \(\tau\) 敏感性

当前代码中，设置 `select_metric` 后模块选择主要由 `keep_ratio` 决定，\(\tau\) 不再主导筛选。因此 \(\tau\) 实验应使用 legacy threshold 模式：

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

### 8.4 rank 敏感性

```text
rank = 8, 16, 32, 64, 128, 256
```

固定：

- 选择方法：Hessian
- `keep_ratio=0.5`
- `tail_ratio=0.01`
- W4A8
- 校准集：`DTU_20`

论文默认 rank 为 `16`，主结果必须单独保留 rank 16 配置。

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
├── projects/             独立项目：QuantVGGT 及其运行脚本
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

## 16. TMM 转投版总体目标

TMM 版本不只重复原论文结果，而是在 TAPTQ 原有三项贡献基础上补齐：

1. 公平的 FP、QuantVGGT、TAPTQ 对比；
2. channel-wise quantization 和量化粒度分析；
3. H800 上的 latency、memory、model size、QwT overhead；
4. TRE、MSE、Hessian、Random 的模块选择对比；
5. \(\rho\)/`tail_ratio`、`keep_ratio`、\(\tau\)、rank 敏感性；
6. ternary search 与 exhaustive search 的效率—精度 trade-off；
7. scene-level failure cases 和方法 limitations；
8. Dust3R、MASt3R 的 FP 跨模型基线。

TMM 版本的核心叙事建议从“提出一个 PTQ 方法”扩展为：

```text
面向 3D geometry foundation models 的可复现、可部署、跨架构 PTQ pipeline。
```

## 17. 论文—代码一致性检查

正式 H800 实验前必须完成以下核对：

### 17.1 校准集定义

论文描述的是：

```text
20 个 DTU training instances
→ progressive calibration construction
→ 8 个最终 calibration samples
```

本地 `data/dtu_20` 应作为 20 个样本的候选池。需要生成固定的校准 manifest，记录：

```text
pool path
pool size
selected scan IDs
selected frame IDs
frame sampling strategy
random seed
```

建议先核对现有 8-scan 配置是否就是论文中的最终 8 个样本：

```text
scan5, scan6, scan7, scan8,
scan18, scan22, scan56, scan123
```

不能只把 `dtu_20` 的所有帧直接当作论文中的 8-sample calibration set。

### 17.2 变量定义

必须严格区分：

| 论文/实验概念 | 当前代码变量 | 含义 |
|---|---|---|
| 论文 \(\rho\) | `compensate.tail_ratio` | TRE top-k 高幅值元素比例，默认 `0.01` |
| 模块补偿比例 | `compensate.keep_ratio` | 实际启用 QwT 的模块比例 |
| \(\tau\) | `compensate.tau_thr` | TRE threshold，论文默认 `0.007` |
| rank | `compensate.rank` | QwT SVD rank，论文默认 `16` |

论文数字和代码实验表中不要把 `tail_ratio` 写成 `keep_ratio`。

### 17.3 代码运行前检查

- 移除或参数化 `taptq.py` 中旧机器的模型硬编码路径；
- 统一 `VGGT_MODEL_PATH`、数据根目录和输出根目录；
- 确认 `datasets/__init__.py` 已存在，避免与 HuggingFace `datasets` 冲突；
- 将主结果默认 rank 对齐到 `16`；
- 确认每种 `(model, bit-width, granularity)` 只校准一次；
- 确认 TRE/MSE/Hessian/Random 复用同一 quantization checkpoint。

## 18. H800 统一实验协议

所有正式数字都在 H800 上重新统计。论文原 rebuttal 中的 B200 数字只作为历史参考，不与 H800 数字混用。

固定并记录：

- GPU 型号、显存、驱动和 CUDA 版本；
- PyTorch、Transformers、Open3D 等依赖版本；
- Git commit；
- 模型 checkpoint 路径和 SHA256；
- 数据集 manifest 和校准 manifest；
- `load_img_size`、视角数、帧 ID 和 AMP 设置；
- warmup 次数、正式计时次数和 CUDA event 计时方式。

建议每个实验目录保存：

```text
config.yaml
runtime.json
metrics.csv
stdout.log
stderr.log
git_commit.txt
nvidia-smi.txt
```

H800 开发机采用单卡独立进程，不默认使用 8 卡 DDP，以避免量化校准和部署测量互相污染。

## 19. H800 实验阶段和 GPU 分配

### 阶段 A：环境与 smoke test

只使用 GPU 0：

- 检查模型加载、数据读取和单序列 FP 推理；
- 检查量化包装、checkpoint 加载和输出格式；
- 确认 `DTU_20` manifest 与正式评估集不重叠。

### 阶段 B：主结果并行

| GPU | 实验 |
|---|---|
| 0 | FP VGGT / FP Pi3 |
| 1 | QuantVGGT W4A8 |
| 2 | QuantVGGT W8A8 |
| 3 | TAPTQ VGGT W4A8 per-tensor |
| 4 | TAPTQ VGGT W6A6 per-tensor |
| 5 | TAPTQ VGGT W4A8 channel-wise |
| 6 | TAPTQ Pi3 W4A8 |
| 7 | FP/Quant deployment benchmark |

### 阶段 C：补偿策略并行

量化 checkpoint 生成后，复用同一 checkpoint：

| GPU | 实验 |
|---|---|
| 0 | TRE，`keep_ratio=0.25` |
| 1 | TRE，`keep_ratio=0.50` |
| 2 | MSE，`keep_ratio=0.25` |
| 3 | MSE，`keep_ratio=0.50` |
| 4 | Hessian，`keep_ratio=0.25` |
| 5 | Hessian，`keep_ratio=0.50` |
| 6 | Random，seed 1/2 |
| 7 | Random，seed 3 和结果汇总 |

### 阶段 D：敏感性实验

| GPU | 实验 |
|---|---|
| 0 | `tail_ratio` sweep 低值区间 |
| 1 | `tail_ratio` sweep 高值区间 |
| 2 | `keep_ratio` sweep |
| 3 | `tau` sweep 低阈值 |
| 4 | `tau` sweep 高阈值 |
| 5 | rank 8/16/32 |
| 6 | rank 64/128/256 |
| 7 | 重复性检查和统计汇总 |

### 阶段 E：搜索、部署和外部 baseline

| GPU | 实验 |
|---|---|
| 0 | ternary search |
| 1 | exhaustive search |
| 2 | ternary + compensation |
| 3 | exhaustive + compensation |
| 4 | H800 FP/Quant latency and memory |
| 5 | Dust3R FP |
| 6 | MASt3R FP |
| 7 | failure cases 和复现实验 |

exhaustive search 可能是最慢的实验，应优先启动并单独记录实际耗时。

## 20. TMM 实验矩阵

### 20.1 主结果

| 模型 | 设置 | 数据集 | 目标 |
|---|---|---|---|
| VGGT | FP | 7Scenes、ETH3D、Co3Dv2 | FP baseline |
| VGGT | W4A8/W6A6/W8A8 | 7Scenes、ETH3D、Co3Dv2 | TAPTQ 主结果 |
| Pi3 | FP | 7Scenes、ETH3D | 跨模型主结果 |
| Pi3 | W4A8/W6A6 | 7Scenes、ETH3D | TAPTQ 跨模型结果 |
| QuantVGGT | W4A8/W8A8 | 7Scenes、DTU、ETH3D | 相关 baseline |
| Dust3R | FP | 统一点云评估集 | 外部 baseline |
| MASt3R | FP | 统一点云评估集 | 外部 baseline |

### 20.2 必做消融

- per-tensor vs channel-wise；
- TRE vs MSE/Hessian/Random；
- `tail_ratio`：`0.001、0.005、0.01、0.02、0.05、0.1、0.5、1.0`；
- `keep_ratio`：`0、0.1、0.25、0.5、0.75、1.0`；
- `tau`：`0、0.001、0.003、0.005、0.007、0.01、0.02、0.05`；
- rank：`8、16、32、64、128、256`；
- ternary vs exhaustive；
- calibration pool size：4/8/20；
- Random seed：1/2/3。

## 21. Deployment 测量设计

### PyTorch fake-quant

对比：

```text
FP32
FP16/BF16
W8A8
W8A16
W4A8
W4A8 + QwT
channel-wise W4A8
```

记录：

- model load time；
- calibration time；
- forward latency mean/std/min；
- peak allocated/reserved VRAM；
- state_dict/model size；
- 量化权重大小；
- QwT 参数量；
- 每模块和总 compensation overhead；
- throughput。

### ONNX/TensorRT

按以下顺序推进：

1. FP16 ONNX；
2. TensorRT FP16；
3. ONNX/TensorRT 数值一致性；
4. INT8/QDQ；
5. INT4 或自定义 kernel。

若最终只有 PyTorch fake-quant，必须在论文中明确写出它不等于真实 INT4 Tensor Core/TensorRT 部署。

## 22. Failure cases 与 limitations

每个失败案例必须保存：

```text
dataset
sequence
model
quant_config
tail_ratio
keep_ratio
tau
rank
error_type
log_path
repro_command
```

重点检查：

- 缺失图片、深度、mask、相机参数；
- seq-id-map 与数据不匹配；
- NaN、Inf、空 valid mask；
- interval 为 None；
- R² 小于 0；
- QwT 补偿后指标变差；
- Hessian calibration OOM；
- SVD OOM；
- ICP 不收敛；
- 点云为空或法向估计失败；
- ONNX/TensorRT 不支持算子；
- DTU 校准后其他数据集明显退化。

TMM 版本应明确以下限制：

1. fake-quant benchmark 不代表真实 INT4 硬件性能；
2. 当前 Hessian 分数是输出误差加权近似，不是完整二阶 Hessian；
3. ternary search 依赖近似 unimodal 假设，没有普适理论保证；
4. 尚未系统量化 TAPTQ 与 QAT 的差距；
5. calibration set 可能造成数据集偏差；
6. Dust3R、MASt3R 与 VGGT 的输出坐标系和后处理存在差异；
7. 当前验证的 3D 模型数量仍然有限。

## 23. 最终验收标准

实验完成需要满足：

- 所有原论文主表均有 H800 重跑数字；
- FP、QuantVGGT、TAPTQ 使用同一评估协议；
- 每种 `(model, bit-width, granularity)` 只校准一次并可复用；
- `tail_ratio`、`keep_ratio`、`tau`、rank 定义没有混淆；
- 主结果 rank 与论文默认 `rank=16` 对齐；
- 部署表包含 latency、memory、model size 和 QwT overhead；
- Random 至少有 3 个 seed；
- ternary 有 exhaustive 对照；
- Dust3R 和 MASt3R 至少有 FP 结果；
- 每个失败实验都有日志和复现命令；
- 结果可以由 H800、Git commit、模型版本和数据 manifest 复现。

## 24. TMM 论文实验表清单（按主题索引，待 Review）

> 本节保留按主题分类的完整表格清单。按论文正文实际叙事顺序的推荐排布见第 25 节。

本节只定义论文需要准备的表格，不填入最终实验数字。建议先由作者 review 表格数量、主文/补充材料分配和每张表的实验范围，再开始大规模补实验。

### 24.1 主文建议保留的核心表

建议 TMM 主文控制在 `6–8` 张核心表，避免把所有 sweep 都放进正文。

| 编号 | 建议表名 | 论文作用 | 必须包含的设置 | 主要指标 | 建议位置 | 优先级 |
|---|---|---|---|---|---|---|
| T1 | **Main Quantization Results on 3D Geometry Models** | 证明 TAPTQ 在主要模型和位宽下的整体有效性 | VGGT/Pi3；FP、TAPTQ W8A8/W6A6/W4A8；必要时加入 QuantVGGT | Acc、Comp、NC；mean/median | 主文 | P0 |
| T2 | **Comparison with FP and QuantVGGT** | 直接回应 rebuttal 中缺少公平 FP/QuantVGGT baseline 的问题 | FP VGGT、QuantVGGT、TAPTQ；至少 W4A8/W8A8 | Acc-mean、Comp-mean、NC-mean、相对 FP 变化 | 主文 | P0 |
| T3 | **Quantization Granularity: Per-Tensor vs Channel-Wise** | 证明方法不依赖单一量化粒度 | per-tensor；channel-wise；channel-wise + TAPTQ/QwT | 精度、校准时间、参数量、峰值显存 | 主文或主文 Ablation | P0 |
| T4 | **TRE-Guided Compensation vs Module Selection Baselines** | 证明 TRE 选择优于其他模块选择指标 | TRE、MSE、Hessian、Random；相同 W4A8、相同 keep ratio | Acc、Comp、NC；Random mean±std | 主文 | P0 |
| T5 | **Accuracy–Efficiency Deployment Trade-off on H800** | 给出真实目标硬件上的成本分析 | FP、W4A8、W6A6/W8A8；full model 与 aggregator-only | latency、peak memory、weight size、QwT overhead、speedup | 主文 | P0 |
| T6 | **Ternary Search Efficiency–Accuracy Trade-off** | 支撑 ternary search 的效率贡献 | exhaustive、ternary；有/无 compensation | calibration time、forward/search 次数、Acc、Comp、NC | 主文 | P0 |
| T7 | **Cross-Architecture Generalization** | 证明方法不只适用于 VGGT | VGGT、Pi3；可加入 Dust3R、MASt3R FP baseline | 统一 Acc、Comp、NC；模型/输入说明 | 主文 | P0/P1 |
| T8 | **Ablation of TAPTQ Components** | 归因三项技术组件的独立贡献 | baseline PTQ；+ progressive calibration；+ ternary search；+ TRE/QwT；full TAPTQ | Acc、Comp、NC、校准时间 | 主文 Ablation | P0 |

### 24.2 主文表格的统一字段

T1–T8 尽可能统一以下字段，避免不同表格使用不同统计口径：

```text
Model
Precision / Quantization
Weight activation bits
Granularity
Calibration set
Compensation
Tail ratio
Keep ratio
Tau
Rank
Dataset
Acc-mean / Acc-med
Comp-mean / Comp-med
NC-mean / NC-med
```

部署表另外记录：

```text
GPU
Input shape
Warmup / iterations
AMP dtype
Forward target
Latency mean/std/min
Peak allocated/reserved memory
Model/state_dict size
QwT parameter overhead
```

### 24.3 补充材料建议表

| 编号 | 建议表名 | 论文作用 | 实验内容 | 建议位置 | 优先级 |
|---|---|---|---|---|---|
| S1 | **Per-Dataset Main Results** | 展示不同数据集上的完整结果，避免只报告平均值 | 7Scenes sparse/dense、DTU、ETH3D、NRGBD（数据可用时） | Supplement | P0 |
| S2 | **Per-Scene Results** | 展示场景方差和最差序列 | 每个 scene/sequence 的 Acc、Comp、NC | Supplement | P1 |
| S3 | **Bit-Width Sweep** | 补齐位宽趋势 | W8A8、W6A6、W4A8、W4A6、W4A4 | Supplement | P0 |
| S4 | **Calibration Set Size and Selection** | 验证 calibration set 构造是否关键 | 4-scan、8-scan、20-scan；随机/固定/progressive | Supplement | P0 |
| S5 | **`tail_ratio` / Paper \(\rho\) Sensitivity** | 解释论文中的 \(\rho\) 对 TRE tail 的影响 | `tail_ratio=0.001…1.0`；固定 `keep_ratio` | Supplement | P1 |
| S6 | **`keep_ratio` Sensitivity** | 解释实际补偿模块比例对精度和开销的影响 | `keep_ratio=0、0.1、0.25、0.5、0.75、1.0` | Supplement | P0 |
| S7 | **`tau` Threshold Sensitivity** | 验证 threshold 模式下的稳定性 | `tau=0、0.001、0.003、0.005、0.007、0.01、0.02、0.05` | Supplement | P1 |
| S8 | **Low-Rank Rank Sensitivity** | 证明 rank=16 的精度—开销折中 | rank `8/16/32/64/128/256` | Supplement | P0 |
| S9 | **Random Selection Repeatability** | 量化 Random baseline 的方差 | seed `1/2/3`，保持模块数量一致 | Supplement | P0 |
| S10 | **Compensation Granularity** | 比较 QwT 的补偿粒度 | layer、block、module | Supplement | P1 |
| S11 | **Deployment Detailed Breakdown** | 展示主文部署表的完整开销拆分 | load、calibration、forward、QwT、memory、model size | Supplement | P1 |
| S12 | **Fake-Quant vs TensorRT/ONNX** | 明确 fake-quant 与真实部署的差异 | PyTorch fake-quant、ONNX FP16、TensorRT FP16/INT8/INT4（可行时） | Supplement | P1 |
| S13 | **Failure Cases and Limitations** | 透明报告失败实验和退化场景 | OOM、NaN/Inf、ICP failure、QwT 退化、部署算子不支持 | Supplement | P1 |
| S14 | **Reproducibility and Runtime Configuration** | 便于审稿人复现 | GPU、软件版本、commit、model hash、manifest、命令 | Supplement | P0 |

### 24.4 每张表的推荐具体结构

#### T1：Main Quantization Results

| Model | Setting | W/A | Calibration | Compensation | Dataset | Acc-mean ↓ | Comp-mean ↓ | NC-mean ↑ |
|---|---|---|---|---|---|---:|---:|---:|
| VGGT | FP | FP | – | – | 7Scenes/DTU/ETH3D |  |  |  |
| VGGT | TAPTQ | W8A8 | DTU calibration | QwT | 7Scenes/DTU/ETH3D |  |  |  |
| VGGT | TAPTQ | W6A6 | DTU calibration | QwT | 7Scenes/DTU/ETH3D |  |  |  |
| VGGT | TAPTQ | W4A8 | DTU calibration | QwT | 7Scenes/DTU/ETH3D |  |  |  |
| Pi3 | FP/TAPTQ | 对应设置 | DTU calibration | 对应设置 | 统一数据集 |  |  |  |

#### T2：FP / QuantVGGT / TAPTQ

| Model | Quantizer | W/A | Evaluator | Calibration | Acc-mean ↓ | Comp-mean ↓ | NC-mean ↑ | ΔAcc vs FP |
|---|---|---|---|---|---:|---:|---:|---:|
| VGGT | FP | FP | Unified | – |  |  |  |  |
| QuantVGGT | Official | W4A8 | Unified | Official/cache |  |  |  |  |
| TAPTQ | Ours | W4A8 | Unified | DTU fixed manifest |  |  |  |  |
| VGGT | FP | W8A8 equivalent | Unified | – |  |  |  |  |
| QuantVGGT | Official | W8A8 | Unified | Official/cache |  |  |  |  |
| TAPTQ | Ours | W8A8 | Unified | DTU fixed manifest |  |  |  |  |

`Evaluator` 必须注明是否使用同一个点云读取、Umeyama、ICP 和法向估计流程；如果不能统一，T2 应拆成“严格公平对比”和“原始实现 sanity check”两张表。

#### T3：Per-Tensor / Channel-Wise

| Model | Granularity | W/A | Compensation | Calibration time | Acc-mean ↓ | Comp-mean ↓ | NC-mean ↑ | Peak VRAM | Param/size overhead |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| VGGT | Per-tensor | W4A8 | None |  |  |  |  |  |  |
| VGGT | Channel-wise | W4A8 | None |  |  |  |  |  |  |
| VGGT | Channel-wise | W4A8 | QwT |  |  |  |  |  |  |

#### T4：TRE / MSE / Hessian / Random

| Selection metric | `keep_ratio` | `tail_ratio` | Rank | Random seed | # compensated modules | Acc-mean ↓ | Comp-mean ↓ | NC-mean ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TRE | 0.25 | 0.01 | 16 | – |  |  |  |  |
| MSE | 0.25 | 0.01 | 16 | – |  |  |  |  |
| Hessian | 0.25 | 0.01 | 16 | – |  |  |  |  |
| Random | 0.25 | 0.01 | 16 | 1/2/3 |  |  |  |  |
| TRE/MSE/Hessian/Random | 0.50 | 0.01 | 16 | 对应设置 |  |  |  |  |

TRE、MSE、Hessian 和 Random 必须固定实际补偿模块数量；不能只固定名义上的 `keep_ratio` 而导致模块数量不同。

#### T5：H800 Deployment

| Model | Path | W/A | Forward target | Latency mean/std (ms) | Peak allocated (GiB) | Peak reserved (GiB) | State/model size | QwT overhead | Note |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| VGGT | PyTorch FP | FP | Full |  |  |  |  | – | FP baseline |
| TAPTQ | PyTorch fake-quant | W4A8 | Full |  |  |  |  |  | Not real INT4 |
| TAPTQ | PyTorch fake-quant | W4A8 | Aggregator |  |  |  |  |  | Matches quantized scope |
| TAPTQ | TensorRT/ONNX | W4A8/INT4 | Full |  |  |  |  |  | Only if implemented |

必须同时记录 GPU 型号、输入 shape、warmup、iterations 和 AMP dtype；否则不同 latency 数字不可比较。

#### T6：Ternary Search

| Search method | `search_round` / `eq_n` | Compensation | Calibration time | Search forward count | Acc-mean ↓ | Comp-mean ↓ | NC-mean ↑ | Peak VRAM |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Exhaustive | 原论文配置 | None |  |  |  |  |  |  |
| Ternary | `search_round=1/2/3/4` | None |  |  |  |  |  |  |
| Exhaustive | 原论文配置 | QwT |  |  |  |  |  |  |
| Ternary | 最终推荐配置 | QwT |  |  |  |  |  |  |

#### T7：Cross-Architecture Generalization

| Model | Model status | Quantization | Dataset | Unified output conversion | Acc-mean ↓ | Comp-mean ↓ | NC-mean ↑ | Latency | Peak VRAM |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| VGGT | FP | – | 统一数据集 | – |  |  |  |  |  |
| Pi3 | FP/TAPTQ | FP/W4A8 | 统一数据集 | Required |  |  |  |  |  |
| Dust3R | FP | – | 统一数据集 | Required |  |  |  |  |  |
| MASt3R | FP | – | 统一数据集 | Required |  |  |  |  |  |

Dust3R/MASt3R 第一阶段只要求 FP baseline；不应在尚未完成公平 FP 评估前直接加入量化结果。

### 24.5 建议删除或合并的表

为了控制 TMM 主文篇幅，以下内容不建议各自单独占一张主文表：

1. `tail_ratio`、`keep_ratio`、`tau`、rank 四个 sweep 不要全部放主文，可合并为一张 Supplement 综合敏感性表；
2. 每个 scene 的完整数字放 Supplement，主文只放 mean/median 和 worst-case；
3. deployment 的 load/calibration/forward/QwT 详细拆分放 Supplement，主文保留最终 latency/memory/overhead；
4. failure cases 不建议混入主结果表，单独放 Supplement 的 failure/limitation 表；
5. QuantVGGT 如果暂时无法接入统一 evaluator，应单独标记为 `upstream evaluator`，不要和统一协议结果混排。

### 24.6 论文表格最低完成标准

在 TMM 初稿中，至少应完成以下表格后再组织最终叙事：

- [ ] T1：FP、TAPTQ 多位宽主结果；
- [ ] T2：FP / QuantVGGT / TAPTQ 公平对比；
- [ ] T3：per-tensor / channel-wise；
- [ ] T4：TRE / MSE / Hessian / Random；
- [ ] T5：H800 latency / memory / overhead；
- [ ] T6：ternary vs exhaustive；
- [ ] T7：至少 VGGT/Pi3，最好加入 Dust3R/MASt3R FP；
- [ ] S1：按数据集完整结果；
- [ ] S3：位宽 sweep；
- [ ] S4：calibration size/selection；
- [ ] S6：keep ratio；
- [ ] S8：rank；
- [ ] S9：Random seeds；
- [ ] S11：部署详细开销；
- [ ] S13：failure cases；
- [ ] S14：reproducibility configuration。

### 24.7 待作者 Review 的关键选择

请重点 review 以下问题：

1. 主文是否保留 `6–8` 张表，还是希望更精简到 `5–6` 张？
2. T1 和 T2 是否合并为一张主结果表？如果合并，表格可能会比较宽；
3. Pi3 是否作为主文结果，还是只放 Supplement？
4. Dust3R/MASt3R 是否只做 FP baseline，还是必须进入主文表？
5. T5 是否只报告 H800 PyTorch fake-quant，还是等待 TensorRT 结果后再写 deployment 结论？
6. `tail_ratio` 是否继续使用论文符号 \(\rho\)，并在表注中明确它不等于 `keep_ratio`？
7. 统一 evaluator 如果暂时无法覆盖 QuantVGGT，是否接受在表中分栏标注 `upstream evaluator`？
8. TMM 页数限制下，哪些 sweep 可以全部移到 Supplement？

## 25. 按论文正文逻辑排布的实验表格

本节是推荐的**实际写论文顺序**。表格不再按“实验类型”平铺，而是按照论文从整体结论、方法归因、细节分析到部署和泛化的叙事展开。建议正文使用 `Table 1–9`，补充材料使用 `Table S1–S14`。

### 25.1 正文整体叙事

```text
实验设置与公平协议
        ↓
整体主结果：TAPTQ 是否有效
        ↓
与现有量化方法和 QuantVGGT 比较
        ↓
组件消融：性能提升来自哪里
        ↓
校准集与 ternary search：为什么校准高效
        ↓
TRE/QwT：为什么补偿有效
        ↓
位宽与 channel-wise：方法是否稳健
        ↓
H800 部署：代价是否可接受
        ↓
Pi3/Dust3R/MASt3R：是否跨模型泛化
```

### 25.2 正文推荐表格顺序

#### Table 1：Experimental Setup and Evaluation Protocol

**目的**：先让读者知道所有结果是在什么模型、数据集、硬件和协议下得到的。该表不承担方法效果结论，而是建立公平比较的边界。

| Model | Task | Dataset | Input / views | Evaluation metrics | Quantization scope | Calibration set | Hardware |
|---|---|---|---|---|---|---|---|
| VGGT | Multi-view reconstruction | 7Scenes / DTU / ETH3D | 固定视角数与分辨率 | Acc / Comp / NC | Aggregator | DTU fixed manifest | H800 |
| Pi3 | Multi-view reconstruction | 同上 | 同上 | 同上 | 对应量化模块 | DTU fixed manifest | H800 |
| QuantVGGT | Multi-view reconstruction | 统一可用数据集 | 与统一协议一致 | 同上 | Official scope | Official/cache | H800 |
| Dust3R / MASt3R | FP baseline | 统一可用数据集 | 记录模型原生输入 | 同上 | FP | – | H800 |

表注必须说明：

- 所有 H800 数字使用的 GPU、输入分辨率、视角数、帧采样和后处理；
- `tail_ratio`（论文中的 \(\rho\)）与 `keep_ratio` 的区别；
- QuantVGGT 如果尚未接入统一 evaluator，必须在表中明确标注 `upstream evaluator`。

#### Table 2：Main Results on 3D Geometry Models

**目的**：论文第一个结果表，直接回答“低比特量化后是否仍保持 3D 几何重建能力”。这是全文最重要的精度表。

推荐只放最具代表性的设置，避免把所有 sweep 放在这里：

| Model | Method | W/A | Compensation | 7Scenes | DTU | ETH3D |
|---|---|---|---|---|---|---|
| VGGT | FP | FP | – | Acc / Comp / NC | Acc / Comp / NC | Acc / Comp / NC |
| VGGT | PTQ baseline | W4A8 | None |  |  |  |
| VGGT | TAPTQ | W4A8 | QwT |  |  |  |
| VGGT | TAPTQ | W6A6 | QwT |  |  |  |
| Pi3 | FP | FP | – |  |  |  |
| Pi3 | TAPTQ | W4A8 | QwT |  |  |  |

建议正文报告 `mean` 和 `median`，最差场景放 Supplement。主结果必须使用正式校准 manifest 和统一 evaluator。

#### Table 3：Comparison with Existing Quantization Methods

**目的**：证明 TAPTQ 相比已有 PTQ 方法的增益，而不是只与 FP 比较。

应纳入论文已有或可复现的 baseline：`RTN`、`PTQ4ViT`、`ERQ`、`RepQ`、`GPTQ`，并加入 `QuantVGGT` 作为相关 3D geometry quantization baseline。

| Model | Method | W/A | Calibration | Compensation | Acc-mean ↓ | Comp-mean ↓ | NC-mean ↑ | Relative cost |
|---|---|---|---|---|---:|---:|---:|---:|
| VGGT | FP | FP | – | – |  |  |  |  |
| VGGT | RTN | W4A8 | Same protocol | – |  |  |  |  |
| VGGT | PTQ4ViT | W4A8 | Same protocol | – |  |  |  |  |
| VGGT | ERQ / RepQ / GPTQ | W4A8 | Same protocol | – |  |  |  |  |
| VGGT | QuantVGGT | W4A8 | Official/cache | Official |  |  |  |  |
| VGGT | TAPTQ | W4A8 | DTU fixed manifest | QwT |  |  |  |  |

如果某个 baseline 无法在完全相同的实现和 evaluator 下复现，不能伪装成严格公平数字，应在表中增加 `Protocol` 或 `Source` 字段并单独标注。

#### Table 4：Ablation of TAPTQ Components

**目的**：在主结果之后解释性能提升来自哪些组件。建议采用逐步添加组件的写法，而不是将三个组件分散到不同章节。

| Setting | Progressive calibration | Ternary search | TRE selection | QwT compensation | Calibration time | Acc-mean ↓ | Comp-mean ↓ | NC-mean ↑ |
|---|---|---|---|---|---:|---:|---:|---:|
| PTQ baseline | – | – | – | – |  |  |  |  |
| + progressive calibration | Yes | – | – | – |  |  |  |  |
| + ternary search | Yes | Yes | – | – |  |  |  |  |
| + TRE selection | Yes | Yes | Yes | – |  |  |  |  |
| Full TAPTQ | Yes | Yes | Yes | Yes |  |  |  |  |

这里应固定模型、位宽、评估集和补偿预算；否则该表无法完成组件归因。

#### Table 5：Calibration Set and Search Efficiency

**目的**：解释 TAPTQ 为什么可以降低校准代价，并承接方法章节中的 progressive calibration 和 ternary search。

| Calibration setting | Pool size | Selected size | Search method | Search evaluations | Calibration time | Acc-mean ↓ | Comp-mean ↓ |
|---|---:|---:|---|---:|---:|---:|---:|
| Exhaustive + 20-scan | 20 | 8/20 | Exhaustive |  |  |  |  |
| Ternary + 20-scan | 20 | 8/20 | Ternary |  |  |  |  |
| Ternary + 8-scan | 20 | 8 | Ternary |  |  |  |  |
| Ternary + 4-scan | 20 | 4 | Ternary |  |  |  |  |

该表应同时回答两个问题：

1. ternary search 相比 exhaustive 是否显著减少时间/搜索次数；
2. calibration set 从 20 个候选样本缩减到 8 个或 4 个后，精度损失是否可接受。

#### Table 6：TRE-Guided Compensation and Selection Baselines

**目的**：承接“量化误差具有模块不均匀性”的分析，证明 TRE 比 MSE、Hessian 和 Random 更适合选择补偿模块。

| Selection metric | `keep_ratio` | `tail_ratio` / \(\rho\) | Rank | # modules | Acc-mean ↓ | Comp-mean ↓ | NC-mean ↑ | QwT overhead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| None | 0 | – | – | 0 |  |  |  |  |
| TRE | 0.25 | 0.01 | 16 |  |  |  |  |  |
| MSE | 0.25 | 0.01 | 16 |  |  |  |  |  |
| Hessian | 0.25 | 0.01 | 16 |  |  |  |  |  |
| Random | 0.25 | 0.01 | 16 |  |  |  |  |  |
| TRE | 0.50 | 0.01 | 16 |  |  |  |  |  |
| MSE / Hessian / Random | 0.50 | 0.01 | 16 |  |  |  |  |  |

Random 应报告 seed `1/2/3` 的 mean±std；四种策略必须使用完全相同的补偿模块数量。

#### Table 7：Robustness to Bit-Width and Quantization Granularity

**目的**：在读者已经看到 TAPTQ 有效、且知道其组件来源之后，再展示它对量化配置的稳健性。

| Method | W/A | Granularity | Compensation | Acc-mean ↓ | Comp-mean ↓ | NC-mean ↑ | Calibration time | Model size |
|---|---|---|---|---:|---:|---:|---:|---:|
| TAPTQ | W8A8 | Per-tensor | QwT |  |  |  |  |  |
| TAPTQ | W6A6 | Per-tensor | QwT |  |  |  |  |  |
| TAPTQ | W4A8 | Per-tensor | QwT |  |  |  |  |  |
| TAPTQ | W4A8 | Channel-wise | QwT |  |  |  |  |  |
| TAPTQ | W4A4 | Per-tensor | QwT/None |  |  |  |  |  |

Channel-wise 结果必须先解决 checkpoint 与 wrapper 的 shape 对齐问题；不能使用当前已知不兼容的参数文件直接填表。

#### Table 8：H800 Deployment Cost and Accuracy–Efficiency Trade-off

**目的**：最后回答方法是否具有实际部署价值。该表放在精度和消融之后，避免读者把 fake-quant latency 误解成方法精度结论。

| Model | Runtime path | W/A | Forward target | Latency mean/std (ms) | Peak memory (GiB) | Weight/model size | QwT overhead | Speedup |
|---|---|---|---|---:|---:|---:|---:|---:|
| VGGT | PyTorch FP | FP | Full |  |  |  | – | 1.00× |
| TAPTQ | PyTorch fake-quant | W4A8 | Full |  |  |  |  |  |
| TAPTQ | PyTorch fake-quant | W4A8 | Aggregator |  |  |  |  |  |
| TAPTQ | PyTorch fake-quant | W6A6/W8A8 | Full |  |  |  |  |  |
| TAPTQ | TensorRT/ONNX | INT8/INT4 | Full |  |  |  |  |  |

如果没有真实 TensorRT/INT4 实现，表中最后一行删除，并在表注明确：当前结果是 PyTorch fake-quant，不代表真实 INT4 kernel 的延迟。

#### Table 9：Cross-Model Generalization

**目的**：作为全文最后的扩展实验，说明方法是否可迁移到其他 3D geometry foundation models。

| Model | Model type | Setting | Dataset | Acc-mean ↓ | Comp-mean ↓ | NC-mean ↑ | Latency | Peak VRAM |
|---|---|---|---|---:|---:|---:|---:|---:|
| VGGT | Main model | FP/TAPTQ W4A8 | Unified set |  |  |  |  |  |
| Pi3 | Related geometry model | FP/TAPTQ W4A8 | Unified set |  |  |  |  |  |
| Dust3R | External FP baseline | FP | Unified set |  |  |  |  |  |
| MASt3R | External FP baseline | FP | Unified set |  |  |  |  |  |

Dust3R 和 MASt3R 第一阶段只做 FP baseline；需要统一输出格式、坐标系、尺度对齐、ICP 和指标计算，不能直接把原始官方数字放进该表。

### 25.3 补充材料的排布顺序

补充材料也应按照正文顺序展开，而不是按代码目录排列：

| Supplement 编号 | 对应正文表 | 补充内容 |
|---|---|---|
| Table S1 | Table 1 | 完整环境、数据集、输入、采样和 evaluator 配置 |
| Table S2 | Table 2/3 | 每个数据集的完整 mean/median 结果 |
| Table S3 | Table 2/3 | 每个 sequence 的结果、最差场景和失败场景 |
| Table S4 | Table 4 | 各组件单独 ablation 的完整矩阵 |
| Table S5 | Table 5 | 4/8/20-scan、随机/固定/progressive calibration 对比 |
| Table S6 | Table 5 | exhaustive/ternary 的 `search_round`、`eq_n`、forward count 详情 |
| Table S7 | Table 6 | TRE/MSE/Hessian/Random 在不同 `keep_ratio` 下的完整结果 |
| Table S8 | Table 6/7 | `tail_ratio`、`tau`、rank 敏感性 |
| Table S9 | Table 7 | Bit-width、channel-wise、补偿粒度和 checkpoint size 详情 |
| Table S10 | Table 8 | H800 load/calibration/forward/QwT/memory 详细拆分 |
| Table S11 | Table 8 | Fake-quant、ONNX、TensorRT 的实现和算子支持情况 |
| Table S12 | Table 9 | Dust3R/MASt3R 输入适配、坐标变换和 FP baseline 详情 |
| Table S13 | 全部 | Failure cases、OOM、NaN/Inf、ICP failure、QwT 退化 |
| Table S14 | 全部 | Git commit、模型 hash、数据 manifest、运行命令和环境版本 |

### 25.4 论文写作时的推荐章节—表格对应关系

| 论文小节 | 首要表格 | 读者应该得到的结论 |
|---|---|---|
| 4.1 Experimental Setup | Table 1 | 实验协议是固定且公平的 |
| 4.2 Main Results | Table 2 | TAPTQ 在主要模型和位宽下有效 |
| 4.3 Comparison with Existing Methods | Table 3 | TAPTQ 相比 PTQ/QuantVGGT baseline 有竞争力 |
| 4.4 Component Ablation | Table 4 | progressive calibration、ternary、TRE/QwT 各自有贡献 |
| 4.5 Calibration Efficiency | Table 5 | 校准样本压缩和 ternary search 降低成本 |
| 4.6 Compensation Analysis | Table 6 | TRE 选择能以较小开销恢复量化误差 |
| 4.7 Robustness Analysis | Table 7 | 不同位宽和量化粒度下仍然稳定 |
| 4.8 Deployment Analysis | Table 8 | H800 上的显存、延迟和额外开销可量化 |
| 4.9 Cross-Model Generalization | Table 9 | 方法具有跨模型迁移潜力，同时诚实报告适配成本 |
| Supplement | S1–S14 | 完整数字、敏感性、失败案例和复现细节 |

### 25.5 这一版建议的最小正文表数量

如果 TMM 正文篇幅有限，建议压缩为 `6` 张表：

1. `Table 1`：Experimental setup；
2. `Table 2`：Main results + FP/QuantVGGT comparison；
3. `Table 3`：Existing PTQ baseline + component ablation；
4. `Table 4`：Calibration efficiency + ternary search；
5. `Table 5`：TRE/MSE/Hessian/Random + bit-width/channel-wise；
6. `Table 6`：H800 deployment + cross-model generalization。

如果篇幅允许，建议保留 `9` 张表，因为这样每张表承担一个明确论点，论文叙事更清楚。