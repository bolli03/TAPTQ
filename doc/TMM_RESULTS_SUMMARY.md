# TMM / 多模型量化实验结果汇总

> 更新时间：2026-08-19 13:23（实验节点 CST）  
> 本文只纳入有日志、指标文件或 checkpoint 可追溯的结果；7Scenes 是当前主报告集。ETH3D 为当前 E1 adapter 口径，与论文/PDF 参考协议仍有差异，因此作为辅助结果，不据此单独作算法因果判断。

## 目录

- [1. 执行摘要](#1-执行摘要)
- [2. 统一评估协议](#2-统一评估协议)
- [3. VGGT 统一主表](#3-vggt-统一主表)
- [4. 跨基座模型与指定补充实验](#4-跨基座模型与指定补充实验)
- [5. 校准集与关键实现消融](#5-校准集与关键实现消融)
- [6. TAPTQ 补偿结果与消融](#6-taptq-补偿结果与消融)
- [7. 附录：Quant-only、PDF 与旧协议结果](#7-附录quant-onlypdf-与旧协议结果)
- [8. H800 fused W8A8 与 W4A8 hybrid 部署](#8-h800-fused-w8a8-与-w4a8-hybrid-部署)
- [9. 结果与产物索引](#9-结果与产物索引)

## 1. 执行摘要

- **状态：**H800 lease Run `714219558` 上已完成修正后的 7Scenes 主表重测，GPU0--GPU7 当前均为空闲；该 lease 保持运行。VGGT-Omega 仍因官方 gated checkpoint 未获授权而阻塞。实时状态只以 `doc/H800_LEASE_GPU_STATUS.md` 为准。
- **VGGT 精度：**7Scenes 主表现统一采用官方测试划分的 18 个序列、`kf=200`、RGB 相机内参 `fx=fy=525`、投影深度及正确的横纵内参缩放。所有 25 个配置均已生成逐序列结果；旧 `7Scenes-dense/kf40` 数值不再用于主表。
- **新基座 OpenD4RT：**7Scenes full18 的 FP 为 `0.039151/0.049335/0.647854`；channel-wise W4A8 为 `0.039429/0.048970/0.645904`，基本保持 FP；non-channel-wise W4A8 为 `0.099541/0.182797/0.589010`，明显退化。详见 §4.3。
- **QwT：**外部模型计划搜索 204 个组合，形成 202 条有效筛选结果；失败或跳过项不参与选优。DUSt3R 的门控 channel-wise QwT 在 full18 上有小幅收益，MASt3R 与 clean quant-only 相同；non-channel-wise + QwT 在 DUSt3R、MASt3R 上明显退化。Pi3 non-channel-wise + QwT 的 Acc/NC 退化而 Comp 改善，不推荐作为默认配置。
- **部署：**H800 fused W8A8 已实现。低延迟 single 模式在 8-view 为 `147.225 ms`，同口径 BF16 为 `149.506 ms`，首次实现 **`1.015×` 高于 BF16**；full18 为 `0.017546/0.032976/0.689514`。精确 dual-scale 模式末层 cosine `0.999188`，但 8-view 为 `157.767 ms`。此前 W4/A8-calibrated hybrid FP8/BF16 为 `338.172 ms`，并非原生 fused INT4×INT8。详见 §8。
- **实现审计：**ERQ 补齐 `qkv/fc1` two-part 后显著改善；SmoothQuant W4A8 以 `alpha=.3` 最好。固定 4096-token 预算下，DTU20 未带来一致收益，部分指标改善、部分退化。旧含 QwT hooks 的 quant-only checkpoint 已排除，最终基线均采用 clean checkpoint + `--disable-qwt`。

## 2. 统一评估协议

| 项目 | 设置 |
|---|---|
| 主模型 | VGGT-1B；另测 Pi3、DUSt3R、MASt3R |
| 评估器 | E1 unified evaluator |
| 输入分辨率 | VGGT/Pi3：518；DUSt3R/MASt3R：原生 512 预处理 |
| 7Scenes 主表 | Pi3 evaluation：官方 test split，18 个序列，`kf=40`，`7scenes_mv-recon_seq-id-map-kf40.json` |
| ETH3D | 13 个序列；当前 E1 adapter 辅助口径，尚未与论文/PDF 参考协议完全对齐 |
| 指标 | Acc、Comp 越低越好；NC 越高越好；NC=`(NC1+NC2)/2` |
| 数值格式 | `Mean / Median`；仅有 Mean 时明确标注 |
| 默认校准 | `DTU_train_8`，4096 tokens/layer |
| 共同量化范围 | VGGT aggregator 的 288 个 `qkv/proj/fc1/fc2` Linear |
| TAPTQ 推荐参数 | W4A8 ternary，non-channel-wise，`rho=0.1, tau=0.007, rank=128` |

## 3. VGGT 统一主表

以下结果均采用 Run `714219558` 上截图对应的 Pi3 evaluation 协议：7Scenes test split、18 个序列、`kf=40`、`7scenes_mv-recon_seq-id-map-kf40.json`、518 输入和统一 E1 evaluator。W6A6/W8A8 的 PTQ4ViT 与 TAPTQ 已重新执行；TAPTQ 使用 `mode=compensate_eval`，并确认 checkpoint 加载为 `0 missing, 0 unexpected keys`。

| Bit | 方法 | Acc Mean/Med. | Comp Mean/Med. | NC Mean/Med. |
|---|---|---:|---:|---:|
| — | FP | 0.0185 / 0.0156 | 0.0318 / 0.0272 | 0.6895 / 0.6859 |
| W4A8 | RTN | 0.0317 / 0.0124 | 0.0433 / 0.0158 | 0.6849 / 0.7868 |
| W4A8 | PTQ4ViT | 0.0343 / 0.0152 | 0.0547 / 0.0190 | 0.6877 / 0.7859 |
| W4A8 | RepQ-ViT | 0.0872 / 0.0486 | 0.1939 / 0.0892 | 0.6375 / 0.7073 |
| W4A8 | ERQ two-part | 0.0340 / 0.0158 | 0.0367 / 0.0147 | 0.6839 / 0.7857 |
| W4A8 | GPTQ | 0.0304 / 0.0127 | 0.0345 / 0.0152 | 0.6868 / 0.7897 |
| W4A8 | SmoothQuant | 0.0337 / 0.0131 | 0.0398 / 0.0139 | 0.6883 / 0.7914 |
| W4A8 | QuantVGGT | 0.0207 / 0.0082 | 0.0342 / 0.0176 | 0.6739 / 0.7595 |
| W4A8 | TAPTQ, `rho=.1/tau=.007/r128` | **0.0189 / 0.0081** | 0.0359 / 0.0140 | **0.6943 / 0.7983** |
| W6A6 | RTN | 0.0420 / 0.0179 | 0.0403 / 0.0157 | 0.6775 / 0.7754 |
| W6A6 | PTQ4ViT | 0.0205 / 0.0092 | 0.0355 / 0.0143 | 0.6918 / 0.7955 |
| W6A6 | RepQ-ViT | 0.0429 / 0.0186 | 0.0455 / 0.0157 | 0.6799 / 0.7788 |
| W6A6 | ERQ two-part | 0.0364 / 0.0173 | 0.0407 / 0.0158 | 0.6825 / 0.7830 |
| W6A6 | GPTQ | 0.0404 / 0.0172 | 0.0416 / 0.0150 | 0.6774 / 0.7752 |
| W6A6 | SmoothQuant | 0.0328 / 0.0163 | 0.0455 / 0.0178 | 0.6806 / 0.7813 |
| W6A6 | QuantVGGT | 0.0203 / 0.0082 | 0.0339 / 0.0171 | 0.6749 / 0.7650 |
| W6A6 | TAPTQ, `rho=.1/tau=.007/r16` | 0.0203 / 0.0086 | **0.0349 / 0.0146** | 0.6904 / 0.7935 |
| W8A8 | RTN | 0.0216 / 0.0089 | 0.0353 / 0.0166 | 0.6878 / 0.7899 |
| W8A8 | PTQ4ViT | **0.0181 / 0.0074** | **0.0310 / 0.0152** | **0.6899 / 0.7929** |
| W8A8 | RepQ-ViT | 0.0413 / 0.0180 | 0.0410 / 0.0161 | 0.6817 / 0.7810 |
| W8A8 | ERQ two-part | 0.0335 / 0.0162 | 0.0363 / 0.0156 | 0.7183 / 0.8077 |
| W8A8 | GPTQ | 0.0213 / 0.0088 | 0.0355 / 0.0167 | 0.6880 / 0.7905 |
| W8A8 | SmoothQuant | 0.0230 / 0.0091 | 0.0367 / 0.0170 | 0.6859 / 0.7867 |
| W8A8 | QuantVGGT | 0.0205 / 0.0082 | 0.0338 / 0.0173 | 0.6745 / 0.7619 |
| W8A8 | TAPTQ, `rho=.1/tau=.007/r16` | 0.0182 / 0.0074 | 0.0311 / 0.0152 | 0.6897 / 0.7926 |

精确重测产物位于 `/mnt/cephfs/pansicheng/tmm-results/eval/7scenes_pi3_kf40_recheck/`。其中：PTQ4ViT W6A6 为 `0.0205067/0.0355230/0.6917683`，TAPTQ W6A6 为 `0.0202783/0.0348914/0.6904147`；PTQ4ViT W8A8 为 `0.0181290/0.0310153/0.6899118`，TAPTQ W8A8 为 `0.0181753/0.0310625/0.6896735`。

## 4. 跨基座模型与指定补充实验

以下表格统一报告 7Scenes full18 Mean；`—` 表示没有同口径可追溯结果。跨模型最终选择只在本节维护，避免与后文重复。

### 4.1 FP、clean quant-only 与 channel-wise QwT 最终选择

| 模型 | 方法 | 配置 | Acc / Comp / NC | 结论 |
|---|---|---|---:|---|
| VGGT | FP | 518 输入 | 0.018530 / 0.031756 / 0.689451 | FP 参考 |
| Pi3 | FP | 518 输入 | **0.015044 / 0.022652 / 0.685400** | FP 参考 |
| DUSt3R | FP | 原生 512 | 0.019343 / 0.029319 / 0.680308 | FP 参考 |
| MASt3R | FP | 原生 512 | 0.025488 / 0.031090 / 0.665829 | FP 参考 |
| OpenD4RT | FP | 32CLIP checkpoint，256 原生前向 | 0.039151 / 0.049335 / 0.647854 | FP 参考 |
| OpenD4RT | non-channel-wise quant-only | W4A8 ternary，DTU8 | 0.099541 / 0.182797 / 0.589010 | 明显退化，不推荐 |
| OpenD4RT | channel-wise quant-only | W4A8 ternary，DTU8 | **0.039429 / 0.048970 / 0.645904** | 基本保持 FP，推荐量化配置 |
| DUSt3R | channel-wise clean quant-only | W4A8，224 校准/512 评估 | 0.024119 / 0.027904 / 0.673588 | 最终基线 |
| DUSt3R | channel-wise QwT best | rho=.001/tau=.007/fit≤.6/r16 | **0.023192 / 0.027874 / 0.674361** | 三项均略优于基线，推荐 |
| DUSt3R | channel-wise QwT alternate | rho=.01/tau=.003/fit≤.6/r16 | 0.024533 / **0.027305** / 0.673123 | Comp 最优备选 |
| MASt3R | channel-wise clean quant-only | W4A8，224 校准/512 评估 | **0.032244 / 0.028564 / 0.654313** | 最终推荐 |
| MASt3R | channel-wise QwT best | rho=.01/tau=.01/fit≤.2/r16 | 0.032244 / 0.028564 / 0.654313 | 与 clean 基线相同 |
| MASt3R | channel-wise QwT alternate | rho=.001/tau=.003/fit≤.4/r16 | 0.032244 / 0.028564 / 0.654313 | 与 clean 基线相同 |

### 4.2 用户指定四项实验归档

下表产物路径均相对 `/mnt/cephfs/pansicheng/tmm-results/`；完整配置和绝对路径同步记录在 `doc/tmm_test_manifest.json`。

| 模型 | CW | QwT | 配置 | Acc / Comp / NC | 结论 | 产物 |
|---|---|---|---|---:|---|---|
| VGGT | Yes | No | W4A8 quant-only | **0.017626 / 0.031211 / 0.691679** | 接近 FP，四项中最稳定 | `eval/requested4/vggt_channelwise_quant_only/tre_diff/all_metrics.csv` |
| Pi3 | No | Yes | rho=.1/tau=.007/fit≤.8/r16 | 0.037403 / 0.014163 / 0.503651 | Comp 改善，但 Acc/NC 明显退化，不推荐 | `eval/requested4/pi3_noncw_qwt/tre_diff/all_metrics.csv` |
| DUSt3R | No | Yes | rho=.1/tau=.007/fit≤.8/r16 | 0.062016 / 0.054772 / 0.646184 | 明显差于 channel-wise safe QwT | `eval/requested4/dust3r_noncw_qwt/_all_samples.json` |
| MASt3R | No | Yes | rho=.1/tau=.007/fit≤.8/r16 | 0.042619 / 0.743426 / 0.499973 | Comp/NC 严重退化，不推荐 | `eval/requested4/mast3r_noncw_qwt/_all_samples.json` |

> **当前对照支持的经验结论：** channel-wise quant-only 在 VGGT 上接近 FP；DUSt3R 的门控 channel-wise QwT 有小幅收益，MASt3R 则没有可见收益。non-channel-wise + QwT 同时改变了量化粒度和补偿方式，缺少同口径 non-channel-wise quant-only 控制，因此只能判定该组合不稳定，不能把退化单独归因于 QwT。

### 4.3 OpenD4RT 与 VGGT-Omega 新增基座

OpenD4RT 使用公开 32CLIP 训练 checkpoint，量化范围为 40 个 encoder block 的 160 个投影与 8 个 decoder block 的 48 个投影，共 208 个 `qkv/proj/q_proj/k_proj/v_proj/fc1/fc2` Linear；query embedding、Fourier/patch embedding 与输出头排除。校准使用冻结的 `DTUTrain_8_mv-recon_seq-id-map-kf5.json`，实际遍历 8 个 scan、每个 10 帧，共 80 张图像；运行时打印的 `calibration size: 32` 只是旧元数据，当前代码并未用它截断校准迭代器。DTU8 Hessian 校准分别耗时 32.71 分钟（non-CW）和 35.64 分钟（CW）。

| OpenD4RT 配置 | Acc Mean / Median | Comp Mean / Median | NC Mean / Median | Checkpoint / 评估产物 |
|---|---:|---:|---:|---|
| FP | 0.039151 / 0.015880 | 0.049335 / 0.018543 | 0.647854 / 0.729544 | `eval/d4rt_opend4rt_fp/tre_diff/all_metrics.csv` |
| W4A8 non-CW | 0.099541 / 0.063859 | 0.182797 / 0.103827 | 0.589010 / 0.636253 | `checkpoints/d4rt_opend4rt_w4a8_noncw.pt`；`eval/d4rt_opend4rt_w4a8_noncw/` |
| **W4A8 channel-wise** | **0.039429 / 0.016178** | **0.048970 / 0.018367** | **0.645904 / 0.726407** | `checkpoints/d4rt_opend4rt_w4a8_cw.pt`；`eval/d4rt_opend4rt_w4a8_cw/` |

相对 FP，channel-wise W4A8 的 Acc 仅增加 0.71%，Comp 反而降低 0.74%，NC 降低 0.30%；non-channel-wise 的 Acc/Comp 分别扩大到 FP 的 2.54×/3.71×。因此 OpenD4RT 应默认采用 channel-wise W4A8。两份量化 checkpoint 均约 4.4 GiB。

VGGT-Omega 的代码、288 层 aggregator 量化适配与 E1 输出适配已完成，但本地和 Ceph 均没有 `vggt_omega_1b_512.pt`；官方 Hugging Face 仓库仍需授权访问。因此未以随机权重伪造 FP/W4A8 精度，状态保持 `checkpoint_access_required`。

### 4.4 外部 QwT 实现审计与搜索规模

旧 external QwT 没有与 VGGT 定义对齐：没有 `rho/tail_ratio`；直接复用 `tau=.007` 导致 120/120 分支入选；逐分支即时安装 hook 使后续 capture 被污染；低秩截断后未重算 bias，也缺少 `fit_error` 门控。现已修复为 VGGT 同定义 top-`rho` TRE、截断后重算 bias、全部 capture 后统一安装 hook，并以 `fit_error` 拒绝有害修正。

| 层级 | 模型 | 参数矩阵 | 计划组合数 |
|---|---|---|---:|
| rank16 初筛 | DUSt3R / MASt3R | rho=.001/.01/.1；tau=0/.003/.007/.02；fit≤.8 | 24 |
| rank16 密集门控 | DUSt3R / MASt3R | 各 rho 上 tau=0/.001/.003/.005/.007/.01/.02/.05；fit≤.2/.4/.6 | 144 |
| 高 rank | DUSt3R / MASt3R | rank=32/64/128；rho=.1；tau=.005/.007；fit≤.4/.6/.8 | 36 |

合计计划 204 个组合，其中新增/扩展队列为 180 个；最终形成 DUSt3R 100 条、MASt3R 102 条有效结果，共 202 条。失败或跳过配置不进入选优；已明确记录的早期 DUSt3R `fit≤.8` 诊断配置因 OpenCV SQPnP 失败，后续 `fit≤.6` 选优配置已成功完成 full18。

## 5. 校准集与关键实现消融

### 5.1 ERQ two-part 与 DTU8/DTU20

| 方法 | 位宽 | 校准集 | two-part | 7Scenes Acc / Comp / NC | ETH3D Acc / Comp / NC（辅助） |
|---|---|---|---|---:|---:|
| ERQ | W4A8 | DTU8/4096 | No | 0.077527 / 0.149162 / 0.650189 | 0.723984 / 2.355848 / 0.527638 |
| ERQ | W4A8 | DTU8/4096 | Yes | **0.033990 / 0.036652 / 0.683896** | **0.713734 / 1.591601 / 0.597714** |
| ERQ | W6A6 | DTU8/4096 | Yes | 0.036203 / 0.040491 / 0.681859 | 0.745798 / 1.730803 / 0.601874 |
| ERQ | W4A8 | DTU20/4096 | Yes | 0.088206 / 0.146000 / 0.646577 | 0.735495 / 2.665761 / 0.543135 |
| ERQ | W6A6 | DTU20/4096 | Yes | 0.078380 / 0.125551 / 0.652625 | 0.729353 / 2.384169 / 0.536392 |

ERQ 固定官方 commit `2f5b4cee...`，Aqer/Wqer、Rounding Refinement `top-k=1/T=100`、`coe=20000`（ridge=2000）、row batch=500、4096 tokens；two-part 仅用于 `qkv/fc1`。

### 5.2 GPTQ 的 DTU8/DTU20 对照

| 位宽 | 校准集 | 7Scenes Acc / Comp / NC | ETH3D Acc / Comp / NC（辅助） |
|---|---|---:|---:|
| W4A8 | DTU8/4096 | **0.030389 / 0.034527 / 0.686803** | **0.662946 / 1.665329 / 0.620042** |
| W4A8 | DTU20/4096 | 0.041455 / 0.046168 / 0.686904 | 0.895817 / 1.821033 / 0.605249 |
| W6A6 | DTU8/4096 | 0.039548 / 0.041233 / 0.679349 | 0.687522 / 1.692481 / 0.603864 |
| W6A6 | DTU20/4096 | 0.059999 / 0.054133 / 0.668440 | **0.667404 / 1.541667 / 0.583569** |

固定总 token 数时，DTU20 把每个场景可保留的 token 降到约 205，而 DTU8 每场景约 512；当前结果说明增加场景数但降低单场景覆盖密度并不稳定。

## 6. TAPTQ 补偿结果与消融

### 6.1 五个 checkpoint：`rho=0.1, tau=0.007, rank=16`

Run `701491027`，均为 module-level QwT。单元格格式为 Acc / Comp / NC，且每项为 Mean / Median。

| Setting | 7Scenes Acc / Comp / NC | ETH3D Acc / Comp / NC（辅助） |
|---|---:|---:|
| W4A8 ternary | 0.025325/0.010828 · 0.045779/0.016523 · 0.691264/0.792966 | 0.737394/0.522143 · 1.871054/0.794437 · 0.598173/0.643395 |
| W4A8 exhaustive | 0.025559/0.010913 · 0.044980/0.016859 · 0.691035/0.792994 | 0.703874/0.571900 · 1.714639/0.846824 · 0.603023/0.656432 |
| W4A8 channel-wise | 0.018475/0.007765 · 0.032060/0.015267 · 0.690933/0.794041 | 0.579910/0.482506 · 1.351510/0.707765 · 0.666858/0.756049 |
| W6A6 ternary | 0.020278/0.008620 · 0.034891/0.014573 · 0.690415/0.793453 | 0.631006/0.512185 · 1.446393/0.687413 · 0.633204/0.702324 |
| W8A8 ternary | 0.018175/0.007420 · 0.031062/0.015221 · 0.689673/0.792565 | 0.614368/0.524083 · 1.472902/0.851219 · 0.664082/0.744015 |

### 6.2 rho、tau、rank 消融合并表

均为 W4A8 ternary non-channel-wise，主评估集为 7Scenes dense。单元格格式为 Mean / Median。

| 消融 | 取值 | 固定参数 | Acc | Comp | NC |
|---|---:|---|---:|---:|---:|
| rho | 0.001 | tau=.007, r16 | 0.031290 / 0.013556 | 0.054002 / 0.020365 | 0.686838 / 0.785589 |
| rho | 0.005 | tau=.007, r16 | 0.032472 / 0.013279 | 0.058019 / 0.020586 | 0.685139 / 0.784574 |
| rho | 0.01 | tau=.007, r16 | 0.028793 / 0.011803 | 0.051019 / 0.018225 | 0.688277 / 0.788356 |
| rho | 0.02 | tau=.007, r16 | 0.028101 / 0.012400 | 0.049177 / 0.018192 | 0.688762 / 0.789268 |
| rho | 0.05 | tau=.007, r16 | 0.026571 / 0.011379 | 0.045859 / 0.015848 | 0.689650 / 0.790928 |
| rho | **0.1** | tau=.007, r16 | **0.025325 / 0.010828** | 0.045779 / 0.016523 | **0.691264 / 0.792966** |
| rho | 0.5 | tau=.007, r16 | 0.026750 / 0.011279 | **0.043143 / 0.015477** | 0.687830 / 0.788687 |
| rho | 1.0 | tau=.007, r16 | 0.028978 / 0.012071 | 0.044269 / 0.015732 | 0.687913 / 0.788951 |
| tau | 0（无效，实际回退为 .007） | rho=.1, r16 | 0.025325 / 0.010828 | 0.045779 / 0.016523 | 0.691264 / 0.792966 |
| tau | 0.001 | rho=.1, r16 | 0.027780 / 0.011349 | 0.045384 / 0.015336 | 0.687713 / 0.789177 |
| tau | 0.003 | rho=.1, r16 | 0.029114 / 0.012179 | 0.045643 / 0.015312 | 0.686996 / 0.787027 |
| tau | **0.005** | rho=.1, r16 | **0.024321 / 0.010059** | **0.041814 / 0.014438** | 0.689287 / 0.791060 |
| tau | 0.007 | rho=.1, r16 | 0.025325 / 0.010828 | 0.045779 / 0.016523 | **0.691264 / 0.792966** |
| tau | 0.01 | rho=.1, r16 | 0.027683 / 0.011701 | 0.053068 / 0.019010 | 0.687786 / 0.787751 |
| tau | 0.02 | rho=.1, r16 | 0.029778 / 0.012919 | 0.052901 / 0.019817 | 0.688148 / 0.787293 |
| tau | 0.05 | rho=.1, r16 | 0.029030 / 0.012653 | 0.051640 / 0.019381 | 0.688465 / 0.788458 |
| rank | 8 | rho=.1, tau=.007 | 0.027066 / 0.011533 | 0.045646 / 0.016145 | 0.690162 / 0.791503 |
| rank | 16 | rho=.1, tau=.007 | 0.025325 / 0.010828 | 0.045779 / 0.016523 | 0.691264 / 0.792966 |
| rank | 32 | rho=.1, tau=.007 | 0.020863 / 0.008915 | 0.039108 / 0.015731 | 0.692756 / 0.795568 |
| rank | 64 | rho=.1, tau=.007 | 0.019590 / 0.008517 | 0.037044 / 0.014933 | 0.692338 / 0.795237 |
| rank | 128 | rho=.1, tau=.007 | 0.018886 / 0.008108 | 0.035867 / 0.014019 | 0.694326 / 0.798332 |
| rank | **256** | rho=.1, tau=.007 | **0.018472 / 0.008117** | **0.033707 / 0.014021** | **0.694432 / 0.798279** |

ETH3D rank 辅助结果：r16=`0.737394/1.871054/0.598173`，r32=`0.696210/1.681153/0.604388`（Mean）。

### 6.3 Channel-wise rho、tau、rank 消融合并表

均为 VGGT W4A8 ternary **channel-wise**，DTU8 校准，主评估集为 7Scenes dense。单元格格式为 Mean / Median；数据来自 `vggt_channelwise_ablation` 的逐配置 `all_metrics.csv`，不是 §6.2 的 non-channel-wise sweep。

| 消融 | 取值 | 固定参数 | Acc | Comp | NC |
|---|---:|---|---:|---:|---:|
| rho | 0.001 | tau=.005, r128 | **0.018026 / 0.007687** | **0.031896 / 0.015213** | **0.691003 / 0.793865** |
| rho | 0.005 | tau=.005, r128 | 0.018394 / 0.007722 | 0.032502 / 0.015865 | 0.690571 / 0.793775 |
| rho | **0.01** | tau=.005, r128 | **0.018026 / 0.007687** | **0.031896 / 0.015213** | **0.691003 / 0.793865** |
| rho | 0.02 | tau=.005, r128 | 0.018026 / 0.007687 | 0.031896 / 0.015213 | 0.691003 / 0.793865 |
| rho | 0.05 | tau=.005, r128 | 0.018348 / 0.007925 | 0.032280 / 0.015439 | 0.689540 / 0.792480 |
| rho | 0.1 | tau=.005, r128 | 0.018364 / 0.007913 | 0.032123 / 0.015350 | 0.690079 / 0.792663 |
| rho | 0.5 | tau=.005, r128 | 0.018394 / 0.007722 | 0.032502 / 0.015865 | 0.690571 / 0.793775 |
| rho | 1.0 | tau=.005, r128 | 0.018401 / 0.007965 | 0.032824 / 0.016106 | 0.690143 / 0.793065 |
| tau | 0（无效，实际回退为 .007） | rho=.01, r128 | 0.019100 / 0.008028 | 0.032920 / 0.015457 | 0.688014 / 0.790337 |
| tau | 0.001 | rho=.01, r128 | 0.018053 / 0.007652 | 0.032343 / 0.015619 | 0.690476 / 0.793487 |
| tau | 0.003 | rho=.01, r128 | 0.018288 / 0.007696 | 0.032210 / 0.015435 | 0.689748 / 0.792014 |
| tau | **0.005** | rho=.01, r128 | **0.018026 / 0.007687** | 0.031896 / 0.015213 | 0.691003 / 0.793865 |
| tau | 0.007 | rho=.01, r128 | 0.019100 / 0.008028 | 0.032920 / 0.015457 | 0.688014 / 0.790337 |
| tau | 0.01 | rho=.01, r128 | 0.018031 / 0.007699 | 0.032167 / 0.015132 | 0.690221 / 0.793132 |
| tau | **0.02** | rho=.01, r128 | **0.017528 / 0.007738** | **0.031104 / 0.014808** | **0.691759 / 0.795235** |
| tau | 0.05 | rho=.01, r128 | 0.017630 / 0.007700 | 0.031155 / 0.014870 | 0.691688 / 0.795073 |
| rank | 8 | rho=.01, tau=.005 | 0.018333 / 0.007611 | 0.031925 / 0.015087 | 0.690835 / 0.793852 |
| rank | 16 | rho=.01, tau=.005 | 0.018276 / 0.007680 | 0.032410 / 0.015222 | 0.690000 / 0.793117 |
| rank | **32** | rho=.01, tau=.005 | 0.018067 / 0.007669 | **0.031918 / 0.015356** | **0.691678 / 0.795242** |
| rank | 64 | rho=.01, tau=.005 | 0.018442 / 0.007844 | 0.031989 / 0.015456 | 0.690155 / 0.792975 |
| rank | **128** | rho=.01, tau=.005 | **0.018026 / 0.007687** | 0.031896 / 0.015213 | 0.691003 / 0.793865 |
| rank | 256 | rho=.01, tau=.005 | 0.018566 / 0.007794 | 0.032607 / 0.015484 | 0.689016 / 0.791702 |

已完成的额外交叉点：`rho=.1, tau=.007, r128` 为 `Acc 0.018261/0.007564`、`Comp 0.032282/0.015245`、`NC 0.690139/0.792842`；§6.1 的 `rho=.1, tau=.007, r16` 为 `0.018475/0.007765`、`0.032060/0.015267`、`0.690933/0.794041`。由于固定参数不同，这两个点不混入上面的单变量最优比较。

12 项缺失配置现已全部完成并回填。固定 `tau=.005, r128` 时，`rho=.001/.01/.02` 的结果相同；固定 `rho=.01, tau=.005` 时，rank32 的 Comp/NC 最好，rank128 的 Acc 最好。固定 `rho=.01, rank128` 时，当前 tau=.02 在 Acc/Comp/NC 三项上综合最好。

### 6.4 跨 setting 组合筛选

Run `701546763`，Mean 指标。

| Setting | CW | rho/tau/rank | 7Scenes Acc / Comp / NC | ETH3D Acc / Comp / NC（辅助） |
|---|---|---|---:|---:|
| W4A8 ternary | No | .1/.005/256 | 0.018990 / 0.034746 / 0.693951 | 0.658013 / 1.587040 / 0.618117 |
| W4A8 ternary | No | .01/.005/256 | 0.020057 / 0.031452 / 0.692546 | 0.720945 / 1.784419 / 0.631503 |
| W4A8 ternary | No | .05/.005/128 | 0.019426 / 0.036389 / **0.694892** | 0.838353 / 1.944112 / 0.619188 |
| W4A8 exhaustive | No | .01/.005/256 | 0.020061 / 0.036617 / 0.694042 | 0.772398 / 1.681983 / 0.632591 |
| W4A8 channel-wise | Yes | .01/.005/256 | 0.018566 / 0.032607 / 0.689016 | **0.560725 / 1.429107 / 0.682015** |
| W6A6 ternary | No | .01/.005/256 | **0.018232** / 0.032846 / 0.691639 | 0.639463 / 1.602481 / 0.652772 |
| W8A8 ternary | No | .01/.005/256 | 0.018283 / **0.031144** / 0.689938 | 0.609459 / 1.485683 / 0.665736 |
| W4A8 ternary | No | .1/.007/128 | 0.018886 / 0.035867 / 0.694326 | 0.632187 / 1.603074 / 0.626990 |

综合性能与参数量，默认仍采用 `rho=0.1, tau=0.007, rank=128`；若只追求 7Scenes 上界，可用 rank=256。

### 6.5 W4A8 补偿比例与模块选择消融（新协议）

本节为论文重构新增的冻结结果：VGGT、W4A8、DTU_train_8 校准、7Scenes dense full18，固定 `rho=.1`、`rank=128`，在 144 个候选 attention/MLP 模块上比较五档补偿预算。`62/144` 是默认 `tau=.007` 阈值配置实际选出的模块数；其他比例使用相同 quant-only checkpoint 做固定预算 top-K 选择。Random 固定 seed=42。单元格为 Mean Acc / Comp / NC。

| 补偿预算 | 选择指标 | Acc ↓ | Comp ↓ | NC ↑ |
|---:|---|---:|---:|---:|
| 24/144 | TRE | 0.025874 | 0.044597 | 0.688276 |
| 24/144 | MSE | 0.030371 | 0.045536 | 0.687632 |
| 24/144 | Hessian | 0.030503 | 0.046976 | 0.685933 |
| 24/144 | Random | 0.030941 | 0.046923 | 0.688319 |
| 48/144 | TRE | 0.021498 | 0.038211 | 0.692796 |
| 48/144 | MSE | 0.022470 | 0.038236 | 0.691454 |
| 48/144 | Hessian | 0.024763 | 0.040115 | 0.689811 |
| 48/144 | Random | 0.022949 | 0.039355 | 0.689175 |
| **62/144** | **TRE（默认）** | **0.019822** | 0.036430 | **0.695265** |
| 62/144 | MSE | 0.022227 | 0.037820 | 0.690148 |
| 62/144 | Hessian | 0.020257 | **0.035947** | 0.692438 |
| 62/144 | Random | 0.022525 | 0.040430 | 0.691274 |
| 72/144 | TRE | 0.019895 | 0.035862 | 0.694169 |
| 72/144 | MSE | 0.020140 | 0.037739 | 0.692988 |
| 72/144 | Hessian | **0.019011** | 0.036116 | **0.694760** |
| 72/144 | Random | 0.021396 | 0.038489 | 0.691965 |
| 96/144 | TRE | 0.019795 | **0.034718** | **0.694900** |
| 96/144 | MSE | **0.019551** | 0.036839 | 0.692821 |
| 96/144 | Hessian | 0.020036 | 0.036529 | 0.692752 |
| 96/144 | Random | 0.024370 | 0.042894 | 0.689881 |

结论边界：TRE 在五档预算中始终优于或接近随机选择，并在默认 `62/144` 点取得最佳 NC；但不同预算下 Acc/Comp 的单项最优会在 TRE、MSE 和 Hessian 之间变化。因此论文应表述为“tail-aware selection provides a strong accuracy--geometry trade-off under a matched compensation budget”，而不是宣称 TRE 在每个指标和每个比例上都绝对最优。全部逐配置 CSV 和论文协议台账见 `doc/TMM_PAPER_EXPERIMENT_PROTOCOL_MANIFEST.json`。

## 7. 附录：Quant-only、PDF 与旧协议结果

### 7.1 统一评估器下的 quant-only 历史结果（Mean）

| 配置 | 7Scenes Acc / Comp / NC |
|---|---:|
| TAPTQ W4A8 ternary | 0.034261 / 0.054669 / 0.687745 |
| TAPTQ W4A8 exhaustive | 0.031122 / 0.049532 / 0.690451 |
| TAPTQ W4A8 channel-wise | **0.017626 / 0.031211 / 0.691679** |
| TAPTQ W6A6 ternary | 0.020507 / 0.035523 / 0.691768 |
| TAPTQ W8A8 ternary | 0.018129 / 0.031015 / 0.689912 |

### 7.2 PDF 中的 VGGT `Ours` 参考

| 配置 | PDF 7Scenes Acc / Comp / NC | PDF ETH3D Acc / Comp / NC | 当前可追溯对应结果 |
|---|---:|---:|---:|
| W4A8 Ours | 0.028/0.044/0.687 | 0.492/0.677/0.761 | r16：0.025325/0.045779/0.691264；ETH3D 0.737394/1.871054/0.598173 |
| W6A6 Ours | 0.021/0.037/0.690 | 0.334/0.406/0.813 | r16：0.020278/0.034891/0.690415；ETH3D 0.631006/1.446393/0.633204 |
| W4A8 channel-wise | 0.018/0.032/0.688 | — | 0.018566/0.032607/0.689016；ETH3D 0.560725/1.429107/0.682015 |

ETH3D 当前 FP 与 PDF 也存在明显差距，因此不能把量化行差异单独归因于算法；还需对齐 PDF 的预处理、帧采样、checkpoint 与 evaluator。

### 7.3 不进入主表的旧/异协议结果（Mean）

| 方法 | 7Scenes Acc / Comp / NC | ETH3D Acc / Comp / NC | 原因 |
|---|---:|---:|---|
| QuantVGGT W4A8 | 0.020489 / 0.033833 / 0.676142 | — | upstream evaluator |
| QuantVGGT W8A8 | 0.020497 / 0.033838 / 0.676140 | — | upstream evaluator |
| RTN-style W4A8 | 0.020496 / 0.033840 / 0.676275 | — | upstream evaluator |
| 早期 RTN adapter W4A8 | 0.034997 / 0.057019 / 0.681497 | 0.674124 / 1.803128 / 0.603189 | 已被 aligned 实现替代 |
| 早期 RepQ adapter W4A8 | 0.114202 / 0.269601 / 0.613569 | 0.854631 / 3.267392 / 0.527036 | 已被 aligned 实现替代 |
| 早期 ERQ adapter W4A8 | 0.066906 / 0.607316 / 0.577133 | 0.717029 / 4.477670 / 0.527616 | 遗漏 two-part |
| 早期 SVDQuant-style W4A8 r128 | 0.051505 / 0.057741 / 0.685049 | 0.704531 / 1.653232 / 0.583574 | 已被 DeepCompressor aligned 替代 |
| 早期 SVDQuant-style W4A4 r32 | 0.100695 / 0.279415 / 0.625871 | 0.774879 / 3.532489 / 0.521273 | 已被 aligned 实现替代 |

## 8. H800 fused W8A8 与 W4A8 hybrid 部署

### 8.1 fused W8A8 实现

W8A8 部署使用 `w8a8_ternary.pt` 的 288 个 quant-only Linear。Triton 3.1.0 后端将 activation A8 量化作为 producer kernel；对 72 个 FC2，exact GELU 与 A8 量化合并；INT8 Tensor Core GEMM 的 epilogue 融合 INT32 反量化、per-output weight scale、bias 和 BF16 cast。推理图不包含 QwT/LoRA 额外分支。

当前保留两种 Post-GELU 模式：

- `single`：单个对称 INT8 GEMM，舍弃独立负尾 scale；延迟最低。
- `dual`：同一 Triton kernel 内计算正/负两个 INT8 accumulator，保留原 TAPTQ 双 scale 语义；精度更高但计算量更大。

正式协议：H800、VGGT-1B aggregator、batch=1、`518×518`、CUDA Graph、5 warmup、20 timed、CUDA events。BF16 均单独加载并使用相同 graph 协议复测。

| Views | BF16 latency | fused W8A8 single | vs BF16 | W8A8 throughput |
|---:|---:|---:|---:|---:|
| 1 | 21.483 ms | 21.506 ms | 0.999× | 46.499 views/s |
| 2 | 35.453 ms | **34.259 ms** | **1.035×** | 58.380 views/s |
| 4 | 67.667 ms | **66.850 ms** | **1.012×** | 59.835 views/s |
| 8 | 149.506 ms | **147.225 ms** | **1.015×** | 54.339 views/s |

8-view single 模式运行时模型为 `2,305,939,388` bytes，约为 FP32 的 **45.88%**；CUDA Graph replay peak allocated 为 `3,399,778,304` bytes（约 3.166 GiB）。相对 BF16 的收益目前较小但已经为正，说明 fused INT8 路径已经跨过了 PyTorch primitive 版本的开销拐点。

数值与几何复核：

| 模式 | 末层 cosine / relative L2 | 7Scenes Acc Mean/Med | Comp Mean/Med | NC Mean/Med |
|---|---:|---:|---:|---:|
| single | 0.991220 / 0.135891 | 0.017546 / 0.008017 | 0.032976 / 0.015591 | 0.689514 / 0.792069 |
| dual | **0.999188 / 0.044705** | 0.018333 / 0.007366 | **0.030944 / 0.014997** | 0.687220 / 0.789228 |

`single` 的 full18 几何结果仍接近 W8A8 fake-quant 与 FP，但逐 stage 误差明显高于 `dual`。默认低延迟部署采用 `single`；精度敏感场景采用 `dual`，其 8-view latency 为 `157.767 ms`，相对 BF16 为 `0.948×`。

产物：`/mnt/cephfs/pansicheng/tmm-results/deploy/vggt_w8a8_triton/`；代码：`Pi3-evaluation/deployment/w8a8_triton_deploy.py`、`validate_w8a8_triton.py`、`evaluate_w8a8_triton.py`。

### 8.2 W4/A8-calibrated hybrid FP8/BF16 回顾与 B200 对齐

> 准确命名：**H800 W4/A8-calibrated hybrid FP8/BF16 prototype**。它使用 packed INT4 artifact 和 TAPTQ W4/A8 校准网格，但不是原生 fused INT4×INT8 kernel。

#### 8.2.1 论文数据口径

论文和 rebuttal 实际使用的是 **NVIDIA B200**，不是 B300。口径为 VGGT-1B **aggregator forward**、batch=1、8 views、每帧 `518×518`、5 次 warmup、25 次 CUDA-event timed runs。论文表中只有 FP32、W8A8 和 W8-Abf16，没有 W4A8 部署行：

| B200 配置 | Latency | Speedup | Weight | Peak |
|---|---:|---:|---:|---:|
| FP32 | 401 ms | 1.00× | 5.0 GB | 9.9 GB |
| W8A8 static + compile | 74 ms | 5.42× | 1.7 GB | 4.1 GB |
| W8-Abf16 + compile | 81 ms | 4.95× | 1.4 GB | 3.8 GB |

B200 表使用编译后的 W8 后端，不能与 H800 上的 W4A8 prototype 直接作硬件等价比较。论文输入描述中误排的 `85,182-pixel frames` 已修正为 `8 × 518 × 518` frames。

#### 8.2.2 H800 W4A8 优化实现与公平对照

H800 路径采用 packed signed INT4 artifact。由于 PyTorch 2.5 没有原生 fused INT4×INT8 API，当前最佳后端保持 TAPTQ 的 W4/A8 校准网格，但用 Hopper E4M3 FP8 承载量化整数值执行 216 个普通 Linear；72 个 dual-scale Post-GELU FC2 使用单次 BF16 Tensor Core GEMM。其余 aggregator 算子使用 BF16，任务头保留 FP32；固定 shape 使用 CUDA Graph。INT8 `_int_mm` 后端仍保留为精确整数参考。

正式协议：H800、VGGT-1B aggregator、batch=1、`518×518`、5 warmup、20 timed、CUDA events。除论文对齐的 FP32 基线外，额外加入相同 CUDA Graph 的 BF16 非量化基线，用于分离“量化后端收益”和“单纯降 dtype 收益”。显存为 graph 准备完成后的 replay 峰值，不含首次 capture/compile 瞬时内存。

| Views | FP32 latency | BF16 latency | W4A8-FP8 latency | vs FP32 | vs BF16 | W4A8 throughput | FP32 / BF16 / W4A8 peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 98.531 ms | 21.460 ms | 44.239 ms | **2.227×** | 0.485× | 22.604 views/s | 4.998 / 2.565 / 2.681 GiB |
| 2 | 194.968 ms | 35.403 ms | 79.896 ms | **2.440×** | 0.443× | 25.033 views/s | 5.253 / 2.695 / 2.808 GiB |
| 4 | 412.582 ms | 68.166 ms | 168.058 ms | **2.455×** | 0.406× | 23.801 views/s | 5.762 / 2.956 / 3.063 GiB |
| 8 | 925.427 ms | 148.751 ms | 338.172 ms | **2.737×** | 0.440× | 23.657 views/s | 6.780 / 3.477 / 3.572 GiB |

优化演进（8 views）：初版 hybrid 为 `942.18 ms` / `1.051×`，稳定 INT8+BF16+CUDA Graph 为 `487.32 ms` / `1.898×`，最终 FP8 Tensor Core 后端为 `338.17 ms` / `2.737×`。最终版本相对初版 latency 降低 **64.1%**，相对稳定 INT8 版再降低 **30.6%**。

但公平的 BF16 对照只有 `148.75 ms`：当前 W4A8-FP8 仍比 BF16 慢约 `2.27×`（即 0.440×）。因此 **2.737× 只能表述为相对 FP32 的端到端收益，不能归因于纯 W4A8 kernel**。剩余瓶颈是每层独立的 activation round/clamp/cast、输出反量化与 per-channel scale，尚未融合到 GEMM；要接近论文 B200 的 `5.42×`，需要 CUTLASS/Triton/TensorRT 原生 fused W4A8 kernel，而不是继续叠加 PyTorch eager primitive。

存储方面，packed artifact 为 `458,648,854` bytes（约 437.4 MiB）；完整运行时模型约为 FP32 的 **51.88%**。FP8 路径逐 stage 验证末层 cosine 为 `0.998389`、relative L2 为 `0.06020`，通过默认质量阈值。

最终 FP8 部署已完成 7Scenes E1 full18 几何复核：

| 部署配置 | Acc Mean / Median | Comp Mean / Median | NC Mean / Median |
|---|---:|---:|---:|
| stable INT8/BF16 + FP32 heads | 0.017404 / 0.007699 | 0.031026 / 0.014696 | 0.690020 / 0.792752 |
| **216 FP8 GEMMs + 72 BF16 FC2 + BF16 non-Linear ops + FP32 heads** | **0.017337 / 0.007660** | **0.030804 / 0.014446** | **0.689845 / 0.792377** |

两条实际部署结果均与 channel-wise fake-quant quant-only 的 `0.017626/0.031211/0.691679`（Mean）接近，未出现几何精度崩溃。FP8 相对稳定 INT8 的 Acc/Comp 略优、NC 略低，差异较小。

原始性能产物：`/mnt/cephfs/pansicheng/tmm-results/deploy/vggt_w4a8_int8/fp8_graph_b1v{1,2,4,8}.json`；几何产物：`/mnt/cephfs/pansicheng/tmm-results/eval/vggt_w4a8_fp8_deploy/tre_diff/all_metrics.csv`；代码：`Pi3-evaluation/deployment/w4a8_int8_deploy.py`。

## 9. 结果与产物索引

### 9.1 台账与 Checkpoint

- 结构化测试台账：`doc/tmm_test_manifest.json`
- 模型台账：`doc/tmm_model_manifest.json`
- 实时 GPU 台账：`doc/H800_LEASE_GPU_STATUS.md`
- Checkpoint：`/mnt/cephfs/pansicheng/tmm-results/checkpoints/`
- 日志：`/mnt/cephfs/pansicheng/tmm-results/logs/`

### 9.2 VGGT 主表与消融

- 官方对齐 baseline：`/mnt/cephfs/pansicheng/tmm-results/eval/vggt_official_aligned/`
- ERQ two-part：`/mnt/cephfs/pansicheng/tmm-results/eval/vggt_erq_two_part/`
- GPTQ DTU8/DTU20：`/mnt/cephfs/pansicheng/tmm-results/eval/vggt_gptq_dtu8/`、`/mnt/cephfs/pansicheng/tmm-results/eval/vggt_gptq_dtu20/`
- SmoothQuant：`/mnt/cephfs/pansicheng/tmm-results/eval/vggt_smoothquant/`
- TAPTQ 补偿：`/mnt/cephfs/pansicheng/tmm-results/eval/vggt_taptq_compensated_rho01_tau0007_r16/`
- non-channel-wise 消融：`/mnt/cephfs/pansicheng/tmm-results/eval/vggt_ablation/`
- channel-wise 完整消融：`/mnt/cephfs/pansicheng/tmm-results/eval/vggt_channelwise_ablation/`
- 跨 setting 筛选：`/mnt/cephfs/pansicheng/tmm-results/eval/vggt_combo_selection/`

### 9.3 跨模型与指定实验

- 四项指定实验根目录：`/mnt/cephfs/pansicheng/tmm-results/eval/requested4/`
- DUSt3R/MASt3R external QwT full18：`/mnt/cephfs/pansicheng/tmm-results/eval/full18/`
- OpenD4RT FP/CW/non-CW：`/mnt/cephfs/pansicheng/tmm-results/eval/d4rt_opend4rt_{fp,w4a8_cw,w4a8_noncw}/`
- OpenD4RT W4A8 checkpoint：`/mnt/cephfs/pansicheng/tmm-results/checkpoints/d4rt_opend4rt_w4a8_{cw,noncw}.pt`
- OpenD4RT 日志：`/mnt/cephfs/pansicheng/tmm-results/logs/d4rt_opend4rt_*.log`
- external QwT 快筛脚本：`run-external-qwt-rho-tau.sh`、`run-external-qwt-filter-sweep.sh`、`run-external-qwt-rank-sweep.sh`；日志统一从 `/mnt/cephfs/pansicheng/tmm-results/logs/` 按任务名检索

### 9.4 H800 部署

- packed INT4 artifact：`/mnt/cephfs/pansicheng/tmm-results/deploy/vggt_w4a8_int8/w4a8_packed_v2.pt`
- FP8 性能矩阵：`/mnt/cephfs/pansicheng/tmm-results/deploy/vggt_w4a8_int8/fp8_graph_b1v{1,2,4,8}.json`
- FP8 数值验证：`/mnt/cephfs/pansicheng/tmm-results/deploy/vggt_w4a8_int8/fp8_validation.json`
- stable INT8/BF16 几何结果：`/mnt/cephfs/pansicheng/tmm-results/eval/vggt_w4a8_deploy_optimized/tre_diff/all_metrics.csv`
- FP8/BF16 hybrid 几何结果：`/mnt/cephfs/pansicheng/tmm-results/eval/vggt_w4a8_fp8_deploy/tre_diff/all_metrics.csv`
- fused W8A8 artifact：`/mnt/cephfs/pansicheng/tmm-results/deploy/vggt_w8a8_triton/w8a8_int8_v2.pt`（与已评估 v1 的 288 层权重和 scale 逐项一致）
- fused W8A8 性能矩阵：`/mnt/cephfs/pansicheng/tmm-results/deploy/vggt_w8a8_triton/*graph_b1v*.json`
- fused W8A8 single/dual 几何结果：`/mnt/cephfs/pansicheng/tmm-results/eval/vggt_w8a8_triton_{single,dual}/tre_diff/all_metrics.csv`
- 部署代码：`Pi3-evaluation/deployment/`
