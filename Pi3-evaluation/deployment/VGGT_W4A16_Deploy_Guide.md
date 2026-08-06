# VGGT W4A16 量化部署指南 (基于 torchao)

> 本文档用于指引 AI 完成 VGGT 模型的 **W4A16** 部署:
> - **INT4 权重** (PTQ4ViT 校准) + **FP16 激活** (不做激活量化)
> - **QwT 补偿模块** (FP16 SVD 低秩, rank=64, 仅部分模块启用)
> - 部署框架: **torchao** `Int4WeightOnlyConfig`
>
> W4A16 是三种方案 (W4A16/W4A8/W4A4) 中实现最简单、速度最快、精度最好的方案。

---

## 1. 整体架构

### 1.1 VGGT 模型结构

```
VGGT (约 1B 参数)
├── aggregator                          ← 主干网络, 需要量化
│   ├── patch_embed (DINOv2 ViT-L/14)
│   │   ├── patch_embed (Conv2d 3→1024, kernel=14, stride=14)  ← 不量化
│   │   ├── cls_token (1, 1, 1024)                              ← 不量化
│   │   ├── pos_embed                                           ← 不量化
│   │   └── blocks[0..23] (24 个 ViT Block)
│   │       ├── norm1 (LayerNorm)                               ← 不量化
│   │       ├── attn.qkv (Linear 1024→3072)                    ← W4 量化
│   │       ├── attn.proj (Linear 1024→1024)                   ← W4 量化
│   │       ├── ls1 (LayerScale)                                ← 不量化
│   │       ├── norm2 (LayerNorm)                               ← 不量化
│   │       ├── mlp.fc1 (Linear 1024→4096)                     ← W4 量化
│   │       ├── mlp.fc2 (Linear 4096→1024)                     ← W4 量化
│   │       └── ls2 (LayerScale)                                ← 不量化
│   │
│   ├── frame_blocks[0..23] (24 个局部注意力 Block, 结构同上)   ← W4 量化
│   ├── global_blocks[0..23] (24 个全局注意力 Block, 结构同上)  ← W4 量化
│   └── aa_block_num = 24
│
├── head_point (DPT Head)               ← 不量化, FP16
├── head_depth (DPT Head)               ← 不量化, FP16
├── head_conf  (DPT Head)               ← 不量化, FP16
└── head_camera                         ← 不量化, FP16
```

每个 block 有 4 个量化 Linear 层, 共 `(24 + 24 + 24) × 4 = 288` 个量化层。

### 1.2 QwT 补偿机制

PTQ 校准时, 对每个 block 的 attn 和 mlp 子模块分别计算 Tail Relative Error (TRE):
- `TRE >= 0.007` → 量化误差较大 → **启用 QwT** 补偿
- `TRE < 0.007` → 量化已足够精确 → **关闭 QwT**, 零额外计算

每个 QwT 补偿模块 (启用时) 包含:
- `A`: FP16 tensor, shape `[C_in, 64]` (C_in 通常 = 1024)
- `B`: FP16 tensor, shape `[64, C_out]` (C_out 通常 = 1024)
- `lora_bias`: FP32 tensor, shape `[C_out]`

补偿计算: `compensation = (x.half() @ A @ B).float() + lora_bias`

**每个 block 最多 2 个 QwT 模块** (attn_QwT + mlp_QwT), 共最多 `(24+24+24) × 2 = 144` 个, 但只有部分启用。

---

## 2. 输入文件

运行 `save_int4_weights.py --mode w4a8` (或 w4a4, 权重部分相同) 生成:

```
output_dir/
├── int4_weights.pt          # 288 个量化层的 INT4 权重 + scale
├── qwt_compensation.pt      # QwT 补偿模块 (含 QwT_enabled 标志)
├── non_quant_params.pt      # 非量化层的 FP16 参数
└── deploy_config.json       # 部署配置元信息
```

### 2.1 int4_weights.pt 格式

```python
# torch.load() 得到 list[dict], 长度 = 288 (所有量化 Linear)
# 按模型 named_modules() 遍历顺序排列

entry = {
    'name': 'aggregator.patch_embed.blocks.0.attn.qkv',
    'module_class': 'PTQSLBatchingQuantLinear',

    # INT4 整数权重, 存为 int8 (值域 [-8, 7])
    'int_weight': tensor(dtype=int8, shape=[out_features, in_features]),
    'weight_shape': [out_features, in_features],

    # 权重量化 scale (每层一个或每通道一个)
    # 反量化公式: w_float = int_weight.float() * w_interval
    'w_interval': tensor(dtype=float32),
    'w_qmax': 8,   # INT4 对称量化: 值域 [-8, 7]

    # 激活量化 scale (W4A16 不使用, 但文件中仍有此字段)
    'a_interval': tensor(dtype=float32),
    'a_qmax': 128,  # 或 8, 取决于保存时的 mode

    # 偏置 (大部分 Linear 没有 bias, 此字段为 None)
    'bias': None,  # 或 tensor(dtype=float16)
}
```

**w_interval 的 shape 变体**:
- 普通 Linear (proj, fc1, fc2): `shape=[1, 1, 1, 1]` (tensor-wise 一个 scale)
- qkv Linear (Q/K/V 拼接): `shape=[3, 1, 1, 1]` (每个 head 组一个 scale)
- channel-wise 量化时: `shape=[out_channels, 1, 1, 1]`

### 2.2 qwt_compensation.pt 格式

```python
# torch.load() 得到 list[dict]
# 每个 dict 代表一个 QwT 补偿模块

# ---- QwT_enabled = True 的情况 ----
entry_enabled = {
    'name': 'aggregator.patch_embed.blocks.5.attn_QwT',
    'module_class': 'AttnQwT',    # 或 'MlpQwT'
    'QwT_enabled': True,           # ← 需要执行补偿
    'A': tensor(dtype=float16, shape=[1024, 64]),   # SVD 左因子
    'B': tensor(dtype=float16, shape=[64, 1024]),    # SVD 右因子
    'lora_bias': tensor(dtype=float32, shape=[1024]),
    'rank': 64,
}

# ---- QwT_enabled = False 的情况 ----
entry_disabled = {
    'name': 'aggregator.patch_embed.blocks.2.attn_QwT',
    'module_class': 'AttnQwT',
    'QwT_enabled': False,          # ← 不需要补偿, 跳过
    'lora_bias': tensor(dtype=float32, shape=[1024]),  # 仍有 bias 但不使用
}
```

### 2.3 non_quant_params.pt 格式

```python
# torch.load() 得到 OrderedDict[str, tensor]
# key = 参数完整路径, value = FP16 tensor
{
    'aggregator.patch_embed.cls_token': tensor(float16, [1, 1, 1024]),
    'aggregator.patch_embed.pos_embed': tensor(float16, [1, 1370, 1024]),
    'aggregator.patch_embed.patch_embed.proj.weight': tensor(float16, [1024, 3, 14, 14]),
    'aggregator.patch_embed.patch_embed.proj.bias': tensor(float16, [1024]),
    'aggregator.patch_embed.blocks.0.norm1.weight': tensor(float16, [1024]),
    'aggregator.patch_embed.blocks.0.norm1.bias': tensor(float16, [1024]),
    'aggregator.patch_embed.blocks.0.ls1.gamma': tensor(float16, [1024]),
    # ... 所有 LayerNorm, LayerScale, 位置编码, cls_token ...
    'head_point.scratch.layer1_rn.weight': tensor(float16, ...),
    # ... 所有 DPT head 参数 ...
}
```

### 2.4 deploy_config.json

```json
{
    "mode": "w4a8",
    "w_bit": 4,
    "compensation_type": "QwT_SVD_rank64",
    "compensation_precision": "FP16",
    "num_quant_layers": 288,
    "num_qwt_modules": 144,
    "num_qwt_enabled": 95,
    "total_int4_params_M": 310.5,
    "total_qwt_params_M": 12.8,
    "total_non_quant_params_M": 45.2,
    "estimated_model_size_MB": 280.5
}
```

---

## 3. 实现步骤

### Step 1: 安装依赖

```bash
pip install torchao>=0.5.0
# VGGT 模型依赖
pip install torch>=2.1.0
```

### Step 2: 加载原始 VGGT 模型

```python
import torch
import torch.nn as nn
import json
from collections import OrderedDict

# 加载 VGGT 模型结构 (FP16 权重, 后面会被替换)
from vggt.models.vggt import VGGT
model = VGGT.from_pretrained("path/to/vggt-1b").eval().half().cuda()
```

### Step 3: 反量化 INT4 权重并注入模型

```python
int4_data = torch.load("output_dir/int4_weights.pt", map_location='cpu')

# 构建 name → entry 映射
int4_map = {entry['name']: entry for entry in int4_data}

for name, module in model.named_modules():
    if name not in int4_map:
        continue
    entry = int4_map[name]

    # 反量化: w_float = int_weight * w_interval
    int_w = entry['int_weight'].float()        # [out, in], int8 存储的 INT4 值
    w_interval = entry['w_interval'].float()   # scale

    # 广播 w_interval 到 int_w 的维度
    while w_interval.dim() > int_w.dim():
        w_interval = w_interval.squeeze(-1)
    while w_interval.dim() < int_w.dim():
        w_interval = w_interval.unsqueeze(-1)

    w_float = int_w * w_interval               # FP32 近似权重
    module.weight.data.copy_(w_float.half())   # 存为 FP16

    if entry.get('bias') is not None:
        module.bias.data.copy_(entry['bias'].half())

print(f"已注入 {len(int4_map)} 个量化层的反量化权重")
```

### Step 4: 使用 torchao 打包 INT4 权重

```python
from torchao.quantization import quantize_, Int4WeightOnlyConfig

def filter_aggregator_linears(module, fqn):
    """只量化 aggregator 内部的 Linear 层"""
    return isinstance(module, nn.Linear) and 'aggregator' in fqn

# torchao 会将 FP16 权重重新量化为 INT4 并打包
# group_size=32: 每 32 个权重元素共享一个 scale
quantize_(
    model,
    Int4WeightOnlyConfig(group_size=32),
    filter_fn=filter_aggregator_linears,
)

print("torchao INT4 权重打包完成")
```

> **注意**: torchao 会重新计算量化 scale, 与 PTQ4ViT 的 `w_interval` 可能有微小差异。
> 如果精度要求极高, 可以考虑用 torchao 的底层 API 手动注入 PTQ4ViT 的 scale,
> 但一般情况下重新量化的精度差异可忽略 (< 0.1% 精度影响)。

### Step 5: 构建并挂载 QwT 补偿模块

```python
class QwTCompensation(nn.Module):
    """
    QwT 补偿模块。

    - QwT_enabled=True:  执行 FP16 低秩补偿 (x @ A @ B + bias)
    - QwT_enabled=False: 直接返回 0, 无计算开销
    """
    def __init__(self, A=None, B=None, lora_bias=None, qwt_enabled=False):
        super().__init__()
        self.qwt_enabled = qwt_enabled
        if qwt_enabled and A is not None:
            self.register_buffer('A', A.half())            # [C_in, 64]
            self.register_buffer('B', B.half())            # [64, C_out]
            self.register_buffer('lora_bias', lora_bias.float())  # [C_out]

    def forward(self, x):
        """
        输入 x 是 block 级别的输入 (残差连接前的值)。
        不是 norm 之后的值, 不是量化层的输出。
        """
        if not self.qwt_enabled:
            return 0
        return (x.half() @ self.A @ self.B).float() + self.lora_bias


# 加载并创建所有 QwT 模块
qwt_data = torch.load("output_dir/qwt_compensation.pt", map_location='cpu')
qwt_map = {}  # name → QwTCompensation

enabled_count = 0
for entry in qwt_data:
    name = entry['name']
    if entry['QwT_enabled']:
        comp = QwTCompensation(
            A=entry['A'], B=entry['B'],
            lora_bias=entry['lora_bias'],
            qwt_enabled=True,
        )
        enabled_count += 1
    else:
        comp = QwTCompensation(qwt_enabled=False)
    qwt_map[name] = comp.cuda()

print(f"QwT 补偿: {enabled_count}/{len(qwt_data)} 个模块启用")
```

### Step 6: 将 QwT 挂载到模型 Block 上

```python
def get_submodule(model, path):
    """根据路径获取子模块, 支持数字索引 (如 blocks.0)"""
    obj = model
    for part in path.split('.'):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj

def set_submodule(model, path, value):
    """根据路径设置子模块"""
    parts = path.split('.')
    parent = get_submodule(model, '.'.join(parts[:-1]))
    setattr(parent, parts[-1], value)

for qwt_name, qwt_module in qwt_map.items():
    # qwt_name 形如: "aggregator.patch_embed.blocks.5.attn_QwT"
    set_submodule(model, qwt_name, qwt_module)

print("QwT 补偿模块挂载完成")
```

### Step 7: 加载非量化参数

```python
non_quant = torch.load("output_dir/non_quant_params.pt", map_location='cpu')

for param_name, param_value in non_quant.items():
    try:
        parts = param_name.split('.')
        parent = model
        for p in parts[:-1]:
            if p.isdigit():
                parent = parent[int(p)]
            else:
                parent = getattr(parent, p)
        param = getattr(parent, parts[-1])
        param.data.copy_(param_value.to(param.device))
    except Exception as e:
        print(f"Warning: 跳过参数 {param_name}: {e}")

print(f"已加载 {len(non_quant)} 个非量化参数")
```

### Step 8: 修改 Block 的 forward 以支持 QwT 补偿

这是最关键的一步。需要修改每个 ViT Block 的 forward, 在 attn/mlp 计算后加入 QwT 补偿。

**原始 Block forward** (ptq.py 中 AttnQwT/MlpQwT 的逻辑):
```python
# AttnQwT.forward(x, pos):
#   out = ls1(attn(norm1(x), pos=pos))
#   if QwT_enabled:
#       out = out + (x.half() @ A @ B).float() + lora_bias
#   return out
#
# MlpQwT.forward(x):
#   out = ls2(mlp(norm2(x)))
#   if QwT_enabled:
#       out = out + (x.half() @ A @ B).float() + lora_bias
#   return out
#
# Block.forward(x, pos):
#   x = x + AttnQwT(x, pos)    ← 残差连接
#   x = x + MlpQwT(x)          ← 残差连接
#   return x
```

实现方式有两种:

#### 方式 A: Monkey-patch block forward (简单)

```python
import types

def make_qwt_block_forward(block, attn_qwt, mlp_qwt):
    """
    创建带 QwT 补偿的 block forward 函数。

    关键: 补偿输入是残差连接前的 x, 不是 norm 后的值。
    """
    original_attn = block.attn    # 量化后的 attention
    original_mlp = block.mlp      # 量化后的 MLP
    norm1 = block.norm1
    norm2 = block.norm2
    ls1 = block.ls1
    ls2 = block.ls2

    def new_forward(self, x, pos=None):
        # ---- 自注意力 + 补偿 ----
        if pos is not None:
            attn_out = ls1(original_attn(norm1(x), pos=pos))
        else:
            attn_out = ls1(original_attn(norm1(x)))
        attn_out = attn_out + attn_qwt(x)    # QwT 补偿 (disabled 时返回 0)
        x = x + attn_out                      # 残差连接

        # ---- MLP + 补偿 ----
        mlp_out = ls2(original_mlp(norm2(x)))
        mlp_out = mlp_out + mlp_qwt(x)        # QwT 补偿
        x = x + mlp_out                        # 残差连接

        return x

    return new_forward

# 对所有需要补偿的 block 应用
block_paths = []
# patch_embed blocks
for i in range(24):
    block_paths.append(f'aggregator.patch_embed.blocks.{i}')
# frame_blocks 和 global_blocks
for i in range(24):
    block_paths.append(f'aggregator.frame_blocks.{i}')
    block_paths.append(f'aggregator.global_blocks.{i}')

for block_path in block_paths:
    block = get_submodule(model, block_path)

    attn_qwt_name = f'{block_path}.attn_QwT'
    mlp_qwt_name = f'{block_path}.mlp_QwT'

    # 如果该 block 有 QwT 模块 (可能 enabled 也可能 disabled)
    attn_qwt = qwt_map.get(attn_qwt_name, QwTCompensation(qwt_enabled=False).cuda())
    mlp_qwt = qwt_map.get(mlp_qwt_name, QwTCompensation(qwt_enabled=False).cuda())

    new_fwd = make_qwt_block_forward(block, attn_qwt, mlp_qwt)
    block.forward = types.MethodType(new_fwd, block)

print("Block forward 已修改, 支持 QwT 补偿")
```

#### 方式 B: 自定义 Block 类替换 (更规范)

```python
class DeployBlock(nn.Module):
    """部署版 ViT Block, 集成 W4 量化权重 + QwT 补偿"""

    def __init__(self, original_block, attn_qwt, mlp_qwt):
        super().__init__()
        # 从原始 block 搬运子模块 (torchao 已量化的)
        self.norm1 = original_block.norm1
        self.attn = original_block.attn
        self.ls1 = original_block.ls1
        self.norm2 = original_block.norm2
        self.mlp = original_block.mlp
        self.ls2 = original_block.ls2
        # QwT 补偿
        self.attn_qwt = attn_qwt  # QwTCompensation
        self.mlp_qwt = mlp_qwt    # QwTCompensation

    def forward(self, x, pos=None):
        # Attention + QwT
        if pos is not None:
            attn_out = self.ls1(self.attn(self.norm1(x), pos=pos))
        else:
            attn_out = self.ls1(self.attn(self.norm1(x)))
        attn_out = attn_out + self.attn_qwt(x)
        x = x + attn_out

        # MLP + QwT
        mlp_out = self.ls2(self.mlp(self.norm2(x)))
        mlp_out = mlp_out + self.mlp_qwt(x)
        x = x + mlp_out

        return x

# 替换所有 block
for block_path in block_paths:
    block = get_submodule(model, block_path)
    attn_qwt = qwt_map.get(f'{block_path}.attn_QwT', QwTCompensation(qwt_enabled=False).cuda())
    mlp_qwt = qwt_map.get(f'{block_path}.mlp_QwT', QwTCompensation(qwt_enabled=False).cuda())
    deploy_block = DeployBlock(block, attn_qwt, mlp_qwt).cuda()
    set_submodule(model, block_path, deploy_block)
```

---

## 4. VGGT 推理流程 (decoder 部分)

VGGT 的 decoder 使用 **frame_blocks 和 global_blocks 交替执行**, 期间需要 reshape:

```python
# aggregator.forward 中的关键逻辑:
#
# 1. 编码器 patch_embed
#    x = patch_embed(images)          # → [B*S, P, C]
#
# 2. 交替执行 frame/global blocks
#    for i in range(24):              # aa_block_num = 24
#        # ---- frame_block: per-frame 局部注意力 ----
#        x = x.view(B * S, P, C)     # 每帧独立
#        pos_local = pos.view(B * S, P, 2)
#        x = frame_blocks[i](x, pos=pos_local)
#
#        # ---- global_block: cross-frame 全局注意力 ----
#        x = x.view(B, S * P, C)     # 所有帧拼接
#        pos_global = pos.view(B, S * P, 2)
#        x = global_blocks[i](x, pos=pos_global)
#
# 其中: B=batch, S=视角数(4~8), P=每帧patch数(H/14*W/14), C=1024
```

**你不需要修改这个 reshape 逻辑**, 它是 VGGT aggregator 自身的 forward 方法。你只需要确保每个 block 的 forward 正确执行了 QwT 补偿 (Step 8 已完成)。

---

## 5. 完整推理代码

```python
@torch.no_grad()
def inference(model, images):
    """
    W4A16 量化推理。

    Args:
        images: [B, S, 3, H, W], FP32 或 FP16, 多视角输入图片
                B=batch, S=视角数

    Returns:
        predictions: dict, 包含 points, depth, conf, camera 等
    """
    model.eval()
    images = images.cuda().half()
    predictions = model(images)
    return predictions

# 使用示例:
images = torch.randn(1, 4, 3, 518, 518)  # 1 batch, 4 视角, 518x518
preds = inference(model, images)
# preds['points']:  [B, S, H, W, 3]
# preds['depth']:   [B, S, H, W, 1]
# preds['conf']:    [B, S, H, W, 1]
# preds['camera']:  [B, S, ...]
```

---

## 6. 验证正确性

### 6.1 与 PTQ 模型逐层对比

```python
# 准备相同输入
x = torch.randn(1, 4, 3, 518, 518).cuda().half()

# 方法 1: 用 ptq.py 的量化模型推理 (参考输出)
ptq_output = ptq_model(x)

# 方法 2: 用部署模型推理
deploy_output = deploy_model(x)

# 对比各 head 输出
for key in ['points', 'depth', 'conf']:
    diff = (ptq_output[key] - deploy_output[key]).abs()
    print(f"{key}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
```

### 6.2 预期精度差异

- torchao 重新量化 scale 带来的差异: `< 1e-2` (可接受)
- 如果差异过大, 检查:
  1. QwT 补偿输入是否是 block 输入 `x` (不是 `norm(x)`)
  2. `QwT_enabled` 是否正确读取
  3. 非量化参数是否正确加载

---

## 7. 性能预估

| 指标 | FP16 原始 | W4A16 + QwT |
|------|----------|-------------|
| 模型权重大小 | ~2.0 GB | **~0.6 GB** |
| 推理显存 (4视角 518px) | ~3.5 GB | **~1.5 GB** |
| 推理速度 (相对) | 1x | **1.5-2x** |
| 精度损失 | 0 | 极小 (INT4 权重 + QwT 补偿修正) |

---

## 8. 常见问题

### Q: torchao 的 group_size 应该选多少?
`Int4WeightOnlyConfig(group_size=32)` 是推荐值。group_size 越小精度越高但开销略大。32 是精度和速度的最佳平衡。

### Q: 如何确认 QwT 补偿在正确执行?
在 `QwTCompensation.forward` 中加入 debug 打印:
```python
if self.qwt_enabled:
    comp = (x.half() @ self.A @ self.B).float() + self.lora_bias
    print(f"QwT comp norm: {comp.norm():.4f}")  # 应该非零
    return comp
```

### Q: 某些 block 没有 QwT 模块怎么办?
`qwt_compensation.pt` 中只包含有 QwT 属性的 block。对于没有出现在文件中的 block, 创建 `QwTCompensation(qwt_enabled=False)` 即可 (返回 0, 不影响结果)。

### Q: 可以不使用 torchao, 纯 PyTorch 实现吗?
可以。跳过 Step 4, 直接用 Step 3 注入的 FP16 近似权重做推理。精度完全一致, 但没有 INT4 GEMM 加速 (速度与 FP16 相同, 仅节省模型存储空间)。

---

## 9. 实现清单

按顺序完成以下任务:

- [ ] 安装 torchao >= 0.5.0
- [ ] 加载原始 VGGT 模型 (`VGGT.from_pretrained`)
- [ ] 加载 `int4_weights.pt`, 反量化 INT4 → FP16 并注入模型权重
- [ ] 调用 `torchao.quantize_(model, Int4WeightOnlyConfig(group_size=32), filter_fn)` 打包
- [ ] 加载 `qwt_compensation.pt`, 创建 `QwTCompensation` (检查 `QwT_enabled`)
- [ ] 挂载 QwT 到模型 block 上
- [ ] 修改 block forward 加入 QwT 补偿 (方式 A 或 B)
- [ ] 加载 `non_quant_params.pt` 到模型
- [ ] 运行推理, 验证与 PTQ 模型输出一致
