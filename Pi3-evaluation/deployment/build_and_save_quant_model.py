"""
build_and_save_quant_model.py
==============================
从已有的量化参数 txt/json 文件出发，完成:
  1. 加载 VGGT 模型
  2. 包装量化模块 (wrap_modules_in_net)
  3. 从 txt/json 导入校准好的量化参数 (跳过 Hessian 校准)
  4. 用少量校准数据计算 QwT 补偿 (module-wise SVD 低秩)
  5. 保存部署所需的全部权重:
     - int4_weights.pt       (INT4 整数权重 + scale)
     - qwt_compensation.pt   (QwT 补偿模块 A, B, bias)
     - non_quant_params.pt   (非量化层 FP16 参数)
     - deploy_config.json    (部署配置)

用法:
  CUDA_VISIBLE_DEVICES=0 python build_and_save_quant_model.py

  或通过 hydra 覆盖参数:
  python build_and_save_quant_model.py \
      ptq.quant_param_file=param/channelwise_8scan_w4a8.json \
      ptq.output_dir=outputs/w4a8_deploy

  W4A4 导出 (w4a4.txt + 4bit 激活 wrap):
  python deployment/build_and_save_quant_model.py \
      ptq.quant_param_file=param/vggt/w4a4.txt \
      ptq.output_dir=outputs/w4a4_deploy \
      ptq.bit_setting=[4,4] \
      ptq.deploy_mode=w4a4
"""

import os
import json
import torch
import numpy as np
import os.path as osp
import hydra
import logging
import time
from importlib import reload, import_module
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from PTQ.vggt.models.vggt import VGGT
from utils.messages import set_default_arg

from PTQ.quant_layers.conv import MinMaxQuantConv2d
from PTQ.quant_layers.linear import MinMaxQuantLinear
from PTQ.quant_layers.matmul import MinMaxQuantMatMul
from PTQ.utils.net_wrap import wrap_modules_in_net


# ============================================================
# 从 ptq.py 复用的核心函数
# ============================================================

def convert_to_tensor(value):
    """将保存的值转换为 torch.Tensor"""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, list):
        return torch.tensor(value)
    if isinstance(value, str):
        if value.startswith("tensor(") or value.startswith("Tensor("):
            try:
                start_idx = value.find('[')
                end_idx = value.rfind(']')
                if start_idx != -1 and end_idx != -1:
                    list_str = value[start_idx:end_idx+1]
                    list_data = eval(list_str)
                    return torch.tensor(list_data)
            except:
                pass
    try:
        if isinstance(value, (int, float)):
            return torch.tensor(float(value))
    except:
        pass
    return value


def model_load(model, file_path, logger=None):
    """从 txt/json 文件加载量化参数到已包装的模型"""
    with open(file_path, 'r') as f:
        saved_params = json.load(f)
    dev = next(model.parameters()).device
    cnt = 0
    for name, module in model.named_modules():
        if hasattr(module, 'mode'):
            if cnt >= len(saved_params):
                if logger:
                    logger.warning(f"量化参数不足: 模型有更多量化层, 已加载 {cnt} 个")
                break
            param = saved_params[cnt]
            cnt += 1
            module.calibrated = True
            module.mode = 'quant_forward'
            # 兼容两种格式: 有层名 vs 无层名
            if name in param.get("name", ""):
                for attr_name in ["w_interval", "w_qmax", "a_interval", "a_qmax"]:
                    value = convert_to_tensor(param[attr_name])
                    if isinstance(value, torch.Tensor):
                        setattr(module, attr_name, value.to(dev))
                    else:
                        setattr(module, attr_name, value)
            else:
                # 旧格式 (8scan_w4a8.txt): name 字段是类名而非层路径, 按顺序匹配
                if logger:
                    logger.info(f"按顺序匹配: {name} ← {param.get('name', '?')}")
                for attr_name in ["w_interval", "w_qmax", "a_interval", "a_qmax"]:
                    value = convert_to_tensor(param[attr_name])
                    if isinstance(value, torch.Tensor):
                        setattr(module, attr_name, value.to(dev))
                    else:
                        setattr(module, attr_name, value)
    if logger:
        logger.info(f"已加载 {cnt} 个量化层的参数")
    return model


def enable_quant(submodel):
    for name, module in submodel.named_modules():
        if isinstance(module, (MinMaxQuantLinear, MinMaxQuantConv2d, MinMaxQuantMatMul)):
            module.mode = "quant_forward"


def disable_quant(submodel):
    for name, module in submodel.named_modules():
        if isinstance(module, (MinMaxQuantLinear, MinMaxQuantConv2d, MinMaxQuantMatMul)):
            module.mode = "raw"


def linear_regression(X, Y):
    X = X.reshape(-1, X.size(-1))
    X_add_one = torch.cat([X, torch.ones(size=[X.size(0), ], device=X.device).reshape(-1, 1)], dim=-1)
    Y = Y.reshape(-1, Y.size(-1))

    X_add_one_T = X_add_one.t()
    W_overall = torch.inverse(X_add_one_T @ X_add_one) @ X_add_one_T @ Y

    W = W_overall[:-1, :]
    b = W_overall[-1, :]

    Y_pred = X @ W + b
    ss_tot = torch.sum((Y - Y.mean(dim=0)).pow(2))
    ss_res = torch.sum((Y - Y_pred).pow(2))
    r2_score = 1 - ss_res / ss_tot

    return W, b, r2_score


def svd_low_rank(W: torch.Tensor, rank: int):
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    S_half = torch.sqrt(S_r)
    A = U_r * S_half.unsqueeze(0)
    B = S_half.unsqueeze(1) * Vh_r
    return A, B


@torch.no_grad()
def tail_relative_error(fp_out, quant_out, tail_ratio=0.01, eps=1e-8):
    y = fp_out.reshape(-1)
    y_q = quant_out.reshape(-1)
    k = max(1, int(tail_ratio * y.numel()))
    _, idx = torch.topk(y.abs(), k, largest=True, sorted=False)
    y_tail = y[idx]
    yq_tail = y_q[idx]
    num = (y_tail - yq_tail).pow(2).sum()
    den = y_tail.pow(2).sum() + eps
    return num / den


# ============================================================
# QwT 补偿模块 (从 ptq.py 复用)
# ============================================================

class AttnQwT(nn.Module):
    def __init__(self, W, b, r2_score, block, linear_init=True):
        super().__init__()
        self.norm1 = block.norm1
        self.attn = block.attn
        self.ls1 = block.ls1
        self.QwT_enabled = False

        self.lora_weight = nn.Parameter(torch.zeros((W.size(0), W.size(1))))
        self.lora_bias = nn.Parameter(torch.zeros(W.size(1)))

        if linear_init and (r2_score > 0):
            self.lora_weight.data.copy_(W)
            self.lora_bias.data.copy_(b)
            self.A, self.B = svd_low_rank(W, 64)
            self.QwT_enabled = True
            del self.lora_weight
        else:
            nn.init.zeros_(self.lora_weight)
            nn.init.zeros_(self.lora_bias)

    def forward(self, x, pos=None):
        out = self.ls1(self.attn(self.norm1(x), pos=pos))
        if not self.QwT_enabled:
            return out
        else:
            A = self.A.half()
            B = self.B.half()
            out = out + (x.half() @ A @ B).float() + self.lora_bias
            return out


class MlpQwT(nn.Module):
    def __init__(self, W, b, r2_score, block, linear_init=True):
        super().__init__()
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.ls2 = block.ls2
        self.QwT_enabled = False

        self.lora_weight = nn.Parameter(torch.zeros((W.size(0), W.size(1))))
        self.lora_bias = nn.Parameter(torch.zeros(W.size(1)))

        if linear_init and (r2_score > 0):
            self.lora_weight.data.copy_(W)
            self.lora_bias.data.copy_(b)
            self.A, self.B = svd_low_rank(W, 64)
            self.QwT_enabled = True
            del self.lora_weight
        else:
            nn.init.zeros_(self.lora_weight)
            nn.init.zeros_(self.lora_bias)

    def forward(self, x, pos=None):
        out = self.ls2(self.mlp(self.norm2(x)))
        if not self.QwT_enabled:
            return out
        else:
            A = self.A.half()
            B = self.B.half()
            out = out + (x.half() @ A @ B).float() + self.lora_bias
            return out


# ============================================================
# Module-wise 补偿计算 (从 ptq.py 复用, 略作整理)
# ============================================================

def recon_vit_module_wise(cur_inp, net, Tau=0.007, logger=None):
    """对 patch_embed 的 24 个 ViT block 逐模块计算 QwT 补偿"""
    def attn_module(block, x, pos=None):
        return block.ls1(block.attn(block.norm1(x), pos=pos))

    def mlp_module(block, x):
        return block.ls2(block.mlp(block.norm2(x)))

    masks = None
    cur_inp = net.prepare_tokens_with_masks(cur_inp, masks)
    cur_inp = cur_inp.cuda()
    len_blocks = len(net.blocks)

    tau_list = []
    attn_enabled = 0
    mlp_enabled = 0

    with torch.no_grad():
        for i in range(len_blocks):
            block = net.blocks[i]

            # ---- Attn 模块补偿 ----
            disable_quant(block)
            fp_out = attn_module(block, cur_inp).detach().cpu()
            enable_quant(block)
            quant_out = attn_module(block, cur_inp).detach().cpu()
            target = fp_out - quant_out

            tau = tail_relative_error(fp_out, quant_out)
            tau_list.append(tau.item())
            if logger:
                logger.info(f"ViT block {i} attn TRE: {tau.item():.6f}")

            W, b, r2_score = linear_regression(cur_inp.cuda(), target.cuda())

            # TRE < 阈值 → 关闭补偿
            if tau < Tau:
                r2_score = -3.0
                if logger:
                    logger.info(f"  → 关闭 ViT block {i} attn QwT (TRE < {Tau})")
            else:
                attn_enabled += 1

            comp = AttnQwT(W=W, b=b, r2_score=r2_score, block=block, linear_init=True)
            comp.cuda()
            next_inp = cur_inp + comp(cur_inp).cuda()
            net.blocks[i].attn_QwT = comp
            cur_inp = next_inp.cuda()

            # ---- MLP 模块补偿 ----
            disable_quant(block)
            fp_out = mlp_module(block, cur_inp).detach().cpu()
            enable_quant(block)
            quant_out = mlp_module(block, cur_inp).detach().cpu()
            target = fp_out - quant_out

            tau = tail_relative_error(fp_out, quant_out)
            tau_list.append(tau.item())
            if logger:
                logger.info(f"ViT block {i} mlp TRE: {tau.item():.6f}")

            W, b, r2_score = linear_regression(cur_inp.cuda(), target.cuda())

            if tau < Tau:
                r2_score = -3.0
                if logger:
                    logger.info(f"  → 关闭 ViT block {i} mlp QwT (TRE < {Tau})")
            else:
                mlp_enabled += 1

            comp = MlpQwT(W=W, b=b, r2_score=r2_score, block=block, linear_init=True)
            comp.cuda()
            next_inp = cur_inp + comp(cur_inp).cuda()
            net.blocks[i].mlp_QwT = comp
            cur_inp = next_inp.cuda()
            net.blocks[i].module_QwT = True

    if logger:
        logger.info(f"patch_embed QwT 统计: attn {attn_enabled}/{len_blocks} 启用, "
                     f"mlp {mlp_enabled}/{len_blocks} 启用")
        logger.info(f"TRE 分位数: 25%={np.percentile(tau_list,25):.6f}, "
                     f"50%={np.percentile(tau_list,50):.6f}, "
                     f"75%={np.percentile(tau_list,75):.6f}")
    return net


def recon_blocks_module_wise(cur_inp, B, S, P, C, pos, net, Tau=0.007, logger=None):
    """对 frame_blocks + global_blocks 逐模块计算 QwT 补偿"""
    def attn_module(block, x, pos=None):
        return block.ls1(block.attn(block.norm1(x), pos=pos))

    def mlp_module(block, x):
        return block.ls2(block.mlp(block.norm2(x)))

    cur_inp = cur_inp.cuda()
    pos = pos.cuda()
    tau_list = []
    total_enabled = 0
    total_modules = 0

    with torch.no_grad():
        for i in range(net.aa_block_num):
            # ================================================================
            # frame_block[i] (局部注意力)
            # ================================================================
            if cur_inp.shape != (B * S, P, C):
                cur_inp = cur_inp.view(B, S, P, C).view(B * S, P, C)
            if pos is not None and pos.shape != (B * S, P, 2):
                pos = pos.view(B, S, P, 2).view(B * S, P, 2)

            block = net.frame_blocks[i]

            # -- frame attn --
            disable_quant(block)
            fp_out = attn_module(block, cur_inp, pos=pos).detach().cpu()
            enable_quant(block)
            quant_out = attn_module(block, cur_inp, pos=pos).detach().cpu()
            target = fp_out - quant_out
            tau = tail_relative_error(fp_out, quant_out)
            tau_list.append(tau.item())
            if logger:
                logger.info(f"frame_block {i} attn TRE: {tau.item():.6f}")

            W, b, r2_score = linear_regression(cur_inp.cuda(), target.cuda())
            total_modules += 1
            if tau < Tau:
                r2_score = -3.0
            else:
                total_enabled += 1

            comp = AttnQwT(W=W, b=b, r2_score=r2_score, block=block, linear_init=True)
            comp.cuda()
            next_inp = cur_inp + comp(cur_inp, pos=pos).cuda()
            net.frame_blocks[i].attn_QwT = comp
            cur_inp = next_inp.cuda()

            # -- frame mlp --
            disable_quant(block)
            fp_out = mlp_module(block, cur_inp).detach().cpu()
            enable_quant(block)
            quant_out = mlp_module(block, cur_inp).detach().cpu()
            target = fp_out - quant_out
            tau = tail_relative_error(fp_out, quant_out)
            tau_list.append(tau.item())
            if logger:
                logger.info(f"frame_block {i} mlp TRE: {tau.item():.6f}")

            W, b, r2_score = linear_regression(cur_inp.cuda(), target.cuda())
            total_modules += 1
            if tau < Tau:
                r2_score = -3.0
            else:
                total_enabled += 1

            comp = MlpQwT(W=W, b=b, r2_score=r2_score, block=block, linear_init=True)
            comp.cuda()
            next_inp = cur_inp + comp(cur_inp).cuda()
            net.frame_blocks[i].mlp_QwT = comp
            cur_inp = next_inp.cuda()
            net.frame_blocks[i].module_QwT = True

            # ================================================================
            # global_block[i] (全局注意力)
            # ================================================================
            if cur_inp.shape != (B, S * P, C):
                cur_inp = cur_inp.view(B, S, P, C).view(B, S * P, C)
            if pos is not None and pos.shape != (B, S * P, 2):
                pos = pos.view(B, S, P, 2).view(B, S * P, 2)

            block = net.global_blocks[i]

            # -- global attn --
            disable_quant(block)
            fp_out = attn_module(block, cur_inp, pos=pos).detach().cpu()
            enable_quant(block)
            quant_out = attn_module(block, cur_inp, pos=pos).detach().cpu()
            target = fp_out - quant_out
            tau = tail_relative_error(fp_out, quant_out)
            tau_list.append(tau.item())
            if logger:
                logger.info(f"global_block {i} attn TRE: {tau.item():.6f}")

            W, b, r2_score = linear_regression(cur_inp.cuda(), target.cuda())
            total_modules += 1
            if tau < Tau:
                r2_score = -3.0
            else:
                total_enabled += 1

            comp = AttnQwT(W=W, b=b, r2_score=r2_score, block=block, linear_init=True)
            comp.cuda()
            next_inp = cur_inp + comp(cur_inp, pos=pos).cuda()
            net.global_blocks[i].attn_QwT = comp
            cur_inp = next_inp.cuda()

            # -- global mlp --
            disable_quant(block)
            fp_out = mlp_module(block, cur_inp).detach().cpu()
            enable_quant(block)
            quant_out = mlp_module(block, cur_inp).detach().cpu()
            target = fp_out - quant_out
            tau = tail_relative_error(fp_out, quant_out)
            tau_list.append(tau.item())
            if logger:
                logger.info(f"global_block {i} mlp TRE: {tau.item():.6f}")

            W, b, r2_score = linear_regression(cur_inp.cuda(), target.cuda())
            total_modules += 1
            if tau < Tau:
                r2_score = -3.0
            else:
                total_enabled += 1

            comp = MlpQwT(W=W, b=b, r2_score=r2_score, block=block, linear_init=True)
            comp.cuda()
            next_inp = cur_inp + comp(cur_inp).cuda()
            net.global_blocks[i].mlp_QwT = comp
            cur_inp = next_inp.cuda()
            net.global_blocks[i].module_QwT = True

    if logger:
        logger.info(f"frame+global QwT 统计: {total_enabled}/{total_modules} 模块启用")
        logger.info(f"TRE 分位数: 25%={np.percentile(tau_list,25):.6f}, "
                     f"50%={np.percentile(tau_list,50):.6f}, "
                     f"75%={np.percentile(tau_list,75):.6f}")
    return net


# ============================================================
# 权重保存
# ============================================================

def quantize_weight_to_int(weight, w_interval, w_qmax):
    """FP 权重 → INT4 整数 (对称量化)

    处理 w_interval 的多种 shape:
      - [1,1,1,1] tensor-wise: 所有权重共享一个 scale
      - [out,1,1,1] channel-wise: 每个输出通道一个 scale
      - [3,1,1,1] qkv: Q/K/V 各一个 scale, 权重为 [3*dim, dim]

    weight 可能在 CUDA；w_interval 可能已 .cpu() 用于落盘，此处先对齐到 weight 设备再算。
    """
    weight = weight.float()
    w_interval = w_interval.float().to(weight.device)
    w_interval = w_interval.squeeze()  # 去掉所有尺寸为 1 的维度

    if w_interval.dim() == 0:
        # 标量 scale, 直接除
        pass
    elif w_interval.numel() == weight.shape[0]:
        # channel-wise: scale 数量 == 输出通道数, 直接 reshape
        w_interval = w_interval.view(-1, *([1] * (weight.dim() - 1)))
    elif w_interval.numel() < weight.shape[0] and weight.shape[0] % w_interval.numel() == 0:
        # qkv 情况: scale 数量 (如 3) < 输出通道数 (如 3072)
        # 每个 scale 覆盖 out_features // num_scales 行
        repeat_factor = weight.shape[0] // w_interval.numel()
        w_interval = w_interval.repeat_interleave(repeat_factor)
        w_interval = w_interval.view(-1, *([1] * (weight.dim() - 1)))
    else:
        # fallback: 尝试直接广播
        w_interval = w_interval.view(-1, *([1] * (weight.dim() - 1)))

    out = torch.clamp(torch.round(weight / w_interval), -w_qmax, w_qmax).to(torch.int8)
    return out.cpu()


def save_deploy_weights(model, output_dir, mode="w4a16", logger=None):
    """
    从量化 + QwT 补偿后的模型中提取并保存部署所需的全部权重。

    输出:
      output_dir/
      ├── int4_weights.pt
      ├── qwt_compensation.pt
      ├── non_quant_params.pt
      └── deploy_config.json
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device

    # ---- 1. 提取量化层 INT4 权重 ----
    log = logger.info if logger else print
    log("[1/4] 提取量化层 INT4 权重...")

    int4_data = []
    for name, module in model.named_modules():
        if not (hasattr(module, 'mode') and hasattr(module, 'w_interval')):
            continue
        entry = {
            'name': name,
            'module_class': module.__class__.__name__,
            'w_qmax': int(module.w_qmax) if not isinstance(module.w_qmax, torch.Tensor) else int(module.w_qmax.item()),
            'a_qmax': int(module.a_qmax) if not isinstance(module.a_qmax, torch.Tensor) else int(module.a_qmax.item()),
            'w_interval': module.w_interval.detach().cpu().float() if isinstance(module.w_interval, torch.Tensor) else torch.tensor(float(module.w_interval)),
            'a_interval': module.a_interval.detach().cpu().float() if isinstance(module.a_interval, torch.Tensor) else torch.tensor(float(module.a_interval)),
        }
        if hasattr(module, 'weight') and module.weight is not None:
            entry['int_weight'] = quantize_weight_to_int(
                module.weight.data, entry['w_interval'], entry['w_qmax']
            )
            entry['weight_shape'] = list(module.weight.shape)
        if hasattr(module, 'bias') and module.bias is not None:
            entry['bias'] = module.bias.data.detach().cpu().half()
        int4_data.append(entry)

    torch.save(int4_data, osp.join(output_dir, "int4_weights.pt"))
    total_int4 = sum(e['int_weight'].numel() for e in int4_data if 'int_weight' in e)
    log(f"  已保存 {len(int4_data)} 个量化层, INT4 参数量 {total_int4/1e6:.2f}M")

    # ---- 2. 提取 QwT 补偿模块 ----
    log("[2/4] 提取 QwT 补偿模块...")

    qwt_data = []
    total_qwt_params = 0
    enabled_count = 0

    for name, module in model.named_modules():
        if not hasattr(module, 'QwT_enabled'):
            continue
        entry = {
            'name': name,
            'module_class': module.__class__.__name__,
            'QwT_enabled': module.QwT_enabled,
        }
        if module.QwT_enabled and hasattr(module, 'A'):
            entry['A'] = module.A.data.detach().cpu().half()
            entry['B'] = module.B.data.detach().cpu().half()
            entry['lora_bias'] = module.lora_bias.data.detach().cpu().float()
            entry['rank'] = int(module.A.shape[1])
            total_qwt_params += module.A.numel() + module.B.numel() + module.lora_bias.numel()
            enabled_count += 1
        else:
            if hasattr(module, 'lora_bias'):
                entry['lora_bias'] = module.lora_bias.data.detach().cpu().float()
                total_qwt_params += module.lora_bias.numel()
        qwt_data.append(entry)

    torch.save(qwt_data, osp.join(output_dir, "qwt_compensation.pt"))
    log(f"  已保存 {len(qwt_data)} 个 QwT 模块, {enabled_count} 个启用, "
        f"参数量 {total_qwt_params/1e6:.2f}M")

    # ---- 3. 提取非量化参数 ----
    log("[3/4] 提取非量化层参数...")

    quant_prefixes = set()
    for name, module in model.named_modules():
        if hasattr(module, 'mode') and hasattr(module, 'w_interval'):
            quant_prefixes.add(name)
        if hasattr(module, 'QwT_enabled'):
            quant_prefixes.add(name)

    non_quant = OrderedDict()
    for pname, param in model.named_parameters():
        belongs = any(pname.startswith(qp + '.') for qp in quant_prefixes)
        if not belongs:
            non_quant[pname] = param.data.detach().cpu().half()

    torch.save(non_quant, osp.join(output_dir, "non_quant_params.pt"))
    total_nq = sum(p.numel() for p in non_quant.values())
    log(f"  已保存 {len(non_quant)} 个非量化参数, 参数量 {total_nq/1e6:.2f}M")

    # ---- 4. 部署配置 ----
    log("[4/4] 保存部署配置...")

    config = {
        'mode': mode,
        'w_bit': 4,
        'a_bit': {'w4a16': 16, 'w4a8': 8, 'w4a4': 4}.get(mode, 16),
        'w_qmax': 8,
        'compensation_type': 'QwT_SVD_rank64',
        'compensation_precision': 'FP16',
        'num_quant_layers': len(int4_data),
        'num_qwt_modules': len(qwt_data),
        'num_qwt_enabled': enabled_count,
        'total_int4_params_M': round(total_int4 / 1e6, 2),
        'total_qwt_params_M': round(total_qwt_params / 1e6, 2),
        'total_non_quant_params_M': round(total_nq / 1e6, 2),
        'estimated_model_size_MB': round(
            total_int4 * 4 / 8 / 1e6 +
            total_qwt_params * 2 / 1e6 +
            total_nq * 2 / 1e6, 2
        ),
    }

    with open(osp.join(output_dir, "deploy_config.json"), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    log(f"\n{'='*60}")
    log(f"保存完成! 模式: {mode.upper()}")
    log(f"  INT4 权重:    {total_int4 * 4 / 8 / 1e6:.2f} MB")
    log(f"  QwT 补偿:     {total_qwt_params * 2 / 1e6:.2f} MB  ({enabled_count}/{len(qwt_data)} 启用)")
    log(f"  非量化参数:   {total_nq * 2 / 1e6:.2f} MB")
    log(f"  预估总大小:   {config['estimated_model_size_MB']:.2f} MB")
    log(f"{'='*60}")

    return config


# ============================================================
# quant config 初始化 (从 ptq.py 复用)
# ============================================================

def init_config(config_name):
    _, _, files = next(os.walk("./configs"))
    if config_name + ".py" in files:
        quant_cfg = import_module(f"configs.{config_name}")
    else:
        raise NotImplementedError(f"Invalid config name {config_name}")
    reload(quant_cfg)
    return quant_cfg


class cfg_modifier():
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __call__(self, cfg):
        cfg.bit = self.bit_setting
        cfg.w_bit = {name: self.bit_setting[0] for name in cfg.conv_fc_name_list}
        cfg.a_bit = {name: self.bit_setting[1] for name in cfg.conv_fc_name_list}
        cfg.A_bit = {name: self.bit_setting[1] for name in cfg.matmul_name_list}
        cfg.B_bit = {name: self.bit_setting[1] for name in cfg.matmul_name_list}

        cfg.ptqsl_conv2d_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_conv2d_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_conv2d_kwargs["metric"] = self.metric
        cfg.ptqsl_conv2d_kwargs["init_layerwise"] = False

        cfg.ptqsl_linear_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_linear_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_linear_kwargs["n_a"] = self.linear_ptq_setting[2]
        cfg.ptqsl_linear_kwargs["metric"] = self.metric
        cfg.ptqsl_linear_kwargs["init_layerwise"] = False

        cfg.ptqsl_matmul_kwargs["metric"] = self.metric
        cfg.ptqsl_matmul_kwargs["init_layerwise"] = False

        lc = getattr(self, "linear_channelwise", None)
        if lc is not None:
            cfg.linear_channelwise = bool(lc)

        return cfg


# ============================================================
# 主流程
# ============================================================

@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    logger = logging.getLogger("build_quant_model")

    # ---- 配置 ----
    # 可通过 hydra 命令行覆盖（ptq.* 已在 configs/eval.yaml 与 evaluation/mv_recon.yaml 声明）
    def _ptq_str(key: str, default: str) -> str:
        v = OmegaConf.select(hydra_cfg, key, default=None)
        if v is None or v == "":
            return default
        return str(v)

    MODEL_PATH = _ptq_str("ptq.model_path", "/root/autodl-tmp/hf_hub/models--facebook--VGGT-1B")
    QUANT_PARAM_FILE = _ptq_str("ptq.quant_param_file", "param/vggt/8scan_w4a8.txt")
    OUTPUT_DIR = _ptq_str("ptq.output_dir", "outputs/w4a16_deploy")
    _bs = OmegaConf.select(hydra_cfg, "ptq.bit_setting", default=None)
    if _bs is not None:
        BIT_SETTING = tuple(int(x) for x in list(_bs))
    else:
        BIT_SETTING = (4, 8)  # 默认 W4A8 风格；W4A4 请传 ptq.bit_setting=[4,4]
    CONFIG_NAME = "PTQ4ViT"
    TAU = 0.007             # QwT 启用阈值
    _dm = OmegaConf.select(hydra_cfg, "ptq.deploy_mode", default=None)
    DEPLOY_MODE = str(_dm) if _dm not in (None, "") else "w4a16"

    device = hydra_cfg.device if hasattr(hydra_cfg, 'device') else "cuda"

    # ---- Step 1: 加载 VGGT 模型 ----
    logger.info(f"Step 1: 加载 VGGT 模型 ({MODEL_PATH})")
    model = VGGT.from_pretrained(MODEL_PATH).to(device).eval()
    logger.info("模型加载完成")

    # ---- Step 2: 包装量化模块 ----
    logger.info("Step 2: 包装量化模块")
    quant_cfg = init_config(CONFIG_NAME)
    modifier = cfg_modifier(
        linear_ptq_setting=(1, 1, 1),
        metric="hessian",
        bit_setting=BIT_SETTING,
    )
    quant_cfg = modifier(quant_cfg)

    # 检查是否需要 channelwise
    lc = OmegaConf.select(hydra_cfg, "ptq.linear_channelwise", default=None)
    if lc is not None:
        quant_cfg.linear_channelwise = bool(lc)

    wrapped_modules = wrap_modules_in_net(model, quant_cfg, quantize_aggregator=True)
    logger.info(f"已包装 {len(wrapped_modules)} 个量化模块")

    # ---- Step 3: 从文件加载量化参数 (跳过 Hessian 校准) ----
    logger.info(f"Step 3: 从文件加载量化参数 ({QUANT_PARAM_FILE})")
    model = model_load(model, QUANT_PARAM_FILE, logger=logger)
    enable_quant(model)
    model.cuda()
    logger.info("量化参数加载完成, 已启用量化模式")

    # ---- Step 4: 计算 QwT 补偿 ----
    logger.info("Step 4: 计算 QwT 补偿 (需要校准数据)")

    # 加载校准数据
    all_data_info = hydra_cfg.data
    # 使用 optim_datasets 或 test_datasets 作为补偿计算的数据源
    comp_datasets = OmegaConf.select(hydra_cfg, "optim_datasets", default=None)
    if comp_datasets is None:
        comp_datasets = hydra_cfg.test_datasets
    dataset_name = list(comp_datasets)[0]  # 取第一个数据集

    if dataset_name not in all_data_info:
        raise ValueError(f"数据集 {dataset_name} 未在配置中找到")

    dataset_info = all_data_info[dataset_name]
    dataset = hydra.utils.instantiate(dataset_info.cfg)
    with open(dataset_info.seq_id_map, "r") as f:
        seq_id_map = json.load(f)

    calib_loader = dataset

    # 收集输入
    model.eval()
    with torch.no_grad():
        inputs = []
        for seq_name, ids in seq_id_map.items():
            batch = calib_loader.get_data(sequence_name=seq_name, ids=ids)
            inputs.append(batch['images'])
        inputs = torch.stack(inputs, dim=0).to('cuda')
        logger.info(f"校准数据: {inputs.shape} ({len(seq_id_map)} 个序列)")

        # 4a: patch_embed 补偿
        disable_quant(model)
        cur_inp = model.aggregator.forward_before_patch_embed(inputs)
        logger.info("计算 patch_embed QwT 补偿...")
        model.aggregator.patch_embed = recon_vit_module_wise(
            cur_inp, model.aggregator.patch_embed, Tau=TAU, logger=logger
        )
        logger.info("patch_embed 补偿完成")

        # 4b: frame_blocks + global_blocks 补偿
        cur_inp, B, S, P, C, pos = model.aggregator.forward_before_blocks(inputs)
        logger.info("计算 frame/global blocks QwT 补偿...")
        model.aggregator = recon_blocks_module_wise(
            cur_inp, B, S, P, C, pos, model.aggregator, Tau=TAU, logger=logger
        )
        logger.info("frame/global blocks 补偿完成")

        enable_quant(model)
        model.cuda()

    # ---- Step 5: 保存部署权重 ----
    logger.info(f"Step 5: 保存部署权重 → {OUTPUT_DIR}")
    save_deploy_weights(model, OUTPUT_DIR, mode=DEPLOY_MODE, logger=logger)

    # ---- 清理 ----
    del model
    torch.cuda.empty_cache()
    logger.info("全部完成!")


if __name__ == '__main__':
    set_default_arg("evaluation", "mv_recon")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    main()

# python deployment/build_and_save_quant_model.py \
#   ptq.quant_param_file=param/vggt/w4a4.txt \
#   ptq.output_dir=outputs/w4a4_deploy \
#   ptq.bit_setting=[4,4] \
#   ptq.deploy_mode=w4a4