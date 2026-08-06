import os
import json
import torch
import numpy as np
import open3d as o3d
import os.path as osp
import hydra
import logging
from importlib import reload,import_module
import sys
import time
import pdb
import json
import random
from tqdm import tqdm 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset


from omegaconf import DictConfig

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from PTQ.vggt.models.vggt import VGGT
from pi3.models.pi3 import Pi3
from utils.interfaces import infer_mv_pointclouds
from mv_recon.utils import umeyama, accuracy, completion
from utils.messages import set_default_arg, write_csv
from utils.vis_utils import save_image_grid_auto

from itertools import product

from PTQ.utils.integer import get_model_int_weight
from PTQ.utils.quant_calib import HessianQuantCalibrator, QuantCalibrator
from PTQ.utils.net_wrap import wrap_certain_modules_in_net, wrap_modules_in_net

from PTQ.quant_layers.conv import MinMaxQuantConv2d
from PTQ.quant_layers.linear import MinMaxQuantLinear
from PTQ.quant_layers.matmul import MinMaxQuantMatMul

def init_config(config_name):
    """initialize the config. Use reload to make sure it's fresh one!"""
    _,_,files =  next(os.walk("./configs"))
    if config_name+".py" in files:
        quant_cfg = import_module(f"configs.{config_name}")
    else:
        raise NotImplementedError(f"Invalid config name {config_name}")
    reload(quant_cfg)
    return quant_cfg
      

class RunConfig:
    def __init__(self, name, cfg_modifier, calib_size, config_name):
        self.name = name
        self.cfg_modifier = cfg_modifier
        self.calib_size = calib_size
        self.config_name = config_name

# 当前运行配置
global current_run_config


def linear_regression(X, Y):
    X = X.reshape(-1, X.size(-1))

    X_add_one = torch.cat([X, torch.ones(size=[X.size(0), ], device=X.device).reshape(-1, 1)], dim=-1)
    Y = Y.reshape(-1, Y.size(-1))


    X_add_one_T = X_add_one.t()
    W_overall = torch.inverse(X_add_one_T @ X_add_one) @ X_add_one_T @ Y

    W = W_overall[:-1, :]
    b = W_overall[-1, :]

    Y_pred = X @ W + b

    abs_loss = (Y - Y_pred).abs().mean()

    ss_tot = torch.sum((Y - Y.mean(dim=0)).pow(2))
    ss_res = torch.sum((Y - Y_pred).pow(2))
    r2_score = 1 - ss_res / ss_tot

    return W, b, r2_score

def ridge_regression(X, Y, lamb=1e-7):
    """
    Ridge regression (L2-regularized least squares).
    X: [N, D]
    Y: [N, C]
    lamb: regularization strength
    """
    # reshape and add bias term
    X = X.reshape(-1, X.size(-1))
    Y = Y.reshape(-1, Y.size(-1))

    ones = torch.ones(size=[X.size(0), 1], device=X.device)
    X_add_one = torch.cat([X, ones], dim=-1)  # [N, D+1]

    # Compute (X^T X + λI)
    XTX = X_add_one.T @ X_add_one   # [D+1, D+1]

    # Regularize only weights, not bias
    reg = torch.eye(XTX.shape[0], device=X.device)
    reg[-1, -1] = 0.0               # do NOT penalize bias
    XTX_reg = XTX + lamb * reg

    # Closed-form solution
    W_overall = torch.linalg.solve(XTX_reg, X_add_one.T @ Y)  # [D+1, C]

    # Split weight & bias
    W = W_overall[:-1, :]
    b = W_overall[-1, :]

    # Predictions
    Y_pred = X @ W + b

    # Metrics
    abs_loss = (Y - Y_pred).abs().mean()

    ss_tot = torch.sum((Y - Y.mean(dim=0)).pow(2))
    ss_res = torch.sum((Y - Y_pred).pow(2))
    r2_score = 1 - ss_res / ss_tot

    return W, b, r2_score


class FeatureDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item]

import torch

def svd_low_rank(W: torch.Tensor, rank: int):
    """
    Low-rank SVD factorization: W ≈ A @ B

    Args:
        W: (din, dout)
        rank: target rank r

    Returns:
        A: (din, r)
        B: (r, dout)
    """
    # full SVD
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    
    # truncate
    U_r = U[:, :rank]          # (din, r)
    S_r = S[:rank]             # (r,)
    Vh_r = Vh[:rank, :]        # (r, dout)

    # split sqrt(S)
    S_half = torch.sqrt(S_r)

    A = U_r * S_half.unsqueeze(0)       # (din, r)
    B = S_half.unsqueeze(1) * Vh_r      # (r, dout)

    return A, B


class CompensationBlock(nn.Module):
    def __init__(self, W, b, r2_score, block, linear_init=True):
        super(CompensationBlock, self).__init__()
        self.block = block

        self.lora_weight = nn.Parameter(torch.zeros((W.size(0), W.size(1))))
        self.lora_bias = nn.Parameter(torch.zeros(W.size(1)))
        # pdb.set_trace()

        if linear_init and (r2_score > 0):
            self.lora_weight.data.copy_(W)
            self.lora_bias.data.copy_(b)
        else:
            nn.init.zeros_(self.lora_weight)
            nn.init.zeros_(self.lora_bias)

    def forward(self, x, pos=None):
        if pos is not None:
            out = self.block(x, pos=pos)
        else:
            out = self.block(x)
        if self.training:
            lora_weight = self.lora_weight.float()
            out = out + x @ lora_weight + self.lora_bias
        else:
            # QwT layers run in half mode
            # lora_weight = self.lora_weight.half()
            # out = out + (x.half() @ lora_weight).float() + self.lora_bias
            lora_weight = self.lora_weight
            out = out + (x @ lora_weight).float() + self.lora_bias

        return out

def enable_quant(submodel):
    for name, module in submodel.named_modules():
        if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d) or isinstance(module, MinMaxQuantMatMul):
            module.mode = "quant_forward"

def disable_quant(submodel):
    for name, module in submodel.named_modules():
        if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d) or isinstance(module, MinMaxQuantMatMul):
            module.mode = "raw"

def convert_module_path(path):
    """
    将模块路径从 'aggregator.patch_embed.blocks.0.attn.proj' 格式
    转换为 'aggregator.patch_embed.blocks[0].attn.proj' 格式
    """
    import re
    # 使用正则表达式将 .数字. 替换为 [数字].
    converted_path = re.sub(r'\.(\d+)\.', r'[\1].', path)
    # 处理路径末尾的数字（如果有的话）
    converted_path = re.sub(r'\.(\d+)$', r'[\1]', converted_path)
    return converted_path

def safe_setattr(model, path, new_module):
    """安全设置嵌套模块属性"""
    try:
        parts = path.split('.')
        current = model
        
        for part in parts[:-1]:
            if '[' in part and ']' in part:
                name = part.split('[')[0]
                index = int(part.split('[')[1].split(']')[0])
                current = getattr(current, name)[index]
            else:
                current = getattr(current, part)
        
        setattr(current, parts[-1], new_module)
        return True
    except Exception as e:
        print(f"设置模块失败: {e}")
        return False

def linear_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    module.raw_input.append(input[0].cpu().detach())

def generate_compensation_model_from_layer(q_model, wrapped_modules, calib_loader, seq_id_map):
    # pdb.set_trace()
    # wrapped_modules: dict(name -> module)
    tmp = wrapped_modules.items()
    q = tqdm(tmp, desc="Compensation")
    
    hooks = []
    for name, module in q:
        hooks.append(module.register_forward_hook(linear_forward_hook))
    
    for seq_idx, (seq_name, ids) in enumerate(seq_id_map.items(), start=1):
        data = calib_loader.get_data(sequence_name=seq_name, ids=ids)
        imgs: torch.Tensor     = data['images']       # (N, 3, H, W)
        with torch.no_grad():
            pred = q_model(imgs.cuda())                  # (B, N, H, W, 3)
    
    for name, module in q:
        # if 'qkv' in name:
        #     # skip qkv layers
        #     continue
        # if 'fc1' in name:
        #     # skip fc1 layers
        #     continue
        # if 'fc2' in name:
        #     # skip fc2 layers
        #     continue
        # if "proj" in name:
        #     # skip proj layers
        #     continue
        
        # ===============================
        # 1. 获取该 block 的输入与输出
        # ===============================
        # 已经由你提前缓存好了
        module.raw_input = torch.cat(module.raw_input, dim=0)
        t_in_fp = module.raw_input          # shape: [N, C, ...]
        # fp_out  = module.raw_out            # full precision out

        # 若你没有 quant_out，可以这样自动算:
        if not hasattr(module, "raw_out"):
            disable_quant(module)
            # disable_quant(module)
            with torch.no_grad():
                fp_out = module(module.raw_input.cuda()).detach().cpu()
        else:
            fp_out = module.raw_out
        
        # 若你没有 quant_out，可以这样自动算:
        if not hasattr(module, "quant_out"):
            enable_quant(module)
            # disable_quant(module)
            with torch.no_grad():
                quant_out = module(module.raw_input.cuda()).detach().cpu()
        else:
            quant_out = module.quant_out

        # 补偿目标：FP - Q
        target = fp_out - quant_out         # Δ

        # ===============================
        # 2. 做线性回归去拟合 Δ = W x + b
        # ===============================
        # 注意：linear_regression 接受 input, target
        W, b, r2_score = linear_regression(
            t_in_fp.cuda(), 
            target.cuda()
        )
        del t_in_fp, target, fp_out, quant_out
        if(r2_score > 0):
            # print(f"R2 score for layer {name}: {r2_score.item():.6f}")
            logging.info(f"R2 score for layer {name}: {r2_score.item():.6f}")
        else:
            # print(f"Warning: Negative R2 score for layer {name}: {r2_score.item():.6f}")
            logging.warning(f"Warning: Negative R2 score for layer {name}: {r2_score.item():.6f}")

        # ===============================
        # 3. 创建 CompensationBlock
        # ===============================
        comp = CompensationBlock(
            W=W, 
            b=b,
            r2_score=r2_score,
            block=module,             # 原模块
            linear_init=True
        )

        # ===============================
        # 4. 替换量化模型中的该 block
        # ===============================
        # 如果 wrapped_modules 另外记录了路径，你可以写：
        wrapped_modules[name] = comp

        # 如果模型结构中需要实际替换，也做掉：
        name = convert_module_path(name)

        if safe_setattr(q_model, name, comp):
            print("替换成功")
        else:
            print("替换失败")

        q_model = q_model.cuda()

    return q_model


def recon_vit(cur_inp, net):
    masks = None
    cur_inp = net.prepare_tokens_with_masks(cur_inp, masks)
    cur_inp = cur_inp.cuda()
    len_blocks = len(net.blocks)
    with torch.no_grad():
        for i in range(len_blocks):
            block = net.blocks[i]
            # ===============================
            # 1. 获取该 block 的输入与输出
            # ===============================
            disable_quant(block)
            fp_out = block(cur_inp).detach().cpu()
            enable_quant(block)
            quant_out = block(cur_inp).detach().cpu()
            next_inp = fp_out
            # 补偿目标：FP - Q
            target = fp_out - quant_out         # Δ

            # ===============================
            # 2. 做线性回归去拟合 Δ = W x + b
            # ===============================
            # 注意：linear_regression 接受 input, target
            W, b, r2_score = linear_regression(
                cur_inp.cuda(), 
                target.cuda()
            )
            logging.info(f"R2 score for Vit block {i}: {r2_score.item():.6f}")
            
            comp = CompensationBlock(
                W=W, 
                b=b,
                r2_score=r2_score,
                block=block,             # 原模块
                linear_init=True
            )

            net.blocks[i] = comp
            cur_inp = next_inp.cuda()
    return net

def recon_blocks(cur_inp, B, S, P, C, pos, net):
    cur_inp = cur_inp.cuda()
    pos = pos.cuda()
    with torch.no_grad():
        for i in range(net.aa_block_num):
            # pdb.set_trace()
            if cur_inp.shape != (B * S, P, C):
                cur_inp = cur_inp.view(B, S, P, C).view(B * S, P, C)

            if pos is not None and pos.shape != (B * S, P, 2):
                pos = pos.view(B, S, P, 2).view(B * S, P, 2)

            
            block = net.frame_blocks[i]
            disable_quant(block)
            fp_out = block(cur_inp, pos=pos).detach().cpu()
            enable_quant(block)
            quant_out = block(cur_inp, pos=pos).detach().cpu()

            next_inp = fp_out
            
            target = fp_out - quant_out         # Δ

            W, b, r2_score = linear_regression(
                cur_inp.cuda(), 
                target.cuda()
            )
            logging.info(f"R2 score for frame block {i}: {r2_score.item():.6f}")
            comp = CompensationBlock(
                W=W, 
                b=b,
                r2_score=r2_score,
                block=block,             # 原模块
                linear_init=True
            )

            # ===============================
            net.frame_blocks[i] = comp
            cur_inp = next_inp.cuda()

            if cur_inp.shape != (B, S * P, C):
                cur_inp = cur_inp.view(B, S, P, C).view(B, S * P, C)

            if pos is not None and pos.shape != (B, S * P, 2):
                pos = pos.view(B, S, P, 2).view(B, S * P, 2)
            block = net.global_blocks[i]
            disable_quant(block)
            fp_out = block(cur_inp, pos=pos).detach().cpu()
            enable_quant(block)
            quant_out = block(cur_inp, pos=pos).detach().cpu()

            next_inp = fp_out
            
            target = fp_out - quant_out         # Δ

            W, b, r2_score = linear_regression(
                cur_inp.cuda(), 
                target.cuda()
            )
            logging.info(f"R2 score for global block {i}: {r2_score.item():.6f}")
            comp = CompensationBlock(
                W=W, 
                b=b,
                r2_score=r2_score,
                block=block,             # 原模块
                linear_init=True
            )

            net.global_blocks[i] = comp
            cur_inp = next_inp.cuda()
    return net


class AttnQwT(nn.Module):
    def __init__(self, W, b, r2_score, block, linear_init=True):
        super(AttnQwT, self).__init__()
        self.norm1 = block.norm1
        self.attn = block.attn
        self.ls1 = block.ls1
        self.QwT_enabled = False

        self.lora_weight = nn.Parameter(torch.zeros((W.size(0), W.size(1))))
        self.lora_bias = nn.Parameter(torch.zeros(W.size(1)))
        # pdb.set_trace()

        # if linear_init and (r2_score > 0.8448):
        if linear_init and (r2_score > 0):
            self.lora_weight.data.copy_(W)
            self.lora_bias.data.copy_(b)
            self.A, self.B = svd_low_rank(W, 64)
            self.QwT_enabled = True
        else:
            nn.init.zeros_(self.lora_weight)
            nn.init.zeros_(self.lora_bias)
    def forward(self, x, pos=None):
        out = self.ls1(self.attn(self.norm1(x), pos=pos))
        if self.QwT_enabled == False:
            return out
        # QwT layers run in half mode
        # lora_weight = self.lora_weight.half()
        # out = out + (x.half() @ lora_weight).float() + self.lora_bias
        else:
            A = self.A.half()
            B = self.B.half()
            # print(out.device, x.device, A.device, B.device, self.lora_bias.device)
            # pdb.set_trace()
            out = out + (x.half() @ A @ B).float() + self.lora_bias
            return out
        
class MlpQwT(nn.Module):
    def __init__(self, W, b, r2_score, block, linear_init=True):
        super(MlpQwT, self).__init__()
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.ls2 = block.ls2
        self.QwT_enabled = False

        self.lora_weight = nn.Parameter(torch.zeros((W.size(0), W.size(1))))
        self.lora_bias = nn.Parameter(torch.zeros(W.size(1)))
        # pdb.set_trace()

        # if linear_init and (r2_score > 0.7645):
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
        # QwT layers run in half mode
        # lora_weight = self.lora_weight.half()
        # out = out + (x.half() @ lora_weight).float() + self.lora_bias
        if self.QwT_enabled == False:
            return out
        else:
            A = self.A.half()
            B = self.B.half()
            out = out + (x.half() @ A @ B).float() + self.lora_bias
            return out

import torch

@torch.no_grad()
def tail_relative_error(
    fp_out: torch.Tensor,
    quant_out: torch.Tensor,
    tail_ratio: float = 0.01,
    eps: float = 1e-8,
):
    """
    Compute Tail Relative Error (TRE) between fp and quant outputs.

    Args:
        fp_out: FP32 output tensor
        quant_out: Quantized output tensor (same shape)
        tail_ratio: ratio of top-magnitude elements (e.g. 0.01 = top 1%)
        eps: numerical stability

    Returns:
        tre: scalar tensor
    """
    assert fp_out.shape == quant_out.shape

    # flatten all but batch is also ok; here we flatten everything
    y = fp_out.reshape(-1)
    y_q = quant_out.reshape(-1)

    # number of tail elements
    k = max(1, int(tail_ratio * y.numel()))

    # top-k by magnitude (FP output)
    _, idx = torch.topk(y.abs(), k, largest=True, sorted=False)

    y_tail = y[idx]
    yq_tail = y_q[idx]

    # relative squared error on tail
    num = (y_tail - yq_tail).pow(2).sum()
    den = y_tail.pow(2).sum() + eps

    tre = num / den
    return tre


def compute_cosine_similarity_flat(fp_out, quant_out, dim=1, eps=1e-8):
    """
    计算两个张量之间的平均余弦相似度（使用双精度计算）
    
    参数:
        fp_out: 浮点模型输出张量
        quant_out: 量化模型输出张量  
        dim: 计算相似度的维度
        eps: 防止除零的小常数
    
    返回:
        平均余弦相似度（双精度）
    """
    # pdb.set_trace()
    # 检查输入形状是否一致
    if fp_out.shape != quant_out.shape:
        raise ValueError(f"输入张量形状不一致: {fp_out.shape} 和 {quant_out.shape}")
    
    # 转换为双精度
    if fp_out.dtype != torch.float64:
        fp_out = fp_out.double()
    if quant_out.dtype != torch.float64:
        quant_out = quant_out.double()
    
    # 展平为 (B, P) 形状
    fp_out_flat = fp_out.view(fp_out.size(0), -1)
    quant_out_flat = quant_out.view(quant_out.size(0), -1)
    
    # 计算余弦相似度
    similarities = F.cosine_similarity(fp_out_flat, quant_out_flat, dim=dim, eps=eps)
    
    # 可选：确保结果在有效范围内（针对极端情况）
    # similarities = torch.clamp(similarities, -1.0, 1.0)
    
    return similarities.mean().item()  # 返回Python浮点数

def generate_binary_list(p, l):
    """
    生成一个0和1的整数列表，其中1的比例严格等于p
    参数:
    p: 1出现的概率（0到1之间的浮点数）
    l: 列表长度（正整数）
    返回: 0和1组成的整数列表
    """
    if not 0 <= p <= 1:
        raise ValueError("概率p必须在0到1之间")
    if l <= 0 or not isinstance(l, int):
        raise ValueError("长度l必须是正整数")
    
    # 计算1的个数（四舍五入）
    k = round(l * p)
    # 确保k在有效范围内
    k = max(0, min(l, k))
    # 创建整数列表：k个1和(l-k)个0
    binary_list = [1] * k + [0] * (l - k)
    # 随机打乱列表
    random.shuffle(binary_list)

    # t = l // k if p > 0 else 0
    # binary_list = [0] * l
    # for i in range(k):
    #     idx = int(i * t)
    #     if idx >= l:
    #         idx = l - 1
    #     binary_list[idx] = 1
    
    return binary_list, k

## 以下是对mlp/attn的模块级补偿
def recon_vit_module_wise(cur_inp, net, p=0.0, sim=0.0, Tau=0.01):
    def attn_module(block, x, pos=None):
        return block.ls1(block.attn(block.norm1(x), pos=pos))

    def mlp_module(block, x):
        return block.ls2(block.mlp(block.norm2(x)))
    masks = None
    cur_inp = net.prepare_tokens_with_masks(cur_inp, masks)
    cur_inp = cur_inp.cuda()
    len_blocks = len(net.blocks)

    rand_list, num_enabled = generate_binary_list(p, len_blocks*2)
    logging.info(f"Total {num_enabled} / {len_blocks*2} modules enabled for QwT compensation.")
    logging.info(f"Random list: {rand_list}")
    tau_list = []
    with torch.no_grad():
        for i in range(len_blocks):
            block = net.blocks[i]
            # ===============================
            # 1. 获取 attn module 的输入与输出
            # ===============================
            disable_quant(block)
            # next_inp = cur_inp + attn_module(block, cur_inp).cuda()
            fp_out = attn_module(block, cur_inp).detach().cpu()
            enable_quant(block)
            quant_out = attn_module(block, cur_inp).detach().cpu()

            # 补偿目标：FP - Q
            target = fp_out - quant_out         # Δ
            
            # similarity =  compute_cosine_similarity_flat(fp_out, quant_out, dim=1)
            # logging.info(f"cosine similarity for Vit block {i} attn module: {similarity:.6f}")
            tau =  tail_relative_error(fp_out, quant_out)
            logging.info(f"tail relative error for Vit block {i} attn module: {tau.item():.6f}")
            tau_list.append(tau.item())
            # W, b, r2_score = ridge_regression(
            W, b, r2_score = linear_regression(
                cur_inp.cuda(), 
                target.cuda()
            )
            # logging.info(f"R2 score for Vit block {i} attn module: {r2_score.item():.6f}")
            # if similarity > sim and rand_list[i*2] == 0:
            if tau < Tau and rand_list[i*2] == 0:
                r2_score = -3.0
                logging.info(f"close Vit block {i} attn module QwT")
            comp = AttnQwT(
                W=W, 
                b=b,
                # r2_score=-3.0,   # 注意力补偿效果不佳，强制初始化为0
                r2_score=r2_score,
                block=block,             # 原模块
                linear_init=True
            )
            comp.cuda()
            next_inp = cur_inp + comp(cur_inp).cuda()
            net.blocks[i].attn_QwT = comp
            cur_inp = next_inp.cuda()
            # ==============================
            # 获得 mlp module 的输入与输出
            # ==============================
            disable_quant(block)
            # next_inp = cur_inp + mlp_module(block, cur_inp).cuda()
            fp_out = mlp_module(block, cur_inp).detach().cpu()
            enable_quant(block)
            quant_out = mlp_module(block, cur_inp).detach().cpu()

            # 补偿目标：FP - Q
            target = fp_out - quant_out         # Δ
            # similarity =  compute_cosine_similarity_flat(fp_out, quant_out, dim=1)
            # logging.info(f"cosine similarity for Vit block {i} mlp module: {similarity:.6f}")
            tau =  tail_relative_error(fp_out, quant_out)
            logging.info(f"tail relative error for Vit block {i} mlp module: {tau.item():.6f}")
            tau_list.append(tau.item())
            # W, b, r2_score = ridge_regression(
            W, b, r2_score = linear_regression(
                cur_inp.cuda(), 
                target.cuda()
            )
            
            # logging.info(f"R2 score for Vit block {i} mlp module: {r2_score.item():.6f}")
            # if similarity > sim and rand_list[i*2 + 1] == 0:
            if tau < Tau and rand_list[i*2 + 1] == 0:
                r2_score = -3.0
                logging.info(f"close Vit block {i} mlp module QwT")
            
            comp = MlpQwT(
                W=W, 
                b=b,
                # r2_score=-3.0,   # mlp补偿效果不佳，强制初始化为0
                r2_score=r2_score,
                block=block,             # 原模块
                linear_init=True
            )
            comp.cuda()
            next_inp = cur_inp + comp(cur_inp).cuda()
            net.blocks[i].mlp_QwT = comp
            cur_inp = next_inp.cuda()
            net.blocks[i].module_QwT = True
    
    logging.info(f"Overall 25th percentile tail relative error: {np.percentile(tau_list, 25):.6f}")
    logging.info(f"Overall 50th percentile tail relative error: {np.percentile(tau_list, 50):.6f}")
    logging.info(f"Overall 75th percentile tail relative error: {np.percentile(tau_list, 75):.6f}")
    return net

def recon_blocks_module_wise(cur_inp, B, S, P, C, pos, net, p=0.0, sim=0.0, Tau=0.01):
    def attn_module(block, x, pos=None):
        return block.ls1(block.attn(block.norm1(x), pos=pos))

    def mlp_module(block, x):
        return block.ls2(block.mlp(block.norm2(x)))
    cur_inp = cur_inp.cuda()
    pos = pos.cuda()
    rand_list, num_enabled = generate_binary_list(p, net.aa_block_num * 4)
    logging.info(f"Total {num_enabled} / {net.aa_block_num *4} modules enabled for QwT compensation.")
    logging.info(f"Random list: {rand_list}")
    tau_list = []
    with torch.no_grad():
        for i in range(net.aa_block_num):
            # pdb.set_trace()
            if cur_inp.shape != (B * S, P, C):
                cur_inp = cur_inp.view(B, S, P, C).view(B * S, P, C)

            if pos is not None and pos.shape != (B * S, P, 2):
                pos = pos.view(B, S, P, 2).view(B * S, P, 2)

            block = net.frame_blocks[i]
            disable_quant(block)
            # next_inp = cur_inp + attn_module(block, cur_inp, pos=pos).cuda()
            fp_out = attn_module(block, cur_inp, pos=pos).detach().cpu()
            enable_quant(block)
            quant_out = attn_module(block, cur_inp, pos=pos).detach().cpu()
            
            target = fp_out - quant_out         # Δ
            # similarity =  compute_cosine_similarity_flat(fp_out, quant_out, dim=1)
            # logging.info(f"cosine similarity for frame block {i} attn module: {similarity:.6f}")
            tau =  tail_relative_error(fp_out, quant_out)
            logging.info(f"tail relative error for frame block {i} attn module: {tau.item():.6f}")
            tau_list.append(tau.item())

            # W, b, r2_score = ridge_regression(
            W, b, r2_score = linear_regression(
                cur_inp.cuda(), 
                target.cuda()
            )
            # logging.info(f"R2 score for frame block {i} attn module: {r2_score.item():.6f}")
            # if similarity > sim and rand_list[i*4] == 0:
            if tau < Tau and rand_list[i*4] == 0:
                r2_score = -3.0
                logging.info(f"close frame block {i} attn module QwT")
            comp = AttnQwT(
                W=W, 
                b=b,
                # r2_score=-3.0,   # 局部注意力补偿效果不佳，强制初始化为0
                r2_score=r2_score,
                block=block,             # 原模块
                linear_init=True
            )
            comp.cuda()
            next_inp = cur_inp + comp(cur_inp, pos=pos).cuda()
            net.frame_blocks[i].attn_QwT = comp
            cur_inp = next_inp.cuda()
            # ===============================
            # 获得 mlp module 的输入与输出
            # ===============================
            disable_quant(block)
            # next_inp = cur_inp + mlp_module(block, cur_inp).cuda()
            fp_out = mlp_module(block, cur_inp).detach().cpu()
            enable_quant(block)
            quant_out = mlp_module(block, cur_inp).detach().cpu()
            
            target = fp_out - quant_out         # Δ
            # similarity =  compute_cosine_similarity_flat(fp_out, quant_out, dim=1)
            # logging.info(f"cosine similarity for frame block {i} mlp module: {similarity:.6f}")
            tau =  tail_relative_error(fp_out, quant_out)
            logging.info(f"tail relative error for frame block {i} mlp module: {tau.item():.6f}")
            tau_list.append(tau.item())

            # W, b, r2_score = ridge_regression(
            W, b, r2_score = linear_regression(
                cur_inp.cuda(), 
                target.cuda()
            )
            # logging.info(f"R2 score for frame block {i} mlp module: {r2_score.item():.6f}")
            # if similarity > sim and rand_list[i*4 + 1] == 0:
            if tau < Tau and rand_list[i*4 + 1] == 0:
                r2_score = -3.0
                logging.info(f"close frame block {i} mlp module QwT")
            comp = MlpQwT(
                W=W, 
                b=b,
                # r2_score=-3.0,   # mlp补偿效果不佳，强制初始化为0
                r2_score=r2_score,
                block=block,             # 原模块
                linear_init=True
            )
            comp.cuda()
            next_inp = cur_inp + comp(cur_inp).cuda()
            net.frame_blocks[i].mlp_QwT = comp
            cur_inp = next_inp.cuda()
            net.frame_blocks[i].module_QwT = True
            # ===============================
            # ===============================

            if cur_inp.shape != (B, S * P, C):
                cur_inp = cur_inp.view(B, S, P, C).view(B, S * P, C)

            if pos is not None and pos.shape != (B, S * P, 2):
                pos = pos.view(B, S, P, 2).view(B, S * P, 2)

            block = net.global_blocks[i]
            disable_quant(block)
            # next_inp = cur_inp + attn_module(block, cur_inp, pos=pos).cuda()
            fp_out = attn_module(block, cur_inp, pos=pos).detach().cpu()
            enable_quant(block)
            quant_out = attn_module(block, cur_inp, pos=pos).detach().cpu()
            
            target = fp_out - quant_out         # Δ
            # similarity =  compute_cosine_similarity_flat(fp_out, quant_out, dim=1)
            # logging.info(f"cosine similarity for global block {i} attn module: {similarity:.6f}")
            tau =  tail_relative_error(fp_out, quant_out)
            logging.info(f"tail relative error for global block {i} attn module: {tau.item():.6f}")
            tau_list.append(tau.item())

            # W, b, r2_score = ridge_regression(
            W, b, r2_score = linear_regression(
                cur_inp.cuda(), 
                target.cuda()
            )
            # logging.info(f"R2 score for global block {i} attn module: {r2_score.item():.6f}")
            # if similarity > sim and rand_list[i*4 + 2] == 0:
            if tau < Tau and rand_list[i*4 + 2] == 0:
                r2_score = -3.0
                logging.info(f"close global block {i} attn module QwT")
            comp = AttnQwT(
                W=W, 
                b=b,
                # r2_score=-3.0,   # 全局注意力补偿效果不佳，强制初始化为0
                r2_score=r2_score,
                block=block,             # 原模块
                linear_init=True
            )
            comp.cuda()
            next_inp = cur_inp + comp(cur_inp, pos=pos).cuda()
            net.global_blocks[i].attn_QwT = comp
            cur_inp = next_inp.cuda()
            # ===============================
            # 获得 mlp module 的输入与输出
            # ===============================
            disable_quant(block)
            # next_inp = cur_inp + mlp_module(block, cur_inp).cuda()
            fp_out = mlp_module(block, cur_inp).detach().cpu()
            enable_quant(block)
            quant_out = mlp_module(block, cur_inp).detach().cpu()
            
            target = fp_out - quant_out         # Δ
            # similarity =  compute_cosine_similarity_flat(fp_out, quant_out, dim=1)
            # logging.info(f"cosine similarity for global block {i} mlp module: {similarity:.6f}")
            tau =  tail_relative_error(fp_out, quant_out)
            logging.info(f"tail relative error for global block {i} mlp module: {tau.item():.6f}")
            # tau_list.append(tau.item())

            # W, b, r2_score = ridge_regression(
            W, b, r2_score = linear_regression(
                cur_inp.cuda(), 
                target.cuda()
            )
            # logging.info(f"R2 score for global block {i} mlp module: {r2_score.item():.6f}")
            # if similarity > sim and rand_list[i*4 + 3] == 0:
            if tau < Tau and rand_list[i*4 + 3] == 0:
                r2_score = -3.0
                logging.info(f"close global block {i} mlp module QwT")
            comp = MlpQwT(
                W=W, 
                b=b,
                # r2_score=-3.0,   # mlp补偿效果不佳，强制初始化为0
                r2_score=r2_score,
                block=block,             # 原模块
                linear_init=True
            )
            comp.cuda()
            next_inp = cur_inp + comp(cur_inp).cuda()
            net.global_blocks[i].mlp_QwT = comp
            cur_inp = next_inp.cuda()
            net.global_blocks[i].module_QwT = True
        
    logging.info(f"25th percentile of tail relative error across all modules: {np.percentile(tau_list, 25):.6f}")
    logging.info(f"50th percentile of tail relative error across all modules: {np.percentile(tau_list, 50):.6f}")
    logging.info(f"75th percentile of tail relative error across all modules: {np.percentile(tau_list, 75):.6f}")
    return net


def generate_compensation_model_from_block(q_model, calib_loader, seq_id_map):
    q_model.eval()
    with torch.no_grad():
        # pdb.set_trace()
        inputs = []
        for seq_idx, (seq_name, ids) in enumerate(seq_id_map.items(), start=1):
            batch = calib_loader.get_data(sequence_name=seq_name, ids=ids)
            inputs.append(batch['images'])
        inputs = torch.stack(inputs, dim=0).to('cuda')  # shape: [num_samples, S, 3, H, W]
        disable_quant(q_model)
        cur_inp = q_model.aggregator.forward_before_patch_embed(inputs)
        q_model.aggregator.patch_embed = recon_vit(cur_inp, q_model.aggregator.patch_embed)
        print("VGGT patch_embed compensation done.")
        cur_inp, B, S, P, C, pos = q_model.aggregator.forward_before_blocks(inputs) 
        q_model.aggregator = recon_blocks(cur_inp, B, S, P, C, pos, q_model.aggregator)
        print("VGGT aggregator compensation done.")
        enable_quant(q_model)
        q_model.cuda()

    return q_model

def generate_compensation_model_from_module(q_model, calib_loader, seq_id_map):
    q_model.eval()
    with torch.no_grad():
        # pdb.set_trace()
        inputs = []
        for seq_idx, (seq_name, ids) in enumerate(seq_id_map.items(), start=1):
            batch = calib_loader.get_data(sequence_name=seq_name, ids=ids)
            inputs.append(batch['images'])
        inputs = torch.stack(inputs, dim=0).to('cuda')  # shape: [num_samples, S, 3, H, W]
        disable_quant(q_model)
        # Set probability p for disabling QwT
        p = 0
        Tau = 0.01
        # sim_vit = 0.9788
        # sim_vggt = 0.9916
        sim_vit = 0
        sim_vggt = 0
        cur_inp = q_model.aggregator.forward_before_patch_embed(inputs)
        q_model.aggregator.patch_embed = recon_vit_module_wise(cur_inp, q_model.aggregator.patch_embed, p=p, sim=sim_vit, Tau=Tau)
        print("VGGT patch_embed compensation done.")
        cur_inp, B, S, P, C, pos = q_model.aggregator.forward_before_blocks(inputs) 
        q_model.aggregator = recon_blocks_module_wise(cur_inp, B, S, P, C, pos, q_model.aggregator, p=p, sim=sim_vggt, Tau=Tau)
        print("VGGT aggregator compensation done.")
        enable_quant(q_model)
        q_model.cuda()

    return q_model

def evaluation(hydra_cfg, model, logger):
    all_eval_datasets: DictConfig = hydra_cfg.eval_datasets  # see configs/evaluation/mv_recon.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1.1 look up dataset config from configs/data, decide the dataset name, and load the dataset
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
        dataset_info = all_data_info[dataset_name]
        dataset = hydra.utils.instantiate(dataset_info.cfg)

        # 1.3 load pre-sampled seq-id-map
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Evaluating Multi-View Pointcloud Reconstruction of Pi3 on dataset {dataset_name}...")
        sample_config: DictConfig = dataset_info.sampling
        logger.info(f"Sampling strategy: {sample_config.strategy}")
        with open(dataset_info.seq_id_map, "r") as f:
            seq_id_map: dict = json.load(f)

        calib_loader = dataset  # 直接使用dataset作为校准样本列表


        # 1.2 ready for output directory & metrics
        output_root = osp.join(hydra_cfg.output_dir, dataset_name)
        os.makedirs(output_root, exist_ok=True)
        all_data_dict = {
            "Acc-mean":  0.0,  "Acc-med":  0.0,
            "Comp-mean": 0.0,  "Comp-med": 0.0,
            "NC-mean":   0.0,  "NC-med":   0.0,
            "NC1-mean":  0.0,  "NC1-med":  0.0,
            "NC2-mean":  0.0,  "NC2-med":  0.0,
        }

        if osp.exists(osp.join(output_root, "_all_samples.csv")):
            os.remove(osp.join(output_root, "_all_samples.csv"))  # remove old csv file
        for seq_idx, (seq_name, ids) in enumerate(seq_id_map.items(), start=1):
            # 2. load data, choose specific ids of a sequence
            data = dataset.get_data(sequence_name=seq_name, ids=ids)
            filelist: list         = data['image_paths']  # [str] * N
            images: torch.Tensor   = data['images']       # (N, 3, H, W)
            gt_pts: np.ndarray     = data['pointclouds']  # (N, H, W, 3)
            valid_mask: np.ndarray = data['valid_mask']   # (N, H, W)

            # 3. real inference, predicted pointcloud aligned to ground truth (data_h, data_w)
            data_h, data_w         = images.shape[-2:]
            pred_pts: np.ndarray   = infer_mv_pointclouds(filelist, model, hydra_cfg, (data_h, data_w))  # (N, H, W, 3)
            assert pred_pts.shape == gt_pts.shape, f"Predicted points shape {pred_pts.shape} does not match ground truth shape {gt_pts.shape}."

            # 4. save input images
            seq_name = seq_name.replace("/", "-")
            save_image_grid_auto(images, osp.join(output_root, f"{seq_name}.png"))
            colors = images.permute(0, 2, 3, 1)[valid_mask].cpu().numpy().reshape(-1, 3)

            # 5. coarse align
            c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
            pred_pts = c * np.einsum('nhwj, ij -> nhwi', pred_pts, R) + t.T

            # 6. filter invalid points
            pred_pts = pred_pts[valid_mask].reshape(-1, 3)
            gt_pts = gt_pts[valid_mask].reshape(-1, 3)

            # 7. save predicted & ground truth point clouds
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pred_pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name}-pred.ply"), pcd)

            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
            pcd_gt.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name}-gt.ply"), pcd_gt)

            # 8. ICP align refinement
            if "DTU" in dataset_name:
                threshold = 100
            else:
                threshold = 0.1

            trans_init = np.eye(4)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd,
                pcd_gt,
                threshold,
                trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )

            transformation = reg_p2p.transformation
            pcd = pcd.transform(transformation)
            
            # 9. estimate normals
            pcd.estimate_normals()
            pcd_gt.estimate_normals()
            pred_normal = np.asarray(pcd.normals)
            gt_normal = np.asarray(pcd_gt.normals)

            # o3d.io.write_point_cloud(
            #     os.path.join(
            #         save_path, f"{seq.replace('/', '_')}-mask-icp.ply"
            #     ),
            #     pcd,
            # )

            # 10. compute metrics
            acc, acc_med, nc1, nc1_med = accuracy(
                pcd_gt.points, pcd.points, gt_normal, pred_normal
            )
            comp, comp_med, nc2, nc2_med = completion(
                pcd_gt.points, pcd.points, gt_normal, pred_normal
            )
            logger.info(
                f"[{dataset_name} {seq_idx}/{len(dataset.sequence_list)}] Seq: {seq_name}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
            )

            # 11. save metrics to csv
            write_csv(osp.join(output_root, f"_all_samples.csv"), {
                "seq":       seq_name,
                "Acc-mean":  acc,
                "Acc-med":   acc_med,
                "Comp-mean": comp,
                "Comp-med":  comp_med,
                "NC1-mean":  nc1,
                "NC1-med":   nc1_med,
                "NC2-mean":  nc2,
                "NC2-med":   nc2_med,
            })
            all_data_dict["Acc-mean"]  += acc
            all_data_dict["Acc-med"]   += acc_med
            all_data_dict["Comp-mean"] += comp
            all_data_dict["Comp-med"]  += comp_med
            all_data_dict["NC-mean"]   += (nc1 + nc2) / 2
            all_data_dict["NC-med"]    += (nc1_med + nc2_med) / 2
            all_data_dict["NC1-mean"]  += nc1
            all_data_dict["NC1-med"]   += nc1_med
            all_data_dict["NC2-mean"]  += nc2
            all_data_dict["NC2-med"]   += nc2_med

            # release cuda memory
            torch.cuda.empty_cache()

        num_samples = len(dataset)
        metric_dict = {
            metric: value / num_samples
            for metric, value in all_data_dict.items()
            if metric != "model"
        }

        statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric")  # + ".csv"
        if getattr(hydra_cfg, "save_suffix", None) is not None:
            statistics_file += f"-{hydra_cfg.save_suffix}"
        statistics_file += ".csv"
        write_csv(statistics_file, metric_dict)

def optimize(hydra_cfg, model, logger, wrapped_modules):
    """QwT补偿训练"""
    all_optim_datasets: DictConfig = hydra_cfg.optim_datasets  # see configs/evaluation/mv_recon.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data

    logger.info(f"QwT开始补偿训练...")
    for idx_dataset, dataset_name in enumerate(all_optim_datasets, start=1):
        # 1.1 look up dataset config from configs/data, decide the dataset name, and load the dataset
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
        dataset_info = all_data_info[dataset_name]
        dataset = hydra.utils.instantiate(dataset_info.cfg)

        # 1.3 load pre-sampled seq-id-map
        logger.info(f"[{idx_dataset}/{len(all_optim_datasets)}] Evaluating Multi-View Pointcloud Reconstruction of Pi3 on dataset {dataset_name}...")
        sample_config: DictConfig = dataset_info.sampling
        logger.info(f"Sampling strategy: {sample_config.strategy}")
        with open(dataset_info.seq_id_map, "r") as f:
            seq_id_map: dict = json.load(f)

        optim_loader = dataset  # 直接使用dataset作为优化样本列表
        # model = generate_compensation_model_from_layer(q_model=model, wrapped_modules=wrapped_modules, calib_loader=optim_loader, seq_id_map=seq_id_map)
        # model = generate_compensation_model_from_block(q_model=model, calib_loader=optim_loader, seq_id_map=seq_id_map)
        model = generate_compensation_model_from_module(q_model=model, calib_loader=optim_loader, seq_id_map=seq_id_map)
        logger.info(f"QwT补偿训练完成。")


def compute_quantized_params(model, local_rank=0, log_file=None):
    quantized_params = 0
    for _name_, _module_ in model.named_modules():
        if len(_module_._parameters) > 0:
                for k in _module_._parameters:
                    if _module_._parameters[k] is not None:
                        # if (k == 'weight') and hasattr(_module_, 'weight_quantizer'):
                        #     n_bits_ = _module_.weight_quantizer.n_bits
                        if (k == 'weight'):
                            n_bits_ = 4
                        elif 'lora_weight' in k:
                            n_bits_ = 0
                        elif 'A' in k:
                            n_bits_ = 8
                        elif 'B' in k:
                            n_bits_ = 8
                        else:
                            n_bits_ = torch.finfo(_module_._parameters[k].dtype).bits

                        numel = _module_._parameters[k].numel()
                        num_bits = numel * n_bits_
                        quantized_params += num_bits
                        # if local_rank == 0:
                        #     write('quantized_params : {}.{} : {} * {} = {}'.format(_name_, k, n_bits_, numel, num_bits), log_file=log_file)

    return quantized_params // 8 / 1e6 #MB

def convert_to_tensor(value):
    """将保存的值转换为torch.Tensor"""
    if value is None:
        return None
    
    # 如果已经是Tensor，直接返回
    if isinstance(value, torch.Tensor):
        return value
        
    # 如果是列表，转换为Tensor
    if isinstance(value, list):
        return torch.tensor(value)
    
    # 如果是字符串，检查是否是Tensor的字符串表示
    if isinstance(value, str):
        # 尝试解析字符串表示的Tensor
        if value.startswith("tensor(") or value.startswith("Tensor("):
            # 可能是"tensor([1, 2, 3])"或"Tensor([1, 2, 3])"格式
            try:
                # 提取方括号内的内容
                start_idx = value.find('[')
                end_idx = value.rfind(']')
                if start_idx != -1 and end_idx != -1:
                    list_str = value[start_idx:end_idx+1]
                    # 使用eval转换为Python列表，然后创建Tensor
                    # 注意：eval有安全风险，但这里我们已经控制了数据来源
                    list_data = eval(list_str)
                    return torch.tensor(list_data)
            except:
                pass
    
    # 如果是数字，转换为标量Tensor
    try:
        if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).isdigit()):
            return torch.tensor(float(value))
    except:
        pass
    
    # 如果无法转换，返回原值
    return value





def model_load(model, file_path, logger=None):
    
    with open(file_path, 'r') as f:
        saved_params = json.load(f)
    cnt = 0
    # for (name, module), param in zip(model.named_modules(), saved_params):
    for name, module in model.named_modules():
        if hasattr(module, 'mode'):
            param = saved_params[cnt]
            cnt = cnt + 1
            # pdb.set_trace()
            module.calibrated = True
            module.mode = 'quant_forward'
            # 处理量化参数（将列表转换为Tensor）
            if name in param["name"]:
                for attr_name in ["w_interval", "w_qmax", "a_interval", "a_qmax"]:
                    value = convert_to_tensor(param[attr_name]) 
                    setattr(module, attr_name, value.to('cuda'))
            else:
                print(f"Warning: Module name mismatch: model has {name}, but saved param has {param['name']}")

            # for attr_name in ["w_interval", "w_qmax", "a_interval", "a_qmax"]:
            #     value = convert_to_tensor(param[attr_name]) 
            #     setattr(module, attr_name, value.to('cuda'))

    return model

# import torch

def _unwrap_state_dict(ckpt):
    """兼容 ckpt = state_dict 或 {'model': state_dict} 或 {'state_dict': state_dict}"""
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
    return ckpt

def _strip_prefix_if_present(state_dict, prefixes=("module.", "model.")):
    """去掉常见前缀（DP/DDP/自定义保存）"""
    out = {}
    for k, v in state_dict.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p):]
        out[kk] = v
    return out

def load_only_encoder_decoder(
    model,
    ckpt_path: str,
    map_location="cpu",
    include_register_token: bool = True,
    strict: bool = False,
):
    """
    只加载 encoder.* 和 decoder.*（以及可选 register_token）到 model，其它权重不动。
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    sd = _unwrap_state_dict(ckpt)
    sd = _strip_prefix_if_present(sd)

    prefixes = ["encoder.", "decoder."]
    if include_register_token:
        prefixes.append("register_token")  # 这是 Pi3.decode 必需的参数，建议一起加载

    filtered = {k: v for k, v in sd.items() if any(k.startswith(p) for p in prefixes)}

    missing, unexpected = model.load_state_dict(filtered, strict=strict)

    print(f"[load_only_encoder_decoder] loaded from: {ckpt_path}")
    print(f"  loaded keys: {len(filtered)}")
    print(f"  missing keys (in provided filtered dict): {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")
    # 你也可以打印前几十个 missing/unexpected 看看是否合理
    # print("missing:", missing[:50])
    # print("unexpected:", unexpected[:50])

    return model

   
@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):

    # 使用全局配置
    global current_run_config
    if current_run_config is None:
        raise ValueError("请先设置运行配置")
    
    name = current_run_config.name
    cfg_modifier = current_run_config.cfg_modifier
    calib_size = current_run_config.calib_size
    config_name = current_run_config.config_name


    all_test_datasets: DictConfig = hydra_cfg.test_datasets  # see configs/evaluation/mv_recon.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data
    pretrained_model_name_or_path: str = hydra_cfg.pi3.pretrained_model_name_or_path  # see configs/evaluation/relpose-angular.yaml

    # 0. create model
    # model = Pi3.from_pretrained(pretrained_model_name_or_path).to(hydra_cfg.device).eval()

    # pretrained_model_name_or_path = "/root/autodl-tmp/hf_hub/models--facebook--VGGT-1B"
    # pretrained_model_name_or_path = "/root/autodl-tmp/hf_hub/models--yyfz233--Pi3"

    MODEL_REGISTRY = {
        "vggt": "/root/autodl-tmp/hf_hub/models--facebook--VGGT-1B",
        "pi3": "/root/autodl-tmp/hf_hub/models--yyfz233--Pi3",
        # "da3": "xxx/DA3",          # todo: add DA3 model path
    }

    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name: {name}, "
            f"supported: {list(MODEL_REGISTRY.keys())}"
        )

    pretrained_model_name_or_path = MODEL_REGISTRY[name]
    # pretrained_model_name_or_path = "facebook/VGGT-1B"
    if name == "pi3":
        model = Pi3.from_pretrained(pretrained_model_name_or_path).to(hydra_cfg.device).eval()
    elif name == "vggt":
        model = VGGT.from_pretrained(pretrained_model_name_or_path).to(hydra_cfg.device).eval()

    logger = logging.getLogger("mv_recon-ptq-train")
    # logger.info(f"Loaded Pi3 from {pretrained_model_name_or_path}")
    logger.info(f"Loaded VGGT from {pretrained_model_name_or_path}")

    quant_cfg = init_config(config_name)
    quant_cfg = cfg_modifier(quant_cfg)
    # 只对VGGT aggregator主干进行量化包装
    # wrapped_modules = wrap_modules_in_net(model, quant_cfg, quantize_aggregator=True, quantize_point_head=True)

    quant_cfg = init_config(config_name)
    quant_cfg = cfg_modifier(quant_cfg)
    if quant_cfg.bit[0] == 16 and quant_cfg.bit[1] == 8:
        ckpt_path = f"/root/autodl-tmp/Pi3pth/pi3_w4.pth"
    elif quant_cfg.bit[0] == 16 and quant_cfg.bit[1] == 6:
        ckpt_path = f"/root/autodl-tmp/Pi3pth/pi3_w6.pth"
    else:
        ckpt_path = f"/root/autodl-tmp/Pi3pth/pi3_w8.pth"

    # state_dict = torch.load(ckpt_path, map_location="cpu")
    #     # 如果你的 pth 是 {"model": xxx} 这种
    # if "model" in state_dict:
    #     state_dict = state_dict["model"]
    model = load_only_encoder_decoder(model, ckpt_path, map_location="cpu", include_register_token=True, strict=False)
    # missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # logger.info(f"Loaded checkpoint from {ckpt_path}, with missing keys: {missing}, unexpected keys: {unexpected}")

    model.eval().cuda()


    wrapped_modules = wrap_modules_in_net(model, quant_cfg, quantize_aggregator=True)

    for idx_dataset, dataset_name in enumerate(all_test_datasets, start=1):
        # 1.1 look up dataset config from configs/data, decide the dataset name, and load the dataset
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
        dataset_info = all_data_info[dataset_name]
        dataset = hydra.utils.instantiate(dataset_info.cfg)

        # 1.3 load pre-sampled seq-id-map
        logger.info(f"[{idx_dataset}/{len(all_test_datasets)}] Evaluating Multi-View Pointcloud Reconstruction of Pi3 on dataset {dataset_name}...")
        sample_config: DictConfig = dataset_info.sampling
        logger.info(f"Sampling strategy: {sample_config.strategy}")
        with open(dataset_info.seq_id_map, "r") as f:
            seq_id_map: dict = json.load(f)

        calib_loader = dataset  # 直接使用dataset作为校准样本列表

        # 量化校准流程
        calib_start_time = time.time()
        quant_calibrator = HessianQuantCalibrator(model, wrapped_modules, calib_loader, seq_id_map, sequential=False, batch_size=1, device=hydra_cfg.device, logger=logger)
        quant_calibrator.batching_quant_calib()
        # param_file = f"param/w{quant_cfg.bit[0]}a{quant_cfg.bit[1]}.txt"
        # model = model_load(model, param_file, logger)
        # model.cuda()
        # logger.info(f"✅ 已从 {param_file} 加载量化参数到模型。")
        calib_end_time = time.time()
        print(f"model: {name} \n")
        print(f"calibration size: {calib_size} \n")
        print(f"bit settings: {quant_cfg.bit} \n")
        print(f"config: {config_name} \n")
        print(f"ptqsl_conv2d_kwargs: {quant_cfg.ptqsl_conv2d_kwargs} \n")
        print(f"ptqsl_linear_kwargs: {quant_cfg.ptqsl_linear_kwargs} \n")
        print(f"ptqsl_matmul_kwargs: {quant_cfg.ptqsl_matmul_kwargs} \n")
        print(f"calibration time: {(calib_end_time-calib_start_time)/60}min \n")
        print(f"VGGT aggregator量化校准完成。\n")
        logger.info(f"VGGT aggregator量化校准完成。")

        for name, module in model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = "only_activation_quant_forward"
        evaluation(hydra_cfg, model, logger)
        # qwerty_params = compute_quantized_params(model)
        # logger.info(f"量化模型参数大小: {qwerty_params} MB")
        # optimize(hydra_cfg, model, logger, wrapped_modules)
        # logger.info(f"QwT补偿训练完成。")
        # qwerty_params = compute_quantized_params(model)
        # logger.info(f"补偿后量化模型参数大小: {qwerty_params} MB")
        # evaluation(hydra_cfg, model, logger)

#--------------------------------------------------------------------------------
        # 1. 创建模型保存目录
        # model_save_dir = "/root/autodl-tmp/outputs"
        # os.makedirs(model_save_dir, exist_ok=True)
        
        # # 2. 生成唯一的模型保存文件名
        # timestamp = time.strftime("%Y%m%d-%H%M%S")
        # model_name = f"{name}_{dataset_name}_w{quant_cfg.bit[0]}a{quant_cfg.bit[1]}_{timestamp}.pt"
        # int_model_name = f"{name}_{dataset_name}_w{quant_cfg.bit[0]}a{quant_cfg.bit[1]}_{timestamp}_int.pt"
        # model_save_path = osp.join(model_save_dir, model_name)
        # int_model_save_path = osp.join(model_save_dir, int_model_name)

        # 3. 保存模型权重（包括量化参数）
        # torch.save(model.state_dict(), model_save_path)
        # logger.info(f"✅ 已保存量化模型权重至: {model_save_path}")
        
        
        # 4. 可选的：同时保存完整的量化配置
        # config_save_path = osp.join(model_save_dir, f"config_{model_name}.json")
        # with open(config_save_path, "w") as f:
        #     json.dump({
        #         "quant_bit": quant_cfg.bit,
        #         "conv2d_cfg": quant_cfg.ptqsl_conv2d_kwargs,
        #         "linear_cfg": quant_cfg.ptqsl_linear_kwargs,
        #         "matmul_cfg": quant_cfg.ptqsl_matmul_kwargs,
        #         "calib_size": calib_size,
        #         "dataset": dataset_name,
        #         "timestamp": timestamp
        #     }, f, indent=4)
        # logger.info(f"✅ 已保存量化配置文件至: {config_save_path}")

        
        # int_weights = get_model_int_weight(wrapped_modules)
        # torch.save(int_weights, int_model_save_path)
        # logger.info(f"✅ 已保存int8量化模型权重至: {int_model_save_path}")
        

        results = []
        # for module in model.modules():
        for name,module in model.named_modules():
            if hasattr(module, 'mode'):
                results.append({
                    "name": name,
                    "module": module.__class__.__name__,
                    "mode": module.mode,
                    "w_interval": getattr(module, 'w_interval', None),
                    "w_qmax": getattr(module, 'w_qmax', None),
                    "a_interval": getattr(module, 'a_interval', None),
                    "a_qmax": getattr(module, 'a_qmax', None),
                })
        
        with open(f"param/gptq4pi3/8scan_w{quant_cfg.bit[0]}a{quant_cfg.bit[1]}.txt", "w") as f:
            json.dump(results, f, indent=4, default=str)  # default=str 解决 numpy/tensor 无法直接序列化的问题
            
        logger.info(f"✅ 已保存量化参数至: param/gptq4pi3/8scan_w{quant_cfg.bit[0]}a{quant_cfg.bit[1]}.txt")


        
        # for name, module in wrapped_modules.items():
        #     module.mode = "only_weights_quant_forward"
        
        # state_dict = torch.load("/root/autodl-tmp/outputs/vggt_DTU_w8a8_20250820-205751.pt")
        # model.load_state_dict(state_dict)
#--------------------------------------------------------------------------------
        # evaluation(hydra_cfg, model, logger)
    
    del model
    torch.cuda.empty_cache()
    logger.info(f"Finished evaluating PTQ4VGGT on all datasets.")


class cfg_modifier():
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self,name,value)

    def __call__(self, cfg):
        # bit setting
        cfg.bit = self.bit_setting
        cfg.w_bit = {name: self.bit_setting[0] for name in cfg.conv_fc_name_list}
        cfg.a_bit = {name: self.bit_setting[1] for name in cfg.conv_fc_name_list}
        cfg.A_bit = {name: self.bit_setting[1] for name in cfg.matmul_name_list}
        cfg.B_bit = {name: self.bit_setting[1] for name in cfg.matmul_name_list}

        # conv2d configs
        cfg.ptqsl_conv2d_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_conv2d_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_conv2d_kwargs["metric"] = self.metric
        cfg.ptqsl_conv2d_kwargs["init_layerwise"] = False

        # linear configs
        cfg.ptqsl_linear_kwargs["n_V"] = self.linear_ptq_setting[0]
        cfg.ptqsl_linear_kwargs["n_H"] = self.linear_ptq_setting[1]
        cfg.ptqsl_linear_kwargs["n_a"] = self.linear_ptq_setting[2]
        cfg.ptqsl_linear_kwargs["metric"] = self.metric
        cfg.ptqsl_linear_kwargs["init_layerwise"] = False

        # matmul configs
        cfg.ptqsl_matmul_kwargs["metric"] = self.metric
        cfg.ptqsl_matmul_kwargs["init_layerwise"] = False

        return cfg
        

if __name__=='__main__':
    # args = parse_args()

    # names = [
    #     "vit_tiny_patch16_224",
    #     "vit_small_patch32_224",
    #     "vit_small_patch16_224",
    #     "vit_base_patch16_224",
    #     "vit_base_patch16_384",

    #     "deit_tiny_patch16_224",
    #     "deit_small_patch16_224",
    #     "deit_base_patch16_224",
    #     "deit_base_patch16_384",

    #     "swin_tiny_patch4_window7_224",
    #     "swin_small_patch4_window7_224",
    #     "swin_base_patch4_window7_224",
    #     "swin_base_patch4_window12_384",
    #     ]
    names = ["pi3"]
    metrics = ["hessian"]
    linear_ptq_settings = [(1,1,1)] # n_V, n_H, n_a
    # calib_sizes = [32,128]
    calib_sizes = [32]
    bit_settings = [(16,8), (16,6), (32,8)] # weight, activation
    # bit_settings = [(6,8)] # weight, activation
    # config_names = ["PTQ4ViT", "BasePTQ"]
    config_names = ["PTQ4ViT"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_list = []
    for name, metric, linear_ptq_setting, calib_size, bit_setting, config_name in product(names, metrics, linear_ptq_settings, calib_sizes, bit_settings, config_names):
        cfg_list.append({
            "name": name,
            "cfg_modifier":cfg_modifier(linear_ptq_setting=linear_ptq_setting, metric=metric, bit_setting=bit_setting),
            "calib_size":calib_size,
            "config_name": config_name
        })
    
    set_default_arg("evaluation", "mv_recon")
    os.environ["HYDRA_FULL_ERROR"] = '1'

    # 全局变量方式传递参数
    global current_run_config
    for cfg in cfg_list:
        current_run_config = RunConfig(
            name=cfg["name"],
            cfg_modifier=cfg["cfg_modifier"],
            calib_size=cfg["calib_size"],
            config_name=cfg["config_name"]
        )
        main()  # 不再传递任何参数，Hydra会自动处理配置



# CUDA_VISIBLE_DEVICES=2 python mv_recon/ptq.py
