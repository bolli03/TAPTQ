# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import glob
import re


def save_feature_histogram(
    feature: torch.Tensor,
    title: str,
    bins: int = 200,
):
    """
    将 feature 保存为直方图，目录为:
        ./vis/<title>_<timestamp>/

    Args:
        feature: torch.Tensor
        title: 用于目录和图标题，如 "attn_logits"
        bins: 直方图分箱数
    """


    # 构建目录：./vis/title/
    save_dir = f"./vis/{title}"
    os.makedirs(save_dir, exist_ok=True)

    # 获取目录下已有的图片数量作为下一个编号
    existing_files = glob.glob(os.path.join(save_dir, f"{title}_*_hist.png"))
    if existing_files:
        # 提取所有现有的编号
        numbers = []
        for file in existing_files:
            match = re.search(rf"{re.escape(title)}_(\d+)_hist\.png", os.path.basename(file))
            if match:
                numbers.append(int(match.group(1)))
        
        if numbers:
            # 使用最大编号+1作为新编号
            next_number = max(numbers) + 1
        else:
            next_number = 0
    else:
        next_number = 0

    # 构建文件路径
    save_path = os.path.join(save_dir, f"{title}_{next_number:04d}_hist.png")

    # flatten + to numpy
    feat = feature.detach().float().cpu().reshape(-1).numpy()

    plt.figure(figsize=(8, 6))
    plt.hist(feat, bins=bins, edgecolor="black", alpha=0.75)
    plt.title(f"{title} Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[Saved] Histogram → {save_path}")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # self.fused_attn = False
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            # save_feature_histogram(attn, title="attn_logits")
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
