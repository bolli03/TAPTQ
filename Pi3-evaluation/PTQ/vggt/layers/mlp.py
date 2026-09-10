# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional

from torch import Tensor, nn


import os
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
    将 feature 的正值和负值分别绘制成两个独立直方图。

    保存路径:
        ./vis/<title>/<title>_<timestamp>_pos_hist.png
        ./vis/<title>/<title>_<timestamp>_neg_hist.png
    """

    # 保存目录: ./vis/title/
    save_dir = f"./vis/{title}"
    os.makedirs(save_dir, exist_ok=True)

    # 获取目录下已有的图片数量作为下一个编号
    existing_files = glob.glob(os.path.join(save_dir, f"{title}_*_neg_hist.png"))
    if existing_files:
        # 提取所有现有的编号
        numbers = []
        for file in existing_files:
            match = re.search(rf"{re.escape(title)}_(\d+)_neg_hist\.png", os.path.basename(file))
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
    neg_path = os.path.join(save_dir, f"{title}_{next_number:04d}_neg_hist.png")

    # 获取目录下已有的图片数量作为下一个编号
    existing_files = glob.glob(os.path.join(save_dir, f"{title}_*_pos_hist.png"))
    if existing_files:
        # 提取所有现有的编号
        numbers = []
        for file in existing_files:
            match = re.search(rf"{re.escape(title)}_(\d+)_pos_hist\.png", os.path.basename(file))
            if match:
                numbers.append(int(match.group(1)))
        
        if numbers:
            # 使用最大编号+1作为新编号
            next_number = max(numbers) + 1
        else:
            next_number = 0
    else:
        next_number = 0

    pos_path = os.path.join(save_dir, f"{title}_{next_number:04d}_pos_hist.png")

    # 将 tensor 转为 1D numpy
    feat = feature.detach().float().cpu().reshape(-1).numpy()

    # 分为正值 & 负值
    pos_feat = feat[feat >= 0]
    neg_feat = feat[feat <= 0]

    # -------- 绘制正值直方图 --------
    if len(pos_feat) > 0:
        plt.figure(figsize=(8, 6))
        plt.hist(pos_feat, bins=bins, edgecolor="black", alpha=0.75)
        plt.title(f"{title} Positive Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True)

        plt.savefig(pos_path, dpi=200)
        plt.close()

        print(f"[Saved] Positive Histogram → {pos_path}")

    # -------- 绘制负值直方图 --------
    if len(neg_feat) > 0:
        plt.figure(figsize=(8, 6))
        plt.hist(neg_feat, bins=bins, edgecolor="black", alpha=0.75)
        plt.title(f"{title} Negative Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True)

        plt.savefig(neg_path, dpi=200)
        plt.close()

        print(f"[Saved] Negative Histogram → {neg_path}")



class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        if not self.training and isinstance(self.act, nn.GELU) and self.act.approximate == "none" and hasattr(self.fc2, "forward_from_gelu"):
            return self.fc2.forward_from_gelu(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
