import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Callable, List, Any, Tuple, Dict

from quant.quant_layer import QuantModule, UniformAffineQuantizer, StraightThrough, QuantAttention, QuantMlp
from models.resnet import BasicBlock, Bottleneck
from models.regnet import ResBottleneckBlock
from models.mobilenetv2 import InvertedResidual
import pdb
import time

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from PTQ.vggt.layers.block import Block, NestedTensorBlock


class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    Due to the branch architecture, we have to perform activation function
    and quantization after the elemental-wise add operation, therefore, we
    put this part in this class.
    """
    def __init__(self, act_quant_params: dict = {}):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer

        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.activation_function = StraightThrough()

        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)


class QuantBasicBlock(BaseQuantBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """
    def __init__(self, basic_block: BasicBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.conv1 = QuantModule(basic_block.conv1, weight_quant_params, act_quant_params)
        self.conv1.activation_function = basic_block.relu1
        self.conv2 = QuantModule(basic_block.conv2, weight_quant_params, act_quant_params, disable_act_quant=True)

        # modify the activation function to ReLU
        self.activation_function = basic_block.relu2

        if basic_block.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(basic_block.downsample[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        # copying all attributes in original block
        self.stride = basic_block.stride

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

def drop_add_residual_stochastic_depth(
    x: Tensor, residual_func: Callable[[Tensor], Tensor], sample_drop_ratio: float = 0.0, pos=None
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


class QuantBlock(BaseQuantBlock):
    """
    Implementation of Quantized Transformer Block used in ViGT/VGGT.
    """

    def __init__(self, block: Block, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.norm1 = block.norm1
        self.attn = QuantAttention(block.attn, weight_quant_params, act_quant_params)
        self.ls1 = block.ls1
        self.drop_path1 = block.drop_path1

        self.norm2 = block.norm2
        self.mlp = QuantMlp(block.mlp, weight_quant_params, act_quant_params)
        self.ls2 = block.ls2
        self.drop_path2 = block.drop_path2

        # copying all attributes in original block
        self.sample_drop_ratio = block.sample_drop_ratio

    def forward(self, x: Tensor, pos=None) -> Tensor:
        # pdb.set_trace()
        def attn_residual_func(x: Tensor, pos=None) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x, pos=pos, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
            x = drop_add_residual_stochastic_depth(
                x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            # strat_time = time.time()
            x = x + attn_residual_func(x, pos=pos)
            # end_time = time.time()
            # print(f"Attention residual function time: {end_time - strat_time:.6f} seconds")
            # strat_time = time.time()
            x = x + ffn_residual_func(x)
            # end_time = time.time()
            # print(f"FFN residual function time: {end_time - strat_time:.6f} seconds")
        print(f"QuantBlock output finished with shape: {x.shape}")
        return x

# class QuantBottleneck(BaseQuantBlock):
#     """
#     Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and -152.
#     """

#     def __init__(self, bottleneck: Bottleneck, weight_quant_params: dict = {}, act_quant_params: dict = {}):
#         super().__init__(act_quant_params)
#         self.conv1 = QuantModule(bottleneck.conv1, weight_quant_params, act_quant_params)
#         self.conv1.activation_function = bottleneck.relu1
#         self.conv2 = QuantModule(bottleneck.conv2, weight_quant_params, act_quant_params)
#         self.conv2.activation_function = bottleneck.relu2
#         self.conv3 = QuantModule(bottleneck.conv3, weight_quant_params, act_quant_params, disable_act_quant=True)

#         # modify the activation function to ReLU
#         self.activation_function = bottleneck.relu3

#         if bottleneck.downsample is None:
#             self.downsample = None
#         else:
#             self.downsample = QuantModule(bottleneck.downsample[0], weight_quant_params, act_quant_params,
#                                           disable_act_quant=True)
#         # copying all attributes in original block
#         self.stride = bottleneck.stride

#     def forward(self, x):
#         residual = x if self.downsample is None else self.downsample(x)
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out += residual
#         out = self.activation_function(out)
#         if self.use_act_quant:
#             out = self.act_quantizer(out)
#         return out


# class QuantResBottleneckBlock(BaseQuantBlock):
#     """
#     Implementation of Quantized Bottleneck Blockused in RegNetX (no SE module).
#     """

#     def __init__(self, bottleneck: ResBottleneckBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
#         super().__init__(act_quant_params)
#         self.conv1 = QuantModule(bottleneck.f.a, weight_quant_params, act_quant_params)
#         self.conv1.activation_function = bottleneck.f.a_relu
#         self.conv2 = QuantModule(bottleneck.f.b, weight_quant_params, act_quant_params)
#         self.conv2.activation_function = bottleneck.f.b_relu
#         self.conv3 = QuantModule(bottleneck.f.c, weight_quant_params, act_quant_params, disable_act_quant=True)

#         # modify the activation function to ReLU
#         self.activation_function = bottleneck.relu

#         if bottleneck.proj_block:
#             self.downsample = QuantModule(bottleneck.proj, weight_quant_params, act_quant_params,
#                                           disable_act_quant=True)
#         else:
#             self.downsample = None
#         # copying all attributes in original block
#         self.proj_block = bottleneck.proj_block

#     def forward(self, x):
#         residual = x if not self.proj_block else self.downsample(x)
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out += residual
#         out = self.activation_function(out)
#         if self.use_act_quant:
#             out = self.act_quantizer(out)
#         return out


# class QuantInvertedResidual(BaseQuantBlock):
#     """
#     Implementation of Quantized Inverted Residual Block used in MobileNetV2.
#     Inverted Residual does not have activation function.
#     """

#     def __init__(self, inv_res: InvertedResidual, weight_quant_params: dict = {}, act_quant_params: dict = {}):
#         super().__init__(act_quant_params)

#         self.use_res_connect = inv_res.use_res_connect
#         self.expand_ratio = inv_res.expand_ratio
#         if self.expand_ratio == 1:
#             self.conv = nn.Sequential(
#                 QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
#                 QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params, disable_act_quant=True),
#             )
#             self.conv[0].activation_function = nn.ReLU6()
#         else:
#             self.conv = nn.Sequential(
#                 QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
#                 QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params),
#                 QuantModule(inv_res.conv[6], weight_quant_params, act_quant_params, disable_act_quant=True),
#             )
#             self.conv[0].activation_function = nn.ReLU6()
#             self.conv[1].activation_function = nn.ReLU6()

#     def forward(self, x):
#         if self.use_res_connect:
#             out = x + self.conv(x)
#         else:
#             out = self.conv(x)
#         out = self.activation_function(out)
#         if self.use_act_quant:
#             out = self.act_quantizer(out)
#         return out


specials = {
    BasicBlock: QuantBasicBlock,
    Block: QuantBlock, 
}
