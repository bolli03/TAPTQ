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
from tqdm import tqdm 
import torch.nn as nn

from torch.utils.data import Dataset

# # pdb.set_trace()
# # 获取当前文件所在的绝对路径目录（mv_recon）
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # 计算项目根目录（/root/Pi3-evaluation）
# project_root = os.path.dirname(current_dir)

# # 构建vggt模块的绝对路径
# # vggt_path = os.path.join(project_root, "PTQ", "vggt")
# PTQ_path = os.path.join(project_root, "PTQ")
# Pi3_path = os.path.join(project_root)

# # 确保路径存在且未添加过
# if os.path.exists(Pi3_path) and Pi3_path not in sys.path:
#     sys.path.insert(0, Pi3_path)
# if os.path.exists(PTQ_path) and PTQ_path not in sys.path:
#     sys.path.insert(0, PTQ_path)
    
# # 现在可以导入VGGT
# from vggt.models.vggt import VGGT


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

class FeatureDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item]


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
            lora_weight = self.lora_weight.half()
            out = out + (x.half() @ lora_weight).float() + self.lora_bias
            # lora_weight = self.lora_weight
            # out = out + (x @ lora_weight).float() + self.lora_bias

        return out

class CompensationMlp(nn.Module):
    def __init__(self, W, b, r2_score, block, linear_init=True):
        super(CompensationMlp, self).__init__()
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
        out = self.block.norm2(x)
        out = self.block.mlp(out)
        # QwT layers run in half mode
        lora_weight = self.lora_weight.half()
        out = out + (x.half() @ lora_weight).float() + self.lora_bias
        # lora_weight = self.lora_weight
        # out = out + (x @ lora_weight).float() + self.lora_bias

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


def generate_compensation_model_from_layer(q_model, wrapped_modules):
    # pdb.set_trace()

    # wrapped_modules: dict(name -> module)
    tmp = wrapped_modules.items()
    q = tqdm(tmp, desc="Compensation")
    
    # cnt = 0
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
        # pdb.set_trace()
        # # debug only calibrate first 3 layers
        # cnt += 1
        # if cnt > 3:
            # break
        
        # ===============================
        # 1. 获取该 block 的输入与输出
        # ===============================
        # 已经由你提前缓存好了
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
            module = block.mlp
            # ===============================
            # 1. 获取该 block 的输入与输出
            # ===============================
            next_inp = block(cur_inp).detach().cpu()
            cur_inp = block.forward_before_mlp(cur_inp)
            disable_quant(module)
            fp_out = module(cur_inp).detach().cpu()
            enable_quant(module)
            quant_out = module(cur_inp).detach().cpu()

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
            # ===============================
            # 3. 创建 CompensationBlock
            # ===============================
            comp = CompensationBlock(
                W=W, 
                b=b,
                r2_score=r2_score,
                block=block,             # 原模块
                linear_init=True
            )

            # ===============================
            # 4. 替换量化模型中的该 block
            # ===============================
            # net.blocks[i] = comp
            # cur_inp = fp_out.cuda()
            net.blocks[i].mlp = comp
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
            module = block.mlp
            disable_quant(block)
            next_inp = block(cur_inp, pos=pos).detach().cpu()
            cur_inp = block.forward_before_mlp(cur_inp, pos=pos)
            disable_quant(module)
            fp_out = module(cur_inp).detach().cpu()
            enable_quant(module)
            quant_out = module(cur_inp).detach().cpu()
            
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
            # net.frame_blocks[i] = comp
            net.frame_blocks[i].mlp = comp

            # cur_inp = fp_out.cuda()
            cur_inp = next_inp.cuda()
            if cur_inp.shape != (B, S * P, C):
                cur_inp = cur_inp.view(B, S, P, C).view(B, S * P, C)

            if pos is not None and pos.shape != (B, S * P, 2):
                pos = pos.view(B, S, P, 2).view(B, S * P, 2)
            block = net.global_blocks[i]
            module = block.mlp

            disable_quant(block)
            next_inp = block(cur_inp, pos=pos).detach().cpu()
            cur_inp = block.forward_before_mlp(cur_inp, pos=pos)
            disable_quant(module)
            fp_out = module(cur_inp).detach().cpu()
            enable_quant(module)
            quant_out = module(cur_inp).detach().cpu()
            
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
            
            net.global_blocks[i].mlp = comp
            cur_inp = next_inp.cuda()
    return net

# def recon_vit(cur_inp, net):
#     masks = None
#     cur_inp = net.prepare_tokens_with_masks(cur_inp, masks)
#     cur_inp = cur_inp.cuda()
#     len_blocks = len(net.blocks)
#     with torch.no_grad():
#         for i in range(len_blocks):
#             block = net.blocks[i]
#             module = block.attn
#             # ===============================
#             # 1. 获取该 block 的输入与输出
#             # ===============================
#             next_inp = block(cur_inp).detach().cpu()
#             cur_inp = block.norm1(cur_inp)
#             disable_quant(module)
#             fp_out = module(cur_inp).detach().cpu()
#             enable_quant(module)
#             quant_out = module(cur_inp).detach().cpu()

#             # 补偿目标：FP - Q
#             target = fp_out - quant_out         # Δ

#             # ===============================
#             # 2. 做线性回归去拟合 Δ = W x + b
#             # ===============================
#             # 注意：linear_regression 接受 input, target
#             W, b, r2_score = linear_regression(
#                 cur_inp.cuda(), 
#                 target.cuda()
#             )
#             logging.info(f"R2 score for Vit block {i}: {r2_score.item():.6f}")
            
#             comp = CompensationBlock(
#                 W=W, 
#                 b=b,
#                 r2_score=r2_score,
#                 block=module,             # 原模块
#                 linear_init=True
#             )

#             net.blocks[i].attn = comp
#             cur_inp = next_inp.cuda()
#     return net

# def recon_blocks(cur_inp, B, S, P, C, pos, net):
#     cur_inp = cur_inp.cuda()
#     pos = pos.cuda()
#     with torch.no_grad():
#         for i in range(net.aa_block_num):
#             # pdb.set_trace()
#             if cur_inp.shape != (B * S, P, C):
#                 cur_inp = cur_inp.view(B, S, P, C).view(B * S, P, C)

#             if pos is not None and pos.shape != (B * S, P, 2):
#                 pos = pos.view(B, S, P, 2).view(B * S, P, 2)

            
#             block = net.frame_blocks[i]
#             module = block.attn
#             disable_quant(block)
#             next_inp = block(cur_inp, pos=pos).detach().cpu()
#             cur_inp = block.norm1(cur_inp)
#             disable_quant(module)
#             fp_out = module(cur_inp, pos=pos).detach().cpu()
#             enable_quant(module)
#             quant_out = module(cur_inp, pos=pos).detach().cpu()
            
#             target = fp_out - quant_out         # Δ

#             W, b, r2_score = linear_regression(
#                 cur_inp.cuda(), 
#                 target.cuda()
#             )
#             logging.info(f"R2 score for frame block {i}: {r2_score.item():.6f}")
#             # comp = CompensationBlock(
#             #     W=W, 
#             #     b=b,
#             #     r2_score=r2_score,
#             #     block=block,             # 原模块
#             #     linear_init=True
#             # )
#             comp = CompensationBlock(
#                 W=W, 
#                 b=b,
#                 r2_score=r2_score,
#                 block=module,             # 原模块
#                 linear_init=True
#             )

#             # ===============================
#             # net.frame_blocks[i] = comp
#             net.frame_blocks[i].attn = comp

#             # cur_inp = fp_out.cuda()
#             cur_inp = next_inp.cuda()
#             if cur_inp.shape != (B, S * P, C):
#                 cur_inp = cur_inp.view(B, S, P, C).view(B, S * P, C)

#             if pos is not None and pos.shape != (B, S * P, 2):
#                 pos = pos.view(B, S, P, 2).view(B, S * P, 2)
#             block = net.global_blocks[i]
#             module = block.attn
#             disable_quant(block)
#             next_inp = block(cur_inp, pos=pos).detach().cpu()
#             cur_inp = block.norm1(cur_inp)
#             disable_quant(module)
#             fp_out = module(cur_inp, pos=pos).detach().cpu()
#             enable_quant(module)
#             quant_out = module(cur_inp, pos=pos).detach().cpu()
            
#             target = fp_out - quant_out         # Δ

#             W, b, r2_score = linear_regression(
#                 cur_inp.cuda(), 
#                 target.cuda()
#             )
#             logging.info(f"R2 score for global block {i}: {r2_score.item():.6f}")
#             # comp = CompensationBlock(
#             #     W=W, 
#             #     b=b,
#             #     r2_score=r2_score,
#             #     block=block,             # 原模块
#             #     linear_init=True
#             # )
#             comp = CompensationBlock(
#                 W=W,
#                 b=b,
#                 r2_score=r2_score,
#                 block=module,             # 原模块
#                 linear_init=True
#             )

#             # net.global_blocks[i] = comp
#             net.global_blocks[i].attn = comp
#             # cur_inp = fp_out.cuda()
#             cur_inp = next_inp.cuda()
#     return net

def generate_compensation_model_from_wrapped(q_model, calib_loader, seq_id_map):
    q_model.eval()
    with torch.no_grad():
        # pdb.set_trace()
        inputs = []
        for seq_idx, (seq_name, ids) in enumerate(seq_id_map.items(), start=1):
            batch = calib_loader.get_data(sequence_name=seq_name, ids=ids)
            inputs.append(batch['images'])
        inputs = torch.stack(inputs, dim=0).to('cuda')  # shape: [num_samples, S, 3, H, W]
        disable_quant(q_model)
        cur_inp, B, S, P, C, pos = q_model.aggregator.forward_before_blocks(inputs) 
        q_model.aggregator = recon_blocks(cur_inp, B, S, P, C, pos, q_model.aggregator)
        print("VGGT aggregator compensation done.")
        cur_inp = q_model.aggregator.forward_before_patch_embed(inputs)
        q_model.aggregator.patch_embed = recon_vit(cur_inp, q_model.aggregator.patch_embed)
        print("VGGT patch_embed compensation done.")
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
        # model = generate_compensation_model_from_layer(q_model=model, wrapped_modules=wrapped_modules)
        model = generate_compensation_model_from_wrapped(q_model=model, calib_loader=optim_loader, seq_id_map=seq_id_map)
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
                            n_bits_ = 16
                        elif 'lora_weight' in k:
                            n_bits_ = 16
                        else:
                            n_bits_ = torch.finfo(_module_._parameters[k].dtype).bits

                        numel = _module_._parameters[k].numel()
                        num_bits = numel * n_bits_
                        quantized_params += num_bits
                        # if local_rank == 0:
                        #     write('quantized_params : {}.{} : {} * {} = {}'.format(_name_, k, n_bits_, numel, num_bits), log_file=log_file)

    return quantized_params // 8 / 1e6 #MB




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

    pretrained_model_name_or_path = "/root/autodl-tmp/hf_hub/models--facebook--VGGT-1B"
    # pretrained_model_name_or_path = "facebook/VGGT-1B"
    model = VGGT.from_pretrained(pretrained_model_name_or_path).to(hydra_cfg.device).eval()

    logger = logging.getLogger("mv_recon-ptq-train")
    # logger.info(f"Loaded Pi3 from {pretrained_model_name_or_path}")
    logger.info(f"Loaded VGGT from {pretrained_model_name_or_path}")

    quant_cfg = init_config(config_name)
    quant_cfg = cfg_modifier(quant_cfg)
    # 只对VGGT aggregator主干进行量化包装
    # wrapped_modules = wrap_modules_in_net(model, quant_cfg, quantize_aggregator=True, quantize_point_head=True)
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
        # quant_calibrator.batching_quant_calib_with_QwT()
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
        evaluation(hydra_cfg, model, logger)

#--------------------------------------------------------------------------------
        # # 1. 创建模型保存目录
        # model_save_dir = "/root/autodl-tmp/outputs"
        # os.makedirs(model_save_dir, exist_ok=True)
        
        # # # 2. 生成唯一的模型保存文件名
        # timestamp = time.strftime("%Y%m%d-%H%M%S")
        # model_name = f"{name}_{dataset_name}_w{quant_cfg.bit[0]}a{quant_cfg.bit[1]}_{timestamp}.pt"
        # int_model_name = f"{name}_{dataset_name}_w{quant_cfg.bit[0]}a{quant_cfg.bit[1]}_{timestamp}_int.pt"
        # model_save_path = osp.join(model_save_dir, model_name)
        # int_model_save_path = osp.join(model_save_dir, int_model_name)

        # # 3. 保存模型权重（包括量化参数）
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
        for name, module in model.named_modules():
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
        
        with open(f"param/mix_w{quant_cfg.bit[0]}a{quant_cfg.bit[1]}.txt", "w") as f:
            json.dump(results, f, indent=4, default=str)  # default=str 解决 numpy/tensor 无法直接序列化的问题
        logger.info(f"✅ 已保存量化参数详情至: param/20scanw{quant_cfg.bit[0]}a{quant_cfg.bit[1]}.txt")

        # evaluation(hydra_cfg, model, logger)

        # nodel_save_path = "param/model_w{quant_cfg.bit[0]}a{quant_cfg.bit[1]}.pt"
        # # 保存模型权重（包括QwT参数）
        # torch.save(model.state_dict(), nodel_save_path)
        # logger.info(f"✅ 已保存量化模型权重至: {nodel_save_path}")

        optimize(hydra_cfg, model, logger, wrapped_modules)
        logger.info(f"QwT补偿训练完成。")
        evaluation(hydra_cfg, model, logger)

        
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
    names = ["vggt"]
    metrics = ["hessian"]
    linear_ptq_settings = [(1,1,1)] # n_V, n_H, n_a
    # calib_sizes = [32,128]
    calib_sizes = [32]
    # bit_settings = [(4,4), (4,8)] # weight, activation
    bit_settings = [(4,6)] # weight, activation
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



# CUDA_VISIBLE_DEVICES=6 python mv_recon/ptq.py
