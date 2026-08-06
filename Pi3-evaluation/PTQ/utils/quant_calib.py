from numpy import isin
import torch
from PTQ.quant_layers.conv import *
from PTQ.quant_layers.linear import *
from PTQ.quant_layers.matmul import *
import torch.nn.functional as F
from tqdm import tqdm
import pdb
import os
import numpy as np
from PTQ.vggt.training.loss import compute_point_loss

def chamfer_distance(pred_points, gt_points):
    """
    pred_points: [B, N, 3] 预测点云
    gt_points: [B, M, 3] GT点云
    返回：每个batch的Chamfer距离
    """
    B, N, _ = pred_points.shape
    M = gt_points.shape[1]
    dist1 = torch.cdist(pred_points, gt_points)  # [B, N, M]
    min_dist1, _ = torch.min(dist1, dim=2)       # [B, N]
    min_dist2, _ = torch.min(dist1, dim=1)       # [B, M]
    cd = min_dist1.mean(dim=1) + min_dist2.mean(dim=1)  # [B]
    return cd.mean()  # 标量损失

class QuantCalibrator():
    """
    Modularization of quant calib.

    Notice: 
    all quant modules has method "calibration_step1" that should only store raw inputs and outputs
    all quant modules has method "calibration_step2" that should only quantize its intervals
    and we assume we could feed in all calibration data in one batch, without backward propagations

    sequential calibration is memory-friendly, while parallel calibration may consume 
    hundreds of GB of memory.
    """
    def __init__(self, net, wrapped_modules, calib_loader, sequential=True):
        self.net = net
        self.wrapped_modules = wrapped_modules
        self.calib_loader = calib_loader
        self.sequential = sequential
        self.calibrated = False
    
    def sequential_quant_calib(self):
        """
        A quick implementation of calibration.
        Assume calibration dataset could be fed at once.
        """
        # run calibration
        n_calibration_steps=2
        for step in range(n_calibration_steps):
            print(f"Start calibration step={step+1}")
            for name,module in self.wrapped_modules.items():
                # corner cases for calibrated modules
                if hasattr(module, "calibrated"):
                    if step == 1:
                        module.mode = "raw"
                    elif step == 2:
                        module.mode = "quant_forward"
                else:
                    module.mode=f'calibration_step{step+1}'
            with torch.no_grad():
                for inp,target in self.calib_loader:
                    inp=inp.cuda()
                    self.net(inp)
        
        # finish calibration
        for name,module in self.wrapped_modules.items():
            module.mode='quant_forward'
        torch.cuda.empty_cache() # memory footprint cleanup
        print("sequential calibration finished")
    
    def parallel_quant_calib(self):
        """
        A quick implementation of parallel quant calib
        Assume calibration dataset could be fed at once, and memory could hold all raw inputs/outs
        """
        # calibration step1: collect raw data
        print(f"Start calibration step=1")
        for name,module in self.wrapped_modules.items():
            # corner cases for calibrated modules
            if hasattr(module, "calibrated"):
                module.mode = "raw"
            else:
                module.mode=f'calibration_step1'
        with torch.no_grad():
            for inp,target in self.calib_loader:
                inp=inp.cuda()
                self.net(inp)
        # calibration step2: each module run calibration with collected raw data
        for name,module in self.wrapped_modules.items():
            if hasattr(module, "calibrated"):
                continue
            else:
                module.mode=f"calibration_step2"
                with torch.no_grad():
                    if isinstance(module, MinMaxQuantLinear):
                        module.forward(module.raw_input.cuda())
                    elif isinstance(module, MinMaxQuantConv2d):
                        module.forward(module.raw_input.cuda())
                    elif isinstance(module, MinMaxQuantMatMul):
                        module.forward(module.raw_input[0].cuda(), module.raw_input[1].cuda())
                    torch.cuda.empty_cache()
                
        # finish calibration
        for name,module in self.wrapped_modules.items():
            module.mode='quant_forward'
        torch.cuda.empty_cache() # memory footprint cleanup
        print("calibration finished")
    
    def quant_calib(self):
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")
        if self.sequential:
            self.sequential_quant_calib()
        else:
            self.parallel_quant_calib()
        self.calibrated = True

    def batching_quant_calib(self):
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")

        print("start calibration")

        # assume wrapped modules are in order (true for dict in python>=3.5)
        q = tqdm(self.wrapped_modules.items(), desc="Brecq")
        for name, module in q:
            q.set_postfix_str(name)

            # add fp and bp hooks to current modules, which bypass calibration step 1
            # precedent modules are using quant forward
            hooks = []
            if isinstance(module, MinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, MinMaxQuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            if isinstance(module, MinMaxQuantMatMul):
                hooks.append(module.register_forward_hook(matmul_forward_hook))
            
            # feed in calibration data, and store the data
            for inp, target in self.calib_loader:
                for batch_st in range(0,self.calib_loader.batch_size,self.batch_size):
                    self.net.zero_grad()
                    inp_ = inp[batch_st:batch_st+self.batch_size].cuda()
                    self.net(inp_)
                del inp, target
                torch.cuda.empty_cache()
            
            # replace cached raw_inputs, raw_outs
            if isinstance(module, MinMaxQuantLinear):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantConv2d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantMatMul):
                module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                module.raw_out = torch.cat(module.raw_out, dim=0)
            for hook in hooks:
                hook.remove()

            # run calibration step2
            with torch.no_grad():
                if isinstance(module, MinMaxQuantLinear):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantConv2d):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantMatMul):
                    module.calibration_step2()
                torch.cuda.empty_cache()
            
            # finishing up current module calibration
            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        # finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"
        # for name, module in self.wrapped_modules.items():
        #     module.mode = "only_weights_quant_forward"
        
        print("calibration finished")

def grad_hook(module, grad_input, grad_output):
    if module.raw_grad is None:
        module.raw_grad = []
    module.raw_grad.append(grad_output[0].cpu().detach())   # that's a tuple!

def linear_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())

def conv2d_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].cpu().detach())
    module.raw_out.append(output.cpu().detach())

def matmul_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = [[],[]]
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input[0].append(input[0].cpu().detach())
    module.raw_input[1].append(input[1].cpu().detach())
    module.raw_out.append(output.cpu().detach())


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

class AttnQwT(nn.Module):
    def __init__(self, W, b, r2_score, block, linear_init=True):
        super(AttnQwT, self).__init__()
        self.norm1 = block.norm1
        self.attn = block.attn
        self.ls1 = block.ls1

        self.lora_weight = nn.Parameter(torch.zeros((W.size(0), W.size(1))))
        self.lora_bias = nn.Parameter(torch.zeros(W.size(1)))
        # pdb.set_trace()

        # if linear_init and (r2_score > 0.8448):
        if linear_init and (r2_score > 0):
            self.lora_weight.data.copy_(W)
            self.lora_bias.data.copy_(b)
        else:
            nn.init.zeros_(self.lora_weight)
            nn.init.zeros_(self.lora_bias)
    def forward(self, x, pos=None):
        out = self.ls1(self.attn(self.norm1(x), pos=pos))
        # QwT layers run in half mode
        lora_weight = self.lora_weight.half()
        out = out + (x.half() @ lora_weight).float() + self.lora_bias

        return out
class MlpQwT(nn.Module):
    def __init__(self, W, b, r2_score, block, linear_init=True):
        super(MlpQwT, self).__init__()
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.ls2 = block.ls2

        self.lora_weight = nn.Parameter(torch.zeros((W.size(0), W.size(1))))
        self.lora_bias = nn.Parameter(torch.zeros(W.size(1)))
        # pdb.set_trace()

        # if linear_init and (r2_score > 0.7645):
        if linear_init and (r2_score > 0):
            self.lora_weight.data.copy_(W)
            self.lora_bias.data.copy_(b)
        else:
            nn.init.zeros_(self.lora_weight)
            nn.init.zeros_(self.lora_bias)
    def forward(self, x, pos=None):
        out = self.ls2(self.mlp(self.norm2(x)))
        # QwT layers run in half mode
        lora_weight = self.lora_weight.half()
        out = out + (x.half() @ lora_weight).float() + self.lora_bias

        return out

def enable_quant(submodel):
    for name, module in submodel.named_modules():
        if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d) or isinstance(module, MinMaxQuantMatMul):
            module.mode = "quant_forward"

def disable_quant(submodel):
    for name, module in submodel.named_modules():
        if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d) or isinstance(module, MinMaxQuantMatMul):
            module.mode = "raw"


class HessianQuantCalibrator(QuantCalibrator):
    """
    Modularization of hessian_quant_calib

    Hessian metric needs gradients of layer outputs to weigh the loss,
    which calls for back propagation in calibration, both sequentially
    and parallelly. Despite the complexity of bp, hessian quant calibrator
    is compatible with other non-gradient quantization metrics.
    """
    def __init__(self, net, wrapped_modules, calib_loader, seq_id_map, sequential=False, batch_size=1, device='cuda', logger=None):
        super().__init__(net, wrapped_modules, calib_loader, sequential=sequential)
        self.seq_id_map = seq_id_map
        self.batch_size = batch_size
        self.device = device
        self.point = {
            "weight": 1.0,
            "gradient_loss_fn": "normal",
            "valid_range": 0.98
        }
        self.logger = logger

    def quant_calib(self):
        """
        An implementation of original hessian calibration.
        """

        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")

        print("start hessian calibration")

        # get raw_pred as target distribution 
        with torch.no_grad():
            for inp, _ in self.calib_loader:
                raw_pred = self.net(inp.cuda())
                raw_pred_softmax = F.softmax(raw_pred, dim=-1).detach()
            torch.cuda.empty_cache()

        # assume wrapped modules are in order (true for dict in python>=3.5)
        q = tqdm(self.wrapped_modules.items(), desc="Brecq")
        for name, module in q:
            q.set_postfix_str(name)

            # add fp and bp hooks to current modules, which bypass calibration step 1
            # precedent modules are using quant forward
            hooks = []
            if isinstance(module, MinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, MinMaxQuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            if isinstance(module, MinMaxQuantMatMul):
                hooks.append(module.register_forward_hook(matmul_forward_hook))
            if hasattr(module, "metric") and module.metric == "hessian":
                hooks.append(module.register_backward_hook(grad_hook))
            
            # feed in calibration data, and store the data
            for inp, target in self.calib_loader:
                for batch_st in range(0,self.calib_loader.batch_size,self.batch_size):
                    self.net.zero_grad()
                    inp_ = inp[batch_st:batch_st+self.batch_size].cuda()
                    pred = self.net(inp_)
                    loss = F.kl_div(F.log_softmax(pred, dim=-1), raw_pred_softmax[batch_st:batch_st+self.batch_size], reduction="batchmean")
                    loss.backward()
                del inp, target, pred, loss
                torch.cuda.empty_cache()
            
            # replace cached raw_inputs, raw_outs
            if isinstance(module, MinMaxQuantLinear):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantConv2d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantMatMul):
                module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if hasattr(module, "metric") and module.metric == "hessian":
                module.raw_grad = torch.cat(module.raw_grad, dim=0)
            for hook in hooks:
                hook.remove()

            # run calibration step2
            with torch.no_grad():
                if isinstance(module, MinMaxQuantLinear):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, MinMaxQuantConv2d):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, MinMaxQuantMatMul):
                    module.calibration_step2(module.raw_input[0].cuda(), module.raw_input[1].cuda())
                torch.cuda.empty_cache()
            
            # finishing up current module calibration
            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        # finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"
        
        print("hessian calibration finished")

    def batching_quant_calib(self):
        logger = self.logger
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        logger.info(f"prepare parallel calibration for {calib_layers}")

        logger.info("start hessian calibration")

        # assume wrapped modules are in order (true for dict in python>=3.5)
        # cnt = 0
        q = tqdm(self.wrapped_modules.items(), desc="Hessian")
        for name, module in q:
            # # debug only calibrate first 3 layers
            # cnt += 1
            # if cnt > 10:
            #     break
            q.set_postfix_str(name)

            hooks = []
            if isinstance(module, MinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, MinMaxQuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            if isinstance(module, MinMaxQuantMatMul):
                hooks.append(module.register_forward_hook(matmul_forward_hook))
            if hasattr(module, "metric"):
                hooks.append(module.register_backward_hook(grad_hook))
            
            # feed in calibration data, and store the data
            for seq_idx, (seq_name, ids) in enumerate(self.seq_id_map.items(), start=1):
                data = self.calib_loader.get_data(sequence_name=seq_name, ids=ids)
                filelist: list         = data['image_paths']  # [str] * N
                imgs: torch.Tensor     = data['images']       # (N, 3, H, W)
                # gt_points: np.ndarray  = data['pointclouds']  # (N, H, W, 3)
                # valid_mask: np.ndarray = data['valid_mask']   # (N, H, W)

                gt_points = torch.from_numpy(data['pointclouds']).to(self.device)
                valid_mask = torch.from_numpy(data['valid_mask']).to(self.device)

                # pdb.set_trace()
                
                self.net.zero_grad()
                # with torch.inference_mode():
                #     pred = self.net(imgs.cuda())

                pred = self.net(imgs.cuda())                  # (B, N, H, W, 3)
                # pdb.set_trace()

                if isinstance(pred, dict):
                    if "world_points" in pred:
                        pred_points = pred.get("world_points", None)
                        pred_conf = pred.get("world_points_conf", None)
                    elif "points" in pred:
                        pred_points = pred.get("points", None)
                        pred_conf = pred.get("conf", None).squeeze(-1)
                        # pred_conf = 1 + pred_conf.exp()
                    else:
                        raise ValueError("Unknown prediction keys.")
                else:
                    pred_points = pred
                    pred_conf = None

                # === 新增维度调整代码 ===
                # if pred_points is not None:
                #     # 检查并移除batch维度
                #     if pred_points.dim() == 5 and pred_points.shape[0] == 1:
                #         pred_points = pred_points.squeeze(0)
                    
                #     # 可选：对置信度进行同样处理
                #     if pred_conf is not None and pred_conf.dim() == 4 and pred_conf.shape[0] == 1:
                #         pred_conf = pred_conf.squeeze(0)
                
                # 增加batch维度
                if gt_points.dim() == 4 :
                    gt_points = gt_points.unsqueeze(0)
                
                if valid_mask is not None and valid_mask.dim() == 3 :
                    valid_mask = valid_mask.unsqueeze(0)

                assert pred_points.shape == gt_points.shape, f"Predicted points shape {pred_points.shape} does not match ground truth shape {gt_points.shape}."

                # 只用compute_point_loss，不再计算EMD
                assert gt_points is not None and pred_points is not None, f"GT/pred points is None."
                
                # 构造predictions和batch
                predictions = {'world_points': pred_points, 'world_points_conf': pred_conf if pred_conf is not None else torch.ones(pred_points.shape[:-1], device=self.device)}
                batch = {'world_points': gt_points, 'point_masks': valid_mask if valid_mask is not None else torch.ones(gt_points.shape[:-1], dtype=torch.bool, device=self.device)}

                loss_dict = compute_point_loss(predictions, batch, **self.point)
                loss = loss_dict['loss_conf_point'] + loss_dict['loss_reg_point'] + loss_dict['loss_grad_point']
                # pdb.set_trace()
                loss.backward()
                del imgs, gt_points, pred, loss, data, filelist, valid_mask, pred_points, pred_conf, predictions, batch
                # else:
                #     del imgs, pred
                torch.cuda.empty_cache()
            
            logger.info(f"finish loss_backward for {name}")
            # pdb.set_trace()
            # replace cached raw_inputs, raw_outs
            if isinstance(module, MinMaxQuantLinear):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantConv2d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantMatMul):
                module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if hasattr(module, "metric"):
                module.raw_grad = torch.cat(module.raw_grad, dim=0)
            for hook in hooks:
                hook.remove()

            # run calibration step2
            with torch.no_grad():
                if isinstance(module, MinMaxQuantLinear):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantConv2d):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantMatMul):
                    module.calibration_step2()
                torch.cuda.empty_cache()
                
            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

            # # 添加内存清理代码(看起来在step2中已经释放了内存了)
            # if hasattr(module, 'raw_input'):
            #     del module.raw_input
            # if hasattr(module, 'raw_out'):
            #     del module.raw_out
            # if hasattr(module, 'raw_grad'):
            #     del module.raw_grad
            
            logger.info(f"finish hessian calibration for {name}")
                
            # remove_hooks(hooks)
            
        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"
        print("hessian calibration finished")


    def batching_quant_calib_with_QwT(self):
        logger = self.logger
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        logger.info(f"prepare parallel calibration for {calib_layers}")
        logger.info("start hessian calibration")

        vit_blocks = self.net.aggregator.patch_embed.blocks
        frame_blocks = self.net.aggregator.frame_blocks
        global_blocks = self.net.aggregator.global_blocks

        blocks_list = list(vit_blocks) + [block for pair in zip(frame_blocks, global_blocks) for block in pair]
        
        # for idx, block in enumerate(blocks_list):
        for idx, block in tqdm(enumerate(blocks_list), total=len(blocks_list), desc=f"Processing {len(blocks_list)} blocks"):
            attn_input = None
            mlp_input = None
            attn_output = None
            mlp_output = None
            quant_output = None
            # ✅ 将 named_modules 转换为列表
            modules_list = list(block.named_modules())

            # 然后遍历列表
            for name, module in modules_list:
                hooks = []
                # pdb.set_trace()
                basename = name.split(".")[-1]
                if "qkv" == basename:
                    hooks.append(module.register_forward_hook(linear_forward_hook))
                    block.norm1.raw_input = None
                    block.norm1.raw_out = None
                    hooks.append(block.norm1.register_forward_hook(linear_forward_hook))
                    hooks.append(block.attn.proj.register_forward_hook(linear_forward_hook))
                elif "proj" == basename:
                    hooks.append(module.register_forward_hook(linear_forward_hook))
                elif "fc1" == basename:
                    hooks.append(module.register_forward_hook(linear_forward_hook))
                    block.norm2.raw_input = None
                    block.norm2.raw_out = None
                    hooks.append(block.norm2.register_forward_hook(linear_forward_hook))
                    hooks.append(block.mlp.fc2.register_forward_hook(linear_forward_hook))
                elif "fc2" == basename:
                    hooks.append(module.register_forward_hook(linear_forward_hook))
                else:
                    continue
                
                if hasattr(module, "metric"):
                    hooks.append(module.register_backward_hook(grad_hook))

                logger.info(f"Start forward hook for vit_block {idx} module {name}")

                for seq_idx, (seq_name, ids) in enumerate(self.seq_id_map.items(), start=1):
                    data = self.calib_loader.get_data(sequence_name=seq_name, ids=ids)
                    filelist: list         = data['image_paths']  # [str] * N
                    imgs: torch.Tensor     = data['images']       # (N, 3, H, W)
                    # gt_points: np.ndarray  = data['pointclouds']  # (N, H, W, 3)
                    # valid_mask: np.ndarray = data['valid_mask']   # (N, H, W)

                    gt_points = torch.from_numpy(data['pointclouds']).to(self.device)
                    valid_mask = torch.from_numpy(data['valid_mask']).to(self.device)

                    # pdb.set_trace()
                    
                    self.net.zero_grad()
                    # with torch.inference_mode():
                    #     pred = self.net(imgs.cuda())

                    pred = self.net(imgs.cuda())                  # (B, N, H, W, 3)
                    # pdb.set_trace()

                    if isinstance(pred, dict):
                        if "world_points" in pred:
                            pred_points = pred.get("world_points", None)
                            pred_conf = pred.get("world_points_conf", None)
                        elif "points" in pred:
                            pred_points = pred.get("points", None)
                            pred_conf = pred.get("conf", None).squeeze(-1)
                        else:
                            raise ValueError("Unknown prediction keys.")
                    else:
                        pred_points = pred
                        pred_conf = None

                    # === 新增维度调整代码 ===
                    # if pred_points is not None:
                    #     # 检查并移除batch维度
                    #     if pred_points.dim() == 5 and pred_points.shape[0] == 1:
                    #         pred_points = pred_points.squeeze(0)
                        
                    #     # 可选：对置信度进行同样处理
                    #     if pred_conf is not None and pred_conf.dim() == 4 and pred_conf.shape[0] == 1:
                    #         pred_conf = pred_conf.squeeze(0)
                    
                    # 增加batch维度
                    if gt_points.dim() == 4 :
                        gt_points = gt_points.unsqueeze(0)
                    
                    if valid_mask is not None and valid_mask.dim() == 3 :
                        valid_mask = valid_mask.unsqueeze(0)

                    assert pred_points.shape == gt_points.shape, f"Predicted points shape {pred_points.shape} does not match ground truth shape {gt_points.shape}."

                    # 只用compute_point_loss，不再计算EMD
                    assert gt_points is not None and pred_points is not None, f"GT/pred points is None."
                    
                    # 构造predictions和batch
                    predictions = {'world_points': pred_points, 'world_points_conf': pred_conf if pred_conf is not None else torch.ones(pred_points.shape[:-1], device=self.device)}
                    batch = {'world_points': gt_points, 'point_masks': valid_mask if valid_mask is not None else torch.ones(gt_points.shape[:-1], dtype=torch.bool, device=self.device)}
                    # predictions = {'points': pred_points, 'conf': pred_conf if pred_conf is not None else torch.ones(pred_points.shape[:-1], device=self.device)}
                    # batch = {'points': gt_points, 'point_masks': valid_mask if valid_mask is not None else torch.ones(gt_points.shape[:-1], dtype=torch.bool, device=self.device)}
                    loss_dict = compute_point_loss(predictions, batch, **self.point)
                    loss = loss_dict['loss_conf_point'] + loss_dict['loss_reg_point'] + loss_dict['loss_grad_point']
                    # pdb.set_trace()
                    loss.backward()
                    del imgs, gt_points, pred, loss, data, filelist, valid_mask, pred_points, pred_conf, predictions, batch
                    # else:
                    #     del imgs, pred
                    torch.cuda.empty_cache()
                
                logger.info(f"finish forward/backward hook for vit_block {idx} module {name}")
                    
                module.raw_input = torch.cat(module.raw_input, dim=0)
                inp = module.raw_input
                module.raw_out = torch.cat(module.raw_out, dim=0)
                if "qkv" in name:
                    attn_input = torch.cat(block.norm1.raw_input, dim=0).detach().cpu()
                    attn_output = torch.cat(block.attn.proj.raw_out, dim=0).detach().cpu()
                    del block.norm1.raw_input, block.norm1.raw_out
                    block.attn.proj.raw_input = None
                    block.attn.proj.raw_out = None
                elif "fc1" in name:
                    mlp_input = torch.cat(block.norm2.raw_input, dim=0).detach().cpu()
                    mlp_output = torch.cat(block.mlp.fc2.raw_out, dim=0).detach().cpu()
                    del block.norm2.raw_input, block.norm2.raw_out
                    block.mlp.fc2.raw_input = None
                    block.mlp.fc2.raw_out = None
                if hasattr(module, "metric"):
                    module.raw_grad = torch.cat(module.raw_grad, dim=0)
                for hook in hooks:
                    hook.remove()
                # pdb.set_trace()

                with torch.no_grad():
                    module.calibration_step2()
                torch.cuda.empty_cache()
                module.mode = "quant_forward"
                if idx < 24:
                    logger.info(f"finish hessian calibration for vit_block {idx} module {name} (vit stage)")
                else:
                    logger.info(f"finish hessian calibration for vggt_block {idx-24} module {name} (frame/global stage)")

                if "proj" in name:
                    # replace attention module with AttnQwT
                    quant_output = module(inp.cuda()).detach().cpu()
                    target = attn_output - quant_output
                    W, b, r2_score = linear_regression(attn_input.cuda(), target.cuda())
                    logger.info(f"AttnQwT r2_score: {r2_score.item():.6f} for block {idx}")
                    attn_qwt = AttnQwT(W, b, r2_score, block, linear_init=True).to(self.device)
                    block.attn_QwT = attn_qwt
                    del attn_input, attn_output, quant_output, target, inp
                elif "fc2" in name:
                    # replace mlp module with MlpQwT
                    quant_output = module(inp.cuda()).detach().cpu()
                    target = mlp_output - quant_output
                    W, b, r2_score = linear_regression(mlp_input.cuda(), target.cuda())
                    logger.info(f"MlpQwT r2_score: {r2_score.item():.6f} for block {idx}")
                    mlp_qwt = MlpQwT(W, b, r2_score, block, linear_init=True).to(self.device)
                    block.mlp_QwT = mlp_qwt
                    block.module_QwT = True
                    del mlp_input, mlp_output, quant_output, target, inp
        
        logger.info("finish hessian calibration for all blocks")
           
    # def batching_quant_calib_with_QwT_new(self):

    #     def attn_module(block, x, pos=None):
    #         return block.ls1(block.attn(block.norm1(x), pos=pos))

    #     def mlp_module(block, x):
    #         return block.ls2(block.mlp(block.norm2(x)))
        
    #     logger = self.logger
    #     calib_layers=[]
    #     for name,module in self.wrapped_modules.items():
    #         calib_layers.append(name)
    #     logger.info(f"prepare parallel calibration for {calib_layers}")
    #     logger.info("start hessian calibration")

    #     vit_blocks = self.net.aggregator.patch_embed.blocks
    #     frame_blocks = self.net.aggregator.frame_blocks
    #     global_blocks = self.net.aggregator.global_blocks

    #     # blocks_list = list(vit_blocks) + [block for pair in zip(frame_blocks, global_blocks) for block in pair]
        
    #     # for idx, block in enumerate(blocks_list):
    #     with torch.no_grad():
    #         inputs = []
    #         for seq_idx, (seq_name, ids) in enumerate(self.seq_id_map.items(), start=1):
    #             batch = self.calib_loader.get_data(sequence_name=seq_name, ids=ids)
    #             inputs.append(batch['images'])
    #         inputs = torch.stack(inputs, dim=0).to('cuda')  # shape: [num_samples, S, 3, H, W]
    #     cur_inp = self.net.aggregator.forward_before_patch_embed(inputs)
    #     masks = None
    #     cur_inp = self.net.prepare_tokens_with_masks(cur_inp, masks)
    #     cur_inp = cur_inp.cuda()
    #     raw_inp = cur_inp.cuda()

    #     for idx, block in tqdm(enumerate(vit_blocks), total=len(vit_blocks), desc=f"Processing {len(vit_blocks)} vit_blocks"):
    #         # ===============================
    #         # 1. 获取 attn module 的输入与输出
    #         # ===============================
    #         raw_out = attn_module(block, raw_inp).detach().cpu()
    #         cur_out = attn_module(block, cur_inp).detach().cpu()

    #         # 补偿目标：FP - Q
    #         target = fp_out - quant_out         # Δ
            
    #         similarity =  compute_cosine_similarity_flat(fp_out, quant_out, dim=1)
    #         logging.info(f"cosine similarity for Vit block {i} attn module: {similarity.item():.6f}")
    #         # W, b, r2_score = ridge_regression(
    #         W, b, r2_score = linear_regression(
    #             cur_inp.cuda(), 
    #             target.cuda()
    #         )
    #         logging.info(f"R2 score for Vit block {i} attn module: {r2_score.item():.6f}")
    #         if random.random() < p:
    #             r2_score = -3.0
    #             logging.info(f"close Vit block {i} attn module QwT")
    #         comp = AttnQwT(
    #             W=W, 
    #             b=b,
    #             # r2_score=-3.0,   # 注意力补偿效果不佳，强制初始化为0
    #             r2_score=r2_score,
    #             block=block,             # 原模块
    #             linear_init=True
    #         )
    #         comp.cuda()
    #         next_inp = cur_inp + comp(cur_inp).cuda()
    #         net.blocks[i].attn_QwT = comp
    #         cur_inp = next_inp.cuda()
    #         # ==============================
    #         # 获得 mlp module 的输入与输出
    #         # ==============================
    #         disable_quant(block)
    #         # next_inp = cur_inp + mlp_module(block, cur_inp).cuda()
    #         fp_out = mlp_module(block, cur_inp).detach().cpu()
    #         enable_quant(block)
    #         quant_out = mlp_module(block, cur_inp).detach().cpu()

    #         # 补偿目标：FP - Q
    #         target = fp_out - quant_out         # Δ
    #         similarity =  compute_cosine_similarity_flat(fp_out, quant_out, dim=1)
    #         logging.info(f"cosine similarity for Vit block {i} mlp module: {similarity.item():.6f}")
    #         # W, b, r2_score = ridge_regression(
    #         W, b, r2_score = linear_regression(
    #             cur_inp.cuda(), 
    #             target.cuda()
    #         )
            
    #         logging.info(f"R2 score for Vit block {i} mlp module: {r2_score.item():.6f}")
    #         if random.random() < p:
    #             r2_score = -3.0
    #             logging.info(f"close Vit block {i} mlp module QwT")
            
    #         comp = MlpQwT(
    #             W=W, 
    #             b=b,
    #             # r2_score=-3.0,   # mlp补偿效果不佳，强制初始化为0
    #             r2_score=r2_score,
    #             block=block,             # 原模块
    #             linear_init=True
    #         )
    #         comp.cuda()
    #         next_inp = cur_inp + comp(cur_inp).cuda()
    #         net.blocks[i].mlp_QwT = comp
    #         cur_inp = next_inp.cuda()
    #         net.blocks[i].module_QwT = True


        
    #     logger.info("finish hessian calibration for all blocks")
           
