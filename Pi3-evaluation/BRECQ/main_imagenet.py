import torch
import torch.nn as nn
import argparse
import os
import random
import numpy as np
import os.path as osp
import time
import hubconf
import cv2
import torchvision.transforms as tvf
import torch.nn.functional as F
import open3d as o3d
from quant import *
from data.imagenet import build_imagenet_data
from typing import Optional, Union, List
from PIL import Image, ImageFile
import pdb
import time
from importlib import reload,import_module
import sys
to_tensor = tvf.ToTensor()

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from PTQ.vggt.models.vggt import VGGT
from PTQ.vggt.utils.load_fn import load_and_preprocess_images
from datasets.utils.cropping import resize_image, resize_image_depth_and_intrinsic
from utils.geometry import unproject_depth_map_to_point_map
# from pi3.models.pi3 import Pi3
from utils.interfaces import infer_mv_pointclouds
from mv_recon.utils import umeyama, accuracy, completion
from utils.messages import set_default_arg, write_csv
from utils.vis_utils import save_image_grid_auto



def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



def get_train_samples(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]



def save_model_structure(model: nn.Module, input_shape=(3, 224, 224), save_path="vggt_structure.txt"):
    """
    打印并保存模型结构到TXT文件
    """
    with open(save_path, "w") as f:
        # 先写模型整体结构
        f.write("===== VGGT Model Structure =====\n")
        f.write(str(model))
        f.write("\n\n")

        # 再打印参数统计（可选）
        f.write("===== Parameter Summary =====\n")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total parameters: {total_params}\n")
        f.write(f"Trainable parameters: {trainable_params}\n\n")

        # # 如果你有一个标准输入尺寸，也可以用 dummy input 查看结构
        # try:
        #     dummy_input = torch.randn(1, *input_shape)
        #     f.write("===== Layer-wise output shapes =====\n")
        #     summary(model, input_shape, device="cpu", verbose=0, file=f)
        # except Exception as e:
        #     f.write(f"[Warning] summary() failed: {e}\n")

    print(f"✅ 模型结构已保存到: {save_path}")

def load_cam_mvsnet(words, interval_scale=1):
    """read camera txt file"""
    cam = np.zeros((2, 4, 4))
    # words = file.read().split()
    words = words.split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = 192
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    extrinsic = cam[0].astype(np.float32)
    intrinsic = cam[1].astype(np.float32)

    return intrinsic, extrinsic

def test_dtu(model, aggregator, test_data):
    output_dir = "/root/Pi3-evaluation/outputs/bercq"
    output_root = "/root/Pi3-evaluation/outputs/bercq/DTU"
    with torch.no_grad():
        for data in test_data:
            filelist: list         = data['image_paths']  # [str] * N
            seq_name               = data['seq_id']
            images: torch.Tensor   = data['images']       # (N, 3, H, W)
            gt_pts: np.ndarray     = data['pointclouds']  # (N, H, W, 3)
            valid_mask: np.ndarray = data['valid_mask']   # (N, H, W)

            data_h, data_w         = images.shape[-2:]
            images = images.unsqueeze(0)  # (1, N, 3, H, W)
            images = images.cuda()
            predictions = {}
            aggregated_tokens_list, patch_start_idx = aggregator(images)
            pts3d, pts3d_conf = model.point_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
            predictions["world_points"] = pts3d
            predictions["world_points_conf"] = pts3d_conf
            pts3d = predictions["world_points"][0]  # (N, H', W', 3)
            images = images[0]  # (N, 3, H, W)
            global_points = F.interpolate(
                pts3d.permute(0, 3, 1, 2), [data_h, data_w],
                mode="bilinear", align_corners=False, antialias=True
            ).permute(0, 2, 3, 1)  # align to gt

            pred_pts = global_points.cpu().numpy()
            assert pred_pts.shape == gt_pts.shape, f"Predicted points shape {pred_pts.shape} does not match ground truth shape {gt_pts.shape}."

            # 4. save input images
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
            threshold = 100
            
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

            # 10. compute metrics
            acc, acc_med, nc1, nc1_med = accuracy(
                pcd_gt.points, pcd.points, gt_normal, pred_normal
            )
            comp, comp_med, nc2, nc2_med = completion(
                pcd_gt.points, pcd.points, gt_normal, pred_normal
            )
            print(f"Scan {seq_name} - Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2}")
            # logger.info(
            #     f"[{dataset_name} {seq_idx}/{len(dataset.sequence_list)}] Seq: {seq_name}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
            # )

            # 1.2 ready for output directory & metrics
            os.makedirs(output_root, exist_ok=True)
            if osp.exists(osp.join(output_root, "_all_samples.csv")):
                os.remove(osp.join(output_root, "_all_samples.csv"))  # remove old csv file
            all_data_dict = {
                "Acc-mean":  0.0,  "Acc-med":  0.0,
                "Comp-mean": 0.0,  "Comp-med": 0.0,
                "NC-mean":   0.0,  "NC-med":   0.0,
                "NC1-mean":  0.0,  "NC1-med":  0.0,
                "NC2-mean":  0.0,  "NC2-med":  0.0,
            }
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
        num_samples = len(test_data)
        metric_dict = {
            metric: value / num_samples
            for metric, value in all_data_dict.items()
            if metric != "model"
        }

        statistics_file = osp.join(output_dir, f"DTU-metric")  # + ".csv"
        statistics_file += ".csv"
        write_csv(statistics_file, metric_dict)
    
    del model
    torch.cuda.empty_cache()
    print(f"Finished evaluating BERCQ on DTU datasets.")
    # logger.info(f"Finished evaluating Pi3 on all datasets.")



def get_data(
        data_paths: List[int],
        ids: Union[List[int], np.ndarray, None] = None,
    ):
    batchs = []
    for seq_index, data_dir in enumerate(data_paths):
        image_path = osp.join(data_dir, "images")
        depth_path = osp.join(data_dir, "depths")
        mask_path = osp.join(data_dir, "binary_masks")
        cam_path = osp.join(data_dir, "cams")

        image_paths: list      = [""] * len(ids)
        images: list           = [0]  * len(ids)
        depths: list           = [0]  * len(ids)
        extrinsics: np.ndarray = np.zeros((len(ids), 3, 4))
        intrinsics: np.ndarray = np.zeros((len(ids), 3, 3))

        for id_index, id in enumerate(ids):
            impath = osp.join(image_path, f"{id:08d}.jpg")
            depthpath = osp.join(depth_path, f"{id:08d}.npy")
            campath = osp.join(cam_path, f"{id:08d}_cam.txt")
            maskpath = osp.join(mask_path, f"{id:08d}.png")

            rgb_image: Image.Image = Image.open(impath)
            depthmap: np.ndarray   = np.load(depthpath)
            rgb_image: Image.Image = resize_image(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0)

            mask = cv2.imread(maskpath, cv2.IMREAD_UNCHANGED) / 255.0
            mask = mask.astype(np.float32)

            mask[mask > 0.5] = 1.0
            mask[mask < 0.5] = 0.0

            mask = cv2.resize(
                mask,
                (depthmap.shape[1], depthmap.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            kernel = np.ones((10, 10), np.uint8)  # Define the erosion kernel
            mask = cv2.erode(mask, kernel, iterations=1)
            depthmap = depthmap * mask

            cur_intrinsics, extrinsic = load_cam_mvsnet(open(campath, "r").read())
            intrinsic = cur_intrinsics[:3, :3]

            rgb_image, depthmap, intrinsic = resize_image_depth_and_intrinsic(
                image=rgb_image,
                depth_map=depthmap,
                intrinsic=intrinsic,
                output_width=518, # finally width = 518, height = 388
            )

            image_paths[id_index] = impath
            images[id_index]      = to_tensor(rgb_image)
            depths[id_index]      = depthmap
            intrinsics[id_index]  = intrinsic
            extrinsics[id_index]  = extrinsic[:3, :]

        depths = np.array(depths)  # (S, H, W)
        pointclouds = unproject_depth_map_to_point_map(
            depth_map=depths[..., None],
            intrinsics_cam=intrinsics,
            extrinsics_cam=extrinsics
        )

        batch = {"seq_id": osp.basename(data_dir), "seq_len": len(image_path), "ind": torch.tensor(ids)}
        batch['image_paths'] = image_paths  # list of str
        batch['images']      = torch.stack(images, dim=0)
        batch['pointclouds'] = pointclouds  # in numpy
        batch['valid_mask']  = depths > 1e-4
        # batch["extrs"] = extrinsics
        # batch["intrs"] = intrinsics
        # batch["w"] = metadata["w"]
        # batch["h"] = metadata["h"]
        batchs.append(batch)
    return batchs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--arch', default='resnet18', type=str, help='dataset name',
                        choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet', 'vggt'])
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader')
    parser.add_argument('--data_path', default='', type=str, help='path to ImageNet data', required=True)

    # quantization parameters
    parser.add_argument('--n_bits_w', default=4, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', action='store_true', help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=4, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--act_quant', action='store_true', help='apply activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')
    parser.add_argument('--test_before_calibration', action='store_true')

    # weight calibration parameters
    parser.add_argument('--num_samples', default=1024, type=int, help='size of the calibration dataset')
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--sym', action='store_true', help='symmetric reconstruction, not recommended')
    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')
    parser.add_argument('--step', default=20, type=int, help='record snn output per step')

    # activation calibration parameters
    parser.add_argument('--iters_a', default=5000, type=int, help='number of iteration for LSQ')
    parser.add_argument('--lr', default=4e-4, type=float, help='learning rate for LSQ')
    parser.add_argument('--p', default=2.4, type=float, help='L_p norm minimization for LSQ')

    args = parser.parse_args()


    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_all(args.seed)

    # load model
    print("Initializing and loading VGGT model...")
    pretrained_model_name_or_path = "/root/autodl-tmp/hf_hub/models--facebook--VGGT-1B"
    # pretrained_model_name_or_path = "facebook/VGGT-1B"
    model = VGGT.from_pretrained(pretrained_model_name_or_path).to(device).eval()
    print("Model loaded.")
    # build quantization parameters
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': 'mse'}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.act_quant}
    print("Quantization parameters set.")
    aggregator = QuantModel(model=model.aggregator, weight_quant_params=wq_params, act_quant_params=aq_params)
    print("QuantModel created.")
    aggregator.cuda()
    aggregator.eval()

    print("VGGT Model Structure:")
    print("=====================")
    save_model_structure(aggregator, input_shape=(3, 224, 224), save_path="aggregator_structure.txt")
    print("=====================")

    # if not args.disable_8bit_head_stem:
    #     print('Setting the first and the last layer to 8-bit')
    #     model.set_first_last_layer_to_8bit()

    # todo: load data

    # 获取目录下所有文件并过滤出图片文件
    test_path = "/root/autodl-tmp/data/dtu"
    test_numbers = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
    data_paths = [os.path.join(test_path, f'scan' + str(num)) for num in test_numbers]
    image_ids = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    # seq_numbers = [7, 8 ,82, 120]
    seq_numbers = [7]
    cali_data = []
    test_data = get_data(data_paths = data_paths, ids=image_ids)
    for target_dir in [os.path.join(args.data_path, f'scan' + str(num), 'images') for num in seq_numbers]:
        image_names = [
            os.path.join(target_dir, f) 
            for f in os.listdir(target_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
        ]

        # 按文件名排序（可选）
        image_names.sort()  
        images = load_and_preprocess_images(image_names).to(device)
        cali_data.append(images)
        # pdb.set_trace()
    cali_data = torch.stack(cali_data, dim=0)
    print(f'Calibration data shape: {cali_data.shape}')  # 输出形状以确认


    # Initialize weight quantization parameters
    aggregator.set_quant_state(True, False)
    strat_time = time.time()
    print(f'{strat_time},Initializing weight quantization parameters...')
    # aggregated_tokens_list, patch_start_idx = aggregator(cali_data.to(device))
    end_time = time.time()
    print(f'Weight quantization initialization time: {end_time - strat_time} seconds.')


    # Kwargs for weight rounding calibration
    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse')

    def recon_model(model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            # pdb.set_trace()
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(model, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    block_reconstruction(model, module, **kwargs)
            # --- 新增：处理容器类型 ---
            elif isinstance(module, (nn.ModuleList, nn.Sequential)):
                # pdb.set_trace()
                if isinstance(module[0], BaseQuantBlock):
                    for idx, sub_child in enumerate(module):
                        if sub_child.ignore_reconstruction is True:
                            print('Ignore reconstruction of block {}'.format(name))
                            continue
                        else:
                            print('Reconstruction for block {}'.format(name))
                            block_reconstruction(model, sub_child, **kwargs)

            else:
                recon_model(module)

    # Start calibration
    strat_time = time.time()
    print(f'{strat_time},Starting quantization calibration...')
    recon_model(aggregator)
    end_time = time.time()
    print(f'Quantization calibration time: {end_time - strat_time} seconds.')
    aggregator.set_quant_state(weight_quant=True, act_quant=False)
    test_dtu(model, aggregator, test_data)
    # print('Weight quantization accuracy: {}'.format(validate_model(test_loader, qnn)))

    if args.act_quant:
        # Initialize activation quantization parameters
        aggregator.set_quant_state(True, True)
        with torch.no_grad():
            _ = aggregator(cali_data.to(device))
        # Disable output quantization because network output
        # does not get involved in further computation
        aggregator.disable_network_output_quantization()
        # Kwargs for activation rounding calibration
        kwargs = dict(cali_data=cali_data, iters=args.iters_a, act_quant=True, opt_mode='mse', lr=args.lr, p=args.p)
        strat_time = time.time()
        print(f'{strat_time},Starting activation quantization calibration...')
        recon_model(aggregator)
        end_time = time.time()
        print(f'Activation quantization calibration time: {end_time - strat_time} seconds.')
        aggregator.set_quant_state(weight_quant=True, act_quant=True)
        test_dtu(model, aggregator, test_data)
        # print('Full quantization (W{}A{}) accuracy: {}'.format(args.n_bits_w, args.n_bits_a,
        #                                                        validate_model(test_loader, qnn)))

# CUDA_VISIBLE_DEVICES=5 python BRECQ/main_imagenet.py \
#     --data_path /root/Pi3-evaluation/data/calib --arch vggt \
#     --n_bits_w 2 --channel_wise --n_bits_a 4 --act_quant

# python BRECQ/main_imagenet.py \
#     --data_path /root/Pi3-evaluation/data/calib --arch vggt \
#     --n_bits_w 4 --channel_wise --n_bits_a 6 --act_quant
