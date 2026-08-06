import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Union
from PIL import Image
import glob
import cv2
import open3d as o3d
import torchvision.transforms as tvf

from datasets.utils.cropping import resize_image  # ✅ 仅调用，不改动函数

to_tensor = tvf.ToTensor()


class DTUTrain(Dataset):
    """
    轻量版 DTU 数据集（保持与原 DTU 接口完全一致）：
    - 移除 depth / cam 参数
    - mask 全 True
    - 直接读取点云 GT（.ply）
    """

    def __init__(
        self,
        DTU_DIR: str = "/root/autodl-tmp/data/dtu_train",
        split: str = "train",
        load_img_size: int = 518,
        cache_file: str = "data/dataset_cache/dtu_mv_recon_cache.npy",
    ):
        self.DTU_DIR = DTU_DIR
        self.split = split
        self.load_img_size = load_img_size
        self.cache_file = cache_file

        print(f"[DTUTrain] Loading dataset from {DTU_DIR} (split={split})")

        # 收集所有 scan
        scan_dirs = sorted(glob.glob(osp.join(DTU_DIR, "scan*_train")))
        self.sequence_list = []
        self.metadata = {}

        for scan_dir in scan_dirs:
            scan_name = osp.basename(scan_dir).replace("_train", "")
            scan_id = scan_name.replace("scan", "").zfill(3)
            gt_path = osp.join(DTU_DIR, f"stl{scan_id}_total.ply")
            if not osp.exists(gt_path):
                continue

            img_paths = sorted(glob.glob(osp.join(scan_dir, "*.jpg")))
            if len(img_paths) == 0:
                continue

            self.sequence_list.append(scan_name)
            self.metadata[scan_name] = len(img_paths)

        print(f"[DTUTrain] Found {len(self.sequence_list)} sequences.")

    def __len__(self):
        return len(self.sequence_list)

    def get_seq_framenum(self, index=None, sequence_name=None):
        if sequence_name is None:
            sequence_name = self.sequence_list[index]
        return self.metadata[sequence_name]

    def __getitem__(self, idx_N):
        index, n_per_seq = idx_N
        sequence_name = self.sequence_list[index]
        num_imgs = self.metadata[sequence_name]
        ids = np.random.choice(num_imgs, n_per_seq, replace=False)
        return self.get_data(index=index, ids=ids)

    # ============================================================
    # ✅ 完全仿照原 get_data 逻辑，只去掉 depth/cam，mask全True
    # ============================================================
    def get_data(
        self,
        index: Optional[int] = None,
        sequence_name: Optional[str] = None,
        ids: Union[List[int], np.ndarray, None] = None,
    ):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name: str = self.sequence_list[index]
        seq_len: int = self.metadata[sequence_name]

        if ids is None:
            ids = np.arange(seq_len).tolist()
        elif isinstance(ids, np.ndarray):
            assert ids.ndim == 1, f"ids should be a 1D array, but got {ids.ndim}D"
            ids = ids.tolist()

        image_path = osp.join(self.DTU_DIR, f"{sequence_name}_train")
        gt_path = osp.join(self.DTU_DIR, f"stl{sequence_name.replace('scan', '').zfill(3)}_total.ply")

        image_paths: list = [""] * len(ids)
        images: list = [0] * len(ids)
        masks: list = [0] * len(ids)

        # =======================
        # 读取图像
        # =======================
        for id_index, id in enumerate(ids):
            img_list = sorted(glob.glob(osp.join(image_path, "*.jpg")))
            impath = img_list[id % len(img_list)]
            rgb_image: Image.Image = Image.open(impath).convert("RGB")

            # ✅ 使用 resize_image，但提前计算出合法分辨率 (W,H)，高度是14的倍数
            w0, h0 = rgb_image.size
            output_width = self.load_img_size
            output_height = round(h0 * (output_width / w0) / 14) * 14  # 高度对齐到14的倍数
            rgb_image = resize_image(rgb_image, (output_width, output_height))

            image_paths[id_index] = impath
            images[id_index] = to_tensor(rgb_image)

            # mask 全为 True
            masks[id_index] = np.ones(
                (rgb_image.height, rgb_image.width), dtype=bool
            )

        # =======================
        # 读取点云GT
        # =======================
        pcd = o3d.io.read_point_cloud(gt_path)
        points = np.asarray(pcd.points, dtype=np.float32)

        # =======================
        # 构造batch（字段顺序与原版保持一致）
        # =======================
        batch = {"seq_id": sequence_name, "seq_len": seq_len, "ind": torch.tensor(ids)}
        batch["image_paths"] = image_paths
        batch["images"] = torch.stack(images, dim=0)  # (S,3,H,W)
        batch["pointclouds"] = points                 # (N,3)
        batch["valid_mask"] = np.stack(masks, axis=0)  # (S,H,W)

        return batch
