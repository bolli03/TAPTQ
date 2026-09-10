import json
import os
import cv2
import numpy as np
import os.path as osp
from PIL import Image
from collections import deque
from eval.base import BaseStereoViewDataset
import eval.dataset_utils.cropping as cropping
from vggt.utils.eval_utils import imread_cv2, shuffle_deque


class SevenScenes(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=1,
        num_frames=5,
        min_thresh=10,
        max_thresh=100,
        test_id=None,
        full_video=False,
        tuple_list=None,
        seq_id=None,
        rebuttal=False,
        shuffle_seed=-1,
        kf_every=1,
        *args,
        ROOT,
        **kwargs,
    ):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.seq_id = seq_id
        self.rebuttal = rebuttal
        self.shuffle_seed = shuffle_seed

        # load all scenes
        self.load_all_tuples(tuple_list)
        self.load_all_scenes(ROOT)

    def __len__(self):
        if self.tuple_list is not None:
            return len(self.tuple_list)
        return len(self.scene_list) * self.num_seq

    def load_all_tuples(self, tuple_list):
        if tuple_list is not None:
            self.tuple_list = tuple_list
            # with open(tuple_path) as f:
            #     self.tuple_list = f.read().splitlines()

        else:
            self.tuple_list = None

    def load_all_scenes(self, base_dir):

        if self.tuple_list is not None:
            # Use pre-defined simplerecon scene_ids
            self.scene_list = [
                "stairs/seq-06",
                "stairs/seq-02",
                "pumpkin/seq-06",
                "chess/seq-01",
                "heads/seq-02",
                "fire/seq-02",
                "office/seq-03",
                "pumpkin/seq-03",
                "redkitchen/seq-07",
                "chess/seq-02",
                "office/seq-01",
                "redkitchen/seq-01",
                "fire/seq-01",
            ]
            print(f"Found {len(self.scene_list)} sequences in split {self.split}")
            return

        scenes = os.listdir(base_dir)

        file_split = {"train": "TrainSplit.txt", "test": "TestSplit.txt"}[self.split]

        self.scene_list = []
        for scene in scenes:
            if self.test_id is not None and scene != self.test_id:
                continue
            # read file split
            with open(osp.join(base_dir, scene, file_split)) as f:
                seq_ids = f.read().splitlines()

                for seq_id in seq_ids:
                    # seq is string, take the int part and make it 01, 02, 03
                    # seq_id = 'seq-{:2d}'.format(int(seq_id))
                    num_part = "".join(filter(str.isdigit, seq_id))
                    seq_id = f"seq-{num_part.zfill(2)}"
                    if self.seq_id is not None and seq_id != self.seq_id:
                        continue
                    self.scene_list.append(f"{scene}/{seq_id}")

        print(f"Found {len(self.scene_list)} sequences in split {self.split}")

    def _get_views(self, idx, resolution, rng):

        if self.tuple_list is not None:
            line = self.tuple_list[idx].split(" ")
            scene_id = line[0]
            img_idxs = line[1:]

        else:
            scene_id = self.scene_list[idx // self.num_seq]
            seq_id = idx % self.num_seq

            data_path = osp.join(self.ROOT, scene_id)
            num_files = len([name for name in os.listdir(data_path) if "color" in name])
            img_idxs = [f"{i:06d}" for i in range(num_files)]
            img_idxs = img_idxs[:: self.kf_every]

        # Intrinsics used in SimpleRecon
        fx, fy, cx, cy = 525, 525, 320, 240
        intrinsics_ = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        views = []
        imgs_idxs = deque(img_idxs)
        if self.shuffle_seed >= 0:
            imgs_idxs = shuffle_deque(imgs_idxs)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()
            impath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.color.png")
            depthpath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.depth.proj.png")
            posepath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.pose.txt")

            rgb_image = imread_cv2(impath)

            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            depthmap[depthmap == 65535] = 0
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0

            depthmap[depthmap > 10] = 0
            depthmap[depthmap < 1e-3] = 0

            camera_pose = np.loadtxt(posepath).astype(np.float32)

            if resolution != (224, 224) or self.rebuttal:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, (512, 384), rng=rng, info=impath
                )
                W, H = rgb_image.size
                cx = W // 2
                cy = H // 2
                l, t = cx - 112, cy - 112
                r, b = cx + 112, cy + 112
                crop_bbox = (l, t, r, b)
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, crop_bbox
                )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="7scenes",
                    label=osp.join(scene_id, im_idx),
                    instance=impath,
                )
            )
        return views


class DTU(BaseStereoViewDataset):
    """DTU test split with the mv-recon stride-kf5 protocol."""

    TEST_SCANS = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]

    def __init__(self, *, ROOT, kf_every=5, **kwargs):
        self.ROOT = ROOT
        self.kf_every = kf_every
        super().__init__(**kwargs)
        self.scene_list = [f"scan{scan}" for scan in self.TEST_SCANS]

    def __len__(self):
        return len(self.scene_list)

    @staticmethod
    def _load_cam(path):
        words = open(path, "r").read().split()
        extrinsic = np.array([[float(words[4 * i + j + 1]) for j in range(4)] for i in range(4)], dtype=np.float32)
        intrinsic = np.array([[float(words[3 * i + j + 18]) for j in range(3)] for i in range(3)], dtype=np.float32)
        return intrinsic, extrinsic

    def _get_views(self, idx, resolution, rng):
        scene_id = self.scene_list[idx]
        scene_root = osp.join(self.ROOT, scene_id)
        image_root = osp.join(scene_root, "images")
        depth_root = osp.join(scene_root, "depths")
        mask_root = osp.join(scene_root, "binary_masks")
        if not osp.isdir(mask_root):
            mask_root = osp.join(scene_root, "masks")
        cam_root = osp.join(scene_root, "cams")
        frame_ids = list(range(0, len(os.listdir(image_root)), self.kf_every))
        views = []
        for frame_id in frame_ids:
            image_path = osp.join(image_root, f"{frame_id:08d}.jpg")
            depth_path = osp.join(depth_root, f"{frame_id:08d}.npy")
            mask_path = osp.join(mask_root, f"{frame_id:08d}.png")
            cam_path = osp.join(cam_root, f"{frame_id:08d}_cam.txt")
            image = Image.open(image_path).convert("RGB")
            depthmap = np.nan_to_num(np.load(depth_path).astype(np.float32), nan=0.0)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
            mask = (cv2.erode((mask > 0.5).astype(np.uint8), np.ones((10, 10), np.uint8), iterations=1) > 0)
            mask = cv2.resize(mask.astype(np.uint8), (depthmap.shape[1], depthmap.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            depthmap *= mask
            if image.size != depthmap.shape[::-1]:
                image = image.resize(depthmap.shape[::-1], Image.Resampling.LANCZOS)
            intrinsic, extrinsic = self._load_cam(cam_path)
            image, depthmap, intrinsic = self._crop_resize_if_necessary(
                image, depthmap, intrinsic, resolution, rng=rng, info=image_path
            )
            camera_pose = np.linalg.inv(extrinsic).astype(np.float32)
            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsic,
                dataset="DTU",
                label=osp.join(scene_id, f"{frame_id:08d}"),
                instance=image_path,
            ))
        return views


class ETH3D(BaseStereoViewDataset):
    """Pi3-compatible ETH3D sequences with frozen kf=5 frame indices."""

    def __init__(self, *, ROOT, seq_id_map, **kwargs):
        self.ROOT = ROOT
        with open(seq_id_map) as f:
            self.seq_id_map = json.load(f)
        super().__init__(**kwargs)
        self.scene_list = sorted(self.seq_id_map.keys())

    def __len__(self):
        return len(self.scene_list)

    def _get_views(self, idx, resolution, rng):
        seq = self.scene_list[idx]
        image_root = osp.join(self.ROOT, seq, "images", "custom_undistorted")
        image_names = sorted(name for name in os.listdir(image_root) if name.endswith(".JPG"))
        views = []
        for frame_id in self.seq_id_map[seq]:
            name = image_names[frame_id]
            image = Image.open(osp.join(image_root, name)).convert("RGB")
            width, height = image.size
            depth_path = osp.join(self.ROOT, seq, "ground_truth_depth", "custom_undistorted", name)
            depthmap = np.fromfile(depth_path, dtype=np.float32).reshape(height, width)
            camera = np.load(osp.join(self.ROOT, seq, "custom_undistorted_cam", name.replace(".JPG", ".npz")))
            intrinsic = camera["intrinsics"].astype(np.float32)
            camera_pose = np.linalg.inv(camera["extrinsics"]).astype(np.float32)
            image, depthmap, intrinsic = self._crop_resize_if_necessary(
                image, depthmap, intrinsic, resolution, rng=rng, info=depth_path
            )
            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsic,
                dataset="eth3d",
                label=osp.join(seq, name),
                instance=depth_path,
            ))
        return views


class NRGBD(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=1,
        num_frames=5,
        min_thresh=10,
        max_thresh=100,
        test_id=None,
        full_video=False,
        tuple_list=None,
        seq_id=None,
        rebuttal=False,
        shuffle_seed=-1,
        kf_every=1,
        *args,
        ROOT,
        **kwargs,
    ):

        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.seq_id = seq_id
        self.rebuttal = rebuttal
        self.shuffle_seed = shuffle_seed

        # load all scenes
        self.load_all_tuples(tuple_list)
        self.load_all_scenes(ROOT)

    def __len__(self):
        if self.tuple_list is not None:
            return len(self.tuple_list)
        return len(self.scene_list) * self.num_seq

    def load_all_tuples(self, tuple_list):
        if tuple_list is not None:
            self.tuple_list = tuple_list
            # with open(tuple_path) as f:
            #     self.tuple_list = f.read().splitlines()

        else:
            self.tuple_list = None

    def load_all_scenes(self, base_dir):

        scenes = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]

        if self.test_id is not None:
            self.scene_list = [self.test_id]

        else:
            self.scene_list = scenes

        print(f"Found {len(self.scene_list)} sequences in split {self.split}")

    def load_poses(self, path):
        file = open(path, "r")
        lines = file.readlines()
        file.close()
        poses = []
        valid = []
        lines_per_matrix = 4
        for i in range(0, len(lines), lines_per_matrix):
            if "nan" in lines[i]:
                valid.append(False)
                poses.append(np.eye(4, 4, dtype=np.float32).tolist())
            else:
                valid.append(True)
                pose_floats = [
                    [float(x) for x in line.split()]
                    for line in lines[i : i + lines_per_matrix]
                ]
                poses.append(pose_floats)

        return np.array(poses, dtype=np.float32), valid

    def _get_views(self, idx, resolution, rng):

        if self.tuple_list is not None:
            line = self.tuple_list[idx].split(" ")
            scene_id = line[0]
            img_idxs = line[1:]

        else:
            scene_id = self.scene_list[idx // self.num_seq]

            num_files = len(os.listdir(os.path.join(self.ROOT, scene_id, "images")))
            img_idxs = [f"{i}" for i in range(num_files)]
            img_idxs = img_idxs[:: min(self.kf_every, len(img_idxs) // 2)]

        fx, fy, cx, cy = 554.2562584220408, 554.2562584220408, 320, 240
        intrinsics_ = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        posepath = osp.join(self.ROOT, scene_id, f"poses.txt")
        camera_poses, valids = self.load_poses(posepath)

        imgs_idxs = deque(img_idxs)
        if self.shuffle_seed >= 0:
            imgs_idxs = shuffle_deque(imgs_idxs)
        views = []

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()

            impath = osp.join(self.ROOT, scene_id, "images", f"img{im_idx}.png")
            depthpath = osp.join(self.ROOT, scene_id, "depth", f"depth{im_idx}.png")

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
            depthmap[depthmap > 10] = 0
            depthmap[depthmap < 1e-3] = 0

            rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            camera_pose = camera_poses[int(im_idx)]
            # gl to cv
            camera_pose[:, 1:3] *= -1.0
            if resolution != (224, 224) or self.rebuttal:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, (512, 384), rng=rng, info=impath
                )
                W, H = rgb_image.size
                cx = W // 2
                cy = H // 2
                l, t = cx - 112, cy - 112
                r, b = cx + 112, cy + 112
                crop_bbox = (l, t, r, b)
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, crop_bbox
                )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="nrgbd",
                    label=osp.join(scene_id, im_idx),
                    instance=impath,
                )
            )

        return views
