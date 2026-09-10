import json
import os
import os.path as osp

import numpy as np
from PIL import Image

from eval.base import BaseStereoViewDataset


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
            camera = np.load(
                osp.join(self.ROOT, seq, "custom_undistorted_cam", name.replace(".JPG", ".npz"))
            )
            intrinsic = camera["intrinsics"].astype(np.float32)
            camera_pose = np.linalg.inv(camera["extrinsics"]).astype(np.float32)
            image, depthmap, intrinsic = self._crop_resize_if_necessary(
                image, depthmap, intrinsic, resolution, rng=rng, info=depth_path
            )
            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsic,
                    dataset="eth3d",
                    label=osp.join(seq, name),
                    instance=depth_path,
                )
            )
        return views
