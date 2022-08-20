import os
import torch
import numpy as np
import json

import utils.general as utils
from utils import rend_util

class BlenderDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 train_cameras,
                 data_dir,
                 img_res,
                 case_name='lego',
                 split='train',
                 **kwargs
                 ):

        self.instance_dir = os.path.join('../data', data_dir, '{0}'.format(case_name))

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        self.train_cameras = train_cameras

        self.split = split

        # load/parse metadata
        meta_fname = "{}/transforms_{}.json".format(self.instance_dir, self.split)
        with open(meta_fname) as file:
            self.meta = json.load(file)
        self.list = self.meta["frames"]

        self.images_lis = [f"{self.instance_dir}/{x['file_path']}.png" for x in self.list]
        self.n_images = len(self.images_lis)

        self.H, self.W = self.img_res
        self.focal = 0.5*self.W/np.tan(0.5*self.meta["camera_angle_x"])
        self.intrinsics_all = [torch.tensor([
                [self.focal, 0, self.W / 2, 0], 
                [0, self.focal, self.H / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=torch.float32) for _ in range(self.n_images)]
        self.pose_all = [torch.from_numpy(self.parse_raw_camera(np.array(frame['transform_matrix']))).float()
                for frame in self.list]

        self.images_lis = [f"{self.instance_dir}/{x['file_path']}.png" for x in self.list]
        self.rgb_images = []
        self.object_masks = []
        for im_name in self.images_lis:
            rgba = rend_util.load_rgb(im_name)
            rgb = rgba[:3]
            object_mask = rgba[3] > 0
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(object_mask).bool())

    def parse_raw_camera(self, pose_raw):
        """
        Convert blendshape coordinate space to OpenGL space
        pose_raw: [4, 4]
        return: [4, 4] c2w pose
        """
        pose_flip = np.diag([1, -1, -1, 1])
        pose = pose_raw @ pose_flip
        # pose = np.linalg.inv(pose)
        return pose

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "object_mask": self.object_masks[idx],
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]

        if not self.train_cameras:
            sample["pose"] = self.pose_all[idx]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.eye(4)

    def get_gt_pose(self, scaled=False):
        # Load gt pose without normalization to unit sphere
        init_pose = torch.stack(self.pose_all)
        init_quat = rend_util.rot_to_quat(init_pose[:, :3, :3])
        init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat

    def get_pose_init(self):
        # no noise for now
        init_pose = torch.stack(self.pose_all).cuda()
        init_quat = rend_util.rot_to_quat(init_pose[:, :3, :3])
        init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat
