'''
COTR dataset
'''

import random

import numpy as np
import torch
from torchvision.transforms import functional as tvtf
from torch.utils import data

from COTR.datasets import megadepth_dataset
from COTR.utils import debug_utils, utils, constants
from COTR.projector import pcd_projector
from COTR.cameras import capture
from COTR.utils.utils import CropCamConfig
from COTR.inference import inference_helper
from COTR.inference.inference_helper import two_images_side_by_side


class COTRDataset(data.Dataset):
    def __init__(self, opt, dataset_type: str):
        assert dataset_type in ['train', 'val', 'test']
        assert len(opt.scenes_name_list) > 0
        self.opt = opt
        self.dataset_type = dataset_type
        self.sfm_dataset = megadepth_dataset.MegadepthDataset(opt, dataset_type)

        self.kp_pool = opt.kp_pool
        self.num_kp = opt.num_kp
        self.bidirectional = opt.bidirectional
        self.need_rotation = opt.need_rotation
        self.max_rotation = opt.max_rotation
        self.rotation_chance = opt.rotation_chance

    def _trim_corrs(self, in_corrs):
        length = in_corrs.shape[0]
        if length >= self.num_kp:
            mask = np.random.choice(length, self.num_kp)
            return in_corrs[mask]
        else:
            mask = np.random.choice(length, self.num_kp - length)
            return np.concatenate([in_corrs, in_corrs[mask]], axis=0)

    def __len__(self):
        if self.dataset_type == 'val':
            return min(1000, self.sfm_dataset.num_queries)
        else:
            return self.sfm_dataset.num_queries

    def augment_with_rotation(self, query_cap, nn_cap):
        if random.random() < self.rotation_chance:
            theta = np.random.uniform(low=-1, high=1) * self.max_rotation
            query_cap = capture.rotate_capture(query_cap, theta)
        if random.random() < self.rotation_chance:
            theta = np.random.uniform(low=-1, high=1) * self.max_rotation
            nn_cap = capture.rotate_capture(nn_cap, theta)
        return query_cap, nn_cap

    def __getitem__(self, index):
        assert self.opt.k_size == 1
        query_cap, nn_caps = self.sfm_dataset.get_query_with_knn(index)
        nn_cap = nn_caps[0]

        if self.need_rotation:
            query_cap, nn_cap = self.augment_with_rotation(query_cap, nn_cap)

        nn_keypoints_y, nn_keypoints_x = np.where(nn_cap.depth_map > 0)
        nn_keypoints_y = nn_keypoints_y[..., None]
        nn_keypoints_x = nn_keypoints_x[..., None]
        nn_keypoints_z = nn_cap.depth_map[np.floor(nn_keypoints_y).astype('int'), np.floor(nn_keypoints_x).astype('int')]
        nn_keypoints_xy = np.concatenate([nn_keypoints_x, nn_keypoints_y], axis=1)
        nn_keypoints_3d_world, valid_index_1 = pcd_projector.PointCloudProjector.pcd_2d_to_pcd_3d_np(nn_keypoints_xy, nn_keypoints_z, nn_cap.pinhole_cam.intrinsic_mat, motion=nn_cap.cam_pose.camera_to_world, return_index=True)

        query_keypoints_xyz, valid_index_2 = pcd_projector.PointCloudProjector.pcd_3d_to_pcd_2d_np(
            nn_keypoints_3d_world,
            query_cap.pinhole_cam.intrinsic_mat,
            query_cap.cam_pose.world_to_camera[0:3, :],
            query_cap.image.shape[:2],
            keep_z=True,
            crop=True,
            filter_neg=True,
            norm_coord=False,
            return_index=True,
        )
        query_keypoints_xy = query_keypoints_xyz[:, 0:2]
        query_keypoints_z_proj = query_keypoints_xyz[:, 2:3]
        query_keypoints_z = query_cap.depth_map[np.floor(query_keypoints_xy[:, 1:2]).astype('int'), np.floor(query_keypoints_xy[:, 0:1]).astype('int')]
        mask = (abs(query_keypoints_z - query_keypoints_z_proj) < 0.5)[:, 0]
        query_keypoints_xy = query_keypoints_xy[mask]

        if query_keypoints_xy.shape[0] < self.num_kp:
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        nn_keypoints_xy = nn_keypoints_xy[valid_index_1][valid_index_2][mask]
        assert nn_keypoints_xy.shape == query_keypoints_xy.shape
        corrs = np.concatenate([query_keypoints_xy, nn_keypoints_xy], axis=1)
        corrs = self._trim_corrs(corrs)
        # flip augmentation
        if np.random.uniform() < 0.5:
            corrs[:, 0] = constants.MAX_SIZE - 1 - corrs[:, 0]
            corrs[:, 2] = constants.MAX_SIZE - 1 - corrs[:, 2]
            sbs_img = two_images_side_by_side(np.fliplr(query_cap.image), np.fliplr(nn_cap.image))
        else:
            sbs_img = two_images_side_by_side(query_cap.image, nn_cap.image)
        corrs[:, 2] += constants.MAX_SIZE
        corrs /= np.array([constants.MAX_SIZE * 2, constants.MAX_SIZE, constants.MAX_SIZE * 2, constants.MAX_SIZE])
        assert (0.0 <= corrs[:, 0]).all() and (corrs[:, 0] <= 0.5).all()
        assert (0.0 <= corrs[:, 1]).all() and (corrs[:, 1] <= 1.0).all()
        assert (0.5 <= corrs[:, 2]).all() and (corrs[:, 2] <= 1.0).all()
        assert (0.0 <= corrs[:, 3]).all() and (corrs[:, 3] <= 1.0).all()
        out = {
            'image': tvtf.normalize(tvtf.to_tensor(sbs_img), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            'corrs': torch.from_numpy(corrs).float(),
        }
        if self.bidirectional:
            out['queries'] = torch.from_numpy(np.concatenate([corrs[:, :2], corrs[:, 2:]], axis=0)).float()
            out['targets'] = torch.from_numpy(np.concatenate([corrs[:, 2:], corrs[:, :2]], axis=0)).float()
        else:
            out['queries'] = torch.from_numpy(corrs[:, :2]).float()
            out['targets'] = torch.from_numpy(corrs[:, 2:]).float()
        return out


class COTRZoomDataset(COTRDataset):
    def __init__(self, opt, dataset_type: str):
        assert opt.crop_cam in ['no_crop', 'crop_center']
        assert opt.use_ram == False
        super().__init__(opt, dataset_type)
        self.zoom_start = opt.zoom_start
        self.zoom_end = opt.zoom_end
        self.zoom_levels = opt.zoom_levels
        self.zoom_jitter = opt.zoom_jitter
        self.zooms = np.logspace(np.log10(opt.zoom_start),
                                 np.log10(opt.zoom_end),
                                 num=opt.zoom_levels)

    def get_corrs(self, from_cap, to_cap, reduced_size=None):
        from_y, from_x = np.where(from_cap.depth_map > 0)
        from_y, from_x = from_y[..., None], from_x[..., None]
        if reduced_size is not None:
            filter_idx = np.random.choice(from_y.shape[0], reduced_size, replace=False)
            from_y, from_x = from_y[filter_idx], from_x[filter_idx]
        from_z = from_cap.depth_map[np.floor(from_y).astype('int'), np.floor(from_x).astype('int')]
        from_xy = np.concatenate([from_x, from_y], axis=1)
        from_3d_world, valid_index_1 = pcd_projector.PointCloudProjector.pcd_2d_to_pcd_3d_np(from_xy, from_z, from_cap.pinhole_cam.intrinsic_mat, motion=from_cap.cam_pose.camera_to_world, return_index=True)

        to_xyz, valid_index_2 = pcd_projector.PointCloudProjector.pcd_3d_to_pcd_2d_np(
            from_3d_world,
            to_cap.pinhole_cam.intrinsic_mat,
            to_cap.cam_pose.world_to_camera[0:3, :],
            to_cap.image.shape[:2],
            keep_z=True,
            crop=True,
            filter_neg=True,
            norm_coord=False,
            return_index=True,
        )

        to_xy = to_xyz[:, 0:2]
        to_z_proj = to_xyz[:, 2:3]
        to_z = to_cap.depth_map[np.floor(to_xy[:, 1:2]).astype('int'), np.floor(to_xy[:, 0:1]).astype('int')]
        mask = (abs(to_z - to_z_proj) < 0.5)[:, 0]
        if mask.sum() > 0:
            return np.concatenate([from_xy[valid_index_1][valid_index_2][mask], to_xy[mask]], axis=1)
        else:
            return None

    def get_seed_corr(self, from_cap, to_cap, max_try=100):
        seed_corr = self.get_corrs(from_cap, to_cap, reduced_size=max_try)
        if seed_corr is None:
            return None
        shuffle = np.random.permutation(seed_corr.shape[0])
        seed_corr = np.take(seed_corr, shuffle, axis=0)
        return seed_corr[0]

    def get_zoomed_cap(self, cap, pos, scale, jitter):
        patch = inference_helper.get_patch_centered_at(cap.image, pos, scale=scale, return_content=False)
        patch = inference_helper.get_patch_centered_at(cap.image,
                                                  pos + np.array([patch.w, patch.h]) * np.random.uniform(-jitter, jitter, 2),
                                                  scale=scale,
                                                  return_content=False)
        zoom_config = CropCamConfig(x=patch.x,
                                    y=patch.y,
                                    w=patch.w,
                                    h=patch.h,
                                    out_w=constants.MAX_SIZE,
                                    out_h=constants.MAX_SIZE,
                                    orig_w=cap.shape[1],
                                    orig_h=cap.shape[0])
        zoom_cap = capture.crop_capture(cap, zoom_config)
        return zoom_cap

    def __getitem__(self, index):
        assert self.opt.k_size == 1
        query_cap, nn_caps = self.sfm_dataset.get_query_with_knn(index)
        nn_cap = nn_caps[0]
        if self.need_rotation:
            query_cap, nn_cap = self.augment_with_rotation(query_cap, nn_cap)

        # find seed
        seed_corr = self.get_seed_corr(nn_cap, query_cap)
        if seed_corr is None:
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        # crop cap
        s = np.random.choice(self.zooms)
        nn_zoom_cap = self.get_zoomed_cap(nn_cap, seed_corr[:2], s, 0)
        query_zoom_cap = self.get_zoomed_cap(query_cap, seed_corr[2:], s, self.zoom_jitter)
        assert nn_zoom_cap.shape == query_zoom_cap.shape == (constants.MAX_SIZE, constants.MAX_SIZE)
        corrs = self.get_corrs(query_zoom_cap, nn_zoom_cap)
        if corrs is None or corrs.shape[0] < self.num_kp:
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        shuffle = np.random.permutation(corrs.shape[0])
        corrs = np.take(corrs, shuffle, axis=0)
        corrs = self._trim_corrs(corrs)

        # flip augmentation
        if np.random.uniform() < 0.5:
            corrs[:, 0] = constants.MAX_SIZE - 1 - corrs[:, 0]
            corrs[:, 2] = constants.MAX_SIZE - 1 - corrs[:, 2]
            sbs_img = two_images_side_by_side(np.fliplr(query_zoom_cap.image), np.fliplr(nn_zoom_cap.image))
        else:
            sbs_img = two_images_side_by_side(query_zoom_cap.image, nn_zoom_cap.image)

        corrs[:, 2] += constants.MAX_SIZE
        corrs /= np.array([constants.MAX_SIZE * 2, constants.MAX_SIZE, constants.MAX_SIZE * 2, constants.MAX_SIZE])
        assert (0.0 <= corrs[:, 0]).all() and (corrs[:, 0] <= 0.5).all()
        assert (0.0 <= corrs[:, 1]).all() and (corrs[:, 1] <= 1.0).all()
        assert (0.5 <= corrs[:, 2]).all() and (corrs[:, 2] <= 1.0).all()
        assert (0.0 <= corrs[:, 3]).all() and (corrs[:, 3] <= 1.0).all()
        out = {
            'image': tvtf.normalize(tvtf.to_tensor(sbs_img), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            'corrs': torch.from_numpy(corrs).float(),
        }
        if self.bidirectional:
            out['queries'] = torch.from_numpy(np.concatenate([corrs[:, :2], corrs[:, 2:]], axis=0)).float()
            out['targets'] = torch.from_numpy(np.concatenate([corrs[:, 2:], corrs[:, :2]], axis=0)).float()
        else:
            out['queries'] = torch.from_numpy(corrs[:, :2]).float()
            out['targets'] = torch.from_numpy(corrs[:, 2:]).float()

        return out
