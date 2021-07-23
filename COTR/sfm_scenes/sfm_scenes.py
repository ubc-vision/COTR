'''
Scene reconstructed from SFM, mainly colmap
'''
import os
import copy
import math

import numpy as np
from numpy.linalg import inv
from tqdm import tqdm

from COTR.transformations import transformations
from COTR.transformations.transform_basics import Translation, Rotation
from COTR.cameras.camera_pose import CameraPose
from COTR.utils import debug_utils


class SfmScene():
    def __init__(self, captures, point_cloud=None):
        self.captures = captures
        if isinstance(point_cloud, tuple):
            self.point_cloud = point_cloud[0]
            self.point_meta = point_cloud[1]
        else:
            self.point_cloud = point_cloud
        self.img_path_to_index_dict = {}
        self.img_id_to_index_dict = {}
        self.fname_to_index_dict = {}
        self._build_img_X_to_index_dict()

    def __str__(self):
        string = 'Scene contains {0} captures'.format(len(self.captures))
        return string

    def __getitem__(self, x):
        if isinstance(x, str):
            try:
                return self.captures[self.img_path_to_index_dict[x]]
            except:
                return self.captures[self.fname_to_index_dict[x]]
        else:
            return self.captures[x]

    def _build_img_X_to_index_dict(self):
        assert self.captures is not None, 'There is no captures'
        for i, cap in enumerate(self.captures):
            assert cap.img_path not in self.img_path_to_index_dict, 'Image already exists'
            self.img_path_to_index_dict[cap.img_path] = i
            assert os.path.basename(cap.img_path) not in self.fname_to_index_dict, 'Image already exists'
            self.fname_to_index_dict[os.path.basename(cap.img_path)] = i
            if hasattr(cap, 'image_id'):
                self.img_id_to_index_dict[cap.image_id] = i

    def get_captures_given_index_list(self, index_list):
        captures_list = []
        for i in index_list:
            captures_list.append(self.captures[i])
        return captures_list

    def get_covisible_caps(self, cap):
        assert cap.img_path in self.img_path_to_index_dict
        covis_img_id = set()
        point_ids = cap.point3d_id
        for i in point_ids:
            covis_img_id = covis_img_id.union(set(self.point_meta[i].image_ids))
        covis_caps = []
        for i in covis_img_id:
            if i in self.img_id_to_index_dict:
                covis_caps.append(self.captures[self.img_id_to_index_dict[i]])
            else:
                pass
        return covis_caps

    def read_data_to_ram(self, data_list):
        print('warning: you are going to use a lot of RAM.')
        sum_bytes = 0.0
        pbar = tqdm(self.captures, desc='reading data, memory usage {0:.2f} MB'.format(sum_bytes / (1024.0 * 1024.0)))
        for cap in pbar:
            if 'image' in data_list:
                sum_bytes += cap.read_image_to_ram()
            if 'depth' in data_list:
                sum_bytes += cap.read_depth_to_ram()
            if 'pcd' in data_list:
                sum_bytes += cap.read_pcd_to_ram()
            pbar.set_description('reading data, memory usage {0:.2f} MB'.format(sum_bytes / (1024.0 * 1024.0)))
        print('----- total memory usage for images: {0} MB-----'.format(sum_bytes / (1024.0 * 1024.0)))

