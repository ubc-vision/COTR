'''
dataset specific layer for megadepth
'''

import os
import json
import random
from collections import namedtuple

import numpy as np

from COTR.datasets import colmap_helper
from COTR.global_configs import dataset_config
from COTR.sfm_scenes import knn_search
from COTR.utils import debug_utils, utils, constants

SceneCapIndex = namedtuple('SceneCapIndex', ['scene_index', 'capture_index'])


def prefix_of_img_path_for_magedepth(img_path):
    '''
    get the prefix for image of megadepth dataset
    '''
    prefix = os.path.abspath(os.path.join(img_path, '../../../..')) + '/'
    return prefix


class MegadepthSceneDataBase():
    scenes = {}
    knn_engine_dict = {}

    @classmethod
    def _load_scene(cls, opt, scene_dir_dict):
        if scene_dir_dict['scene_dir'] not in cls.scenes:
            if opt.info_level == 'rgb':
                assert 0
            elif opt.info_level == 'rgbd':
                scene_dir = scene_dir_dict['scene_dir']
                images_dir = scene_dir_dict['image_dir']
                depth_dir = scene_dir_dict['depth_dir']
                scene = colmap_helper.ColmapWithDepthAsciiReader.read_sfm_scene_given_valid_list_path(scene_dir, images_dir, depth_dir, dataset_config[opt.dataset_name]['valid_list_json'], opt.crop_cam)
                if opt.use_ram:
                    scene.read_data_to_ram(['image', 'depth'])
            else:
                raise ValueError()
            knn_engine = knn_search.ReprojRatioKnnSearch(scene)
            cls.scenes[scene_dir_dict['scene_dir']] = scene
            cls.knn_engine_dict[scene_dir_dict['scene_dir']] = knn_engine
        else:
            pass


class MegadepthDataset():

    def __init__(self, opt, dataset_type):
        assert dataset_type in ['train', 'val', 'test']
        assert len(opt.scenes_name_list) > 0
        self.opt = opt
        self.dataset_type = dataset_type
        self.use_ram = opt.use_ram
        self.scenes_name_list = opt.scenes_name_list
        self.scenes = None
        self.knn_engine_list = None
        self.total_caps_set = None
        self.query_caps_set = None
        self.db_caps_set = None
        self.img_path_to_scene_cap_index_dict = {}
        self.scene_index_to_db_caps_mask_dict = {}
        self._load_scenes()

    @property
    def num_scenes(self):
        return len(self.scenes)

    @property
    def num_queries(self):
        return len(self.query_caps_set)

    @property
    def num_db(self):
        return len(self.db_caps_set)

    def get_scene_cap_index_by_index(self, index):
        assert index < len(self.query_caps_set)
        img_path = sorted(list(self.query_caps_set))[index]
        scene_cap_index = self.img_path_to_scene_cap_index_dict[img_path]
        return scene_cap_index

    def _get_common_subset_caps_from_json(self, json_path, total_caps):
        prefix = prefix_of_img_path_for_magedepth(list(total_caps)[0])
        with open(json_path, 'r') as f:
            common_caps = [prefix + cap for cap in json.load(f)]
        common_caps = set(total_caps) & set(common_caps)
        return common_caps

    def _extend_img_path_to_scene_cap_index_dict(self, img_path_to_cap_index_dict, scene_id):
        for key in img_path_to_cap_index_dict.keys():
            self.img_path_to_scene_cap_index_dict[key] = SceneCapIndex(scene_id, img_path_to_cap_index_dict[key])

    def _create_scene_index_to_db_caps_mask_dict(self, db_caps_set):
        scene_index_to_db_caps_mask_dict = {}
        for cap in db_caps_set:
            scene_id, cap_id = self.img_path_to_scene_cap_index_dict[cap]
            if scene_id not in scene_index_to_db_caps_mask_dict:
                scene_index_to_db_caps_mask_dict[scene_id] = []
            scene_index_to_db_caps_mask_dict[scene_id].append(cap_id)
        for _k, _v in scene_index_to_db_caps_mask_dict.items():
            scene_index_to_db_caps_mask_dict[_k] = np.array(sorted(_v))
        return scene_index_to_db_caps_mask_dict

    def _load_scenes(self):
        scenes = []
        knn_engine_list = []
        total_caps_set = set()
        for scene_id, scene_dir_dict in enumerate(self.scenes_name_list):
            MegadepthSceneDataBase._load_scene(self.opt, scene_dir_dict)
            scene = MegadepthSceneDataBase.scenes[scene_dir_dict['scene_dir']]
            knn_engine = MegadepthSceneDataBase.knn_engine_dict[scene_dir_dict['scene_dir']]
            total_caps_set = total_caps_set | set(scene.img_path_to_index_dict.keys())
            self._extend_img_path_to_scene_cap_index_dict(scene.img_path_to_index_dict, scene_id)
            scenes.append(scene)
            knn_engine_list.append(knn_engine)
        self.scenes = scenes
        self.knn_engine_list = knn_engine_list
        self.total_caps_set = total_caps_set
        self.query_caps_set = self._get_common_subset_caps_from_json(dataset_config[self.opt.dataset_name][f'{self.dataset_type}_json'], total_caps_set)
        self.db_caps_set = self._get_common_subset_caps_from_json(dataset_config[self.opt.dataset_name]['train_json'], total_caps_set)
        self.scene_index_to_db_caps_mask_dict = self._create_scene_index_to_db_caps_mask_dict(self.db_caps_set)

    def get_query_with_knn(self, index):
        scene_index, cap_index = self.get_scene_cap_index_by_index(index)
        query_cap = self.scenes[scene_index].captures[cap_index]
        knn_engine = self.knn_engine_list[scene_index]
        if scene_index in self.scene_index_to_db_caps_mask_dict:
            db_mask = self.scene_index_to_db_caps_mask_dict[scene_index]
        else:
            db_mask = None
        pool = knn_engine.get_knn(query_cap, self.opt.pool_size, db_mask=db_mask)
        nn_caps = random.sample(pool, min(len(pool), self.opt.k_size))
        return query_cap, nn_caps
