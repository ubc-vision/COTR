'''
compute distance matrix for megadepth using ComputeCanada
'''

import sys
sys.path.append('..')


import os
import argparse
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from COTR.options.options import *
from COTR.options.options_utils import *
from COTR.utils import debug_utils, utils, constants
from COTR.datasets import colmap_helper
from COTR.projector import pcd_projector
from COTR.global_configs import dataset_config


assert colmap_helper.COVISIBILITY_CHECK, 'Please enable COVISIBILITY_CHECK'
assert colmap_helper.LOAD_PCD, 'Please enable LOAD_PCD'

OFFSET_THRESHOLD = 1.0


def get_index_pairs(dist_mat, cells):
    pairs = []
    for row in range(dist_mat.shape[0]):
        for col in range(dist_mat.shape[0]):
            if dist_mat[row][col] == -1:
                pairs.append([row, col])
                if len(pairs) == cells:
                    return pairs
    return pairs


def load_dist_mat(path, size=None):
    if os.path.isfile(path):
        dist_mat = np.load(path)
        assert dist_mat.shape[0] == dist_mat.shape[1]
    else:
        dist_mat = np.ones([size, size], dtype=np.float32) * -1
    assert dist_mat.shape[0] == dist_mat.shape[1]
    return dist_mat


def distance_between_two_caps(caps):
    cap_1, cap_2 = caps
    try:
        if len(np.intersect1d(cap_1.point3d_id, cap_2.point3d_id)) == 0:
            return 0.0
        pcd = cap_2.point_cloud_world
        extrin_cap_1 = cap_1.cam_pose.world_to_camera[0:3, :]
        intrin_cap_1 = cap_1.pinhole_cam.intrinsic_mat
        size = cap_1.pinhole_cam.shape[:2]
        reproj = pcd_projector.PointCloudProjector.pcd_3d_to_pcd_2d_np(pcd[:, 0:3], intrin_cap_1, extrin_cap_1, size, keep_z=True, crop=True, filter_neg=True, norm_coord=False)
        reproj = pcd_projector.PointCloudProjector.pcd_2d_to_img_2d_np(reproj, size)[..., 0]
        # 1. calculate the iou
        query_mask = cap_1.depth_map > 0
        reproj_mask = reproj > 0
        intersection_mask = query_mask * reproj_mask
        union_mask = query_mask | reproj_mask
        if union_mask.sum() == 0:
            return 0.0
        intersection_mask = (abs(cap_1.depth_map - reproj) * intersection_mask < OFFSET_THRESHOLD) * intersection_mask
        ratio = intersection_mask.sum() / union_mask.sum()
        if ratio == 0.0:
            return 0.0
        return ratio
    except Exception as e:
        print(e)
        return 0.0


def fill_covisibility(scene, dist_mat):
    for i in range(dist_mat.shape[0]):
        nns = scene.get_covisible_caps(scene[i])
        covis_list = [scene.img_id_to_index_dict[cap.image_id] for cap in nns]
        for j in range(dist_mat.shape[0]):
            if j not in covis_list:
                dist_mat[i][j] = 0
    return dist_mat


def main(opt):
    # fast fail
    try:
        dist_mat = load_dist_mat(opt.out_path)
        if dist_mat.min() >= 0.0:
            print(f'{opt.out_path} is complete!')
            exit()
        else:
            print('continue working')
    except Exception as e:
        print(e)
        print('first time start working')
    scene_dir = opt.scenes_name_list[0]['scene_dir']
    image_dir = opt.scenes_name_list[0]['image_dir']
    depth_dir = opt.scenes_name_list[0]['depth_dir']
    scene = colmap_helper.ColmapWithDepthAsciiReader.read_sfm_scene_given_valid_list_path(scene_dir, image_dir, depth_dir, dataset_config[opt.dataset_name]['valid_list_json'], opt.crop_cam)
    size = len(scene.captures)
    dist_mat = load_dist_mat(opt.out_path, size)
    if opt.use_ram:
        scene.read_data_to_ram(['depth'])
    if dist_mat.max() == -1:
        dist_mat = fill_covisibility(scene, dist_mat)
        np.save(opt.out_path, dist_mat)
    pairs = get_index_pairs(dist_mat, opt.cells)
    in_pairs = [[scene[p[0]], scene[p[1]]] for p in pairs]
    results = Parallel(n_jobs=opt.num_cpus)(delayed(distance_between_two_caps)(pair) for pair in tqdm(in_pairs, desc='calculating distance matrix', total=len(in_pairs)))
    for i, p in enumerate(pairs):
        r, c = p
        dist_mat[r][c] = results[i]
    np.save(opt.out_path, dist_mat)
    print(f'finished from {pairs[0][0]}-{pairs[0][1]} -> {pairs[-1][0]}-{pairs[-1][1]}')
    print(f'in total {len(pairs)} cells')
    print(f'progress {(dist_mat >= 0).sum() / dist_mat.size}')
    print(f'save at {opt.out_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_general_arguments(parser)
    parser.add_argument('--dataset_name', type=str, default='megadepth', help='dataset name')
    parser.add_argument('--use_ram', type=str2bool, default=False, help='load image/depth to ram')
    parser.add_argument('--info_level', type=str, default='rgbd', help='the information level of dataset')
    parser.add_argument('--scene', type=str, default='0000', required=True, help='what scene want to use')
    parser.add_argument('--seq', type=str, default='0', required=True, help='what seq want to use')
    parser.add_argument('--crop_cam', choices=['no_crop', 'crop_center', 'crop_center_and_resize'], type=str, default='no_crop', help='crop the center of image to avoid changing aspect ratio, resize to make the operations batch-able.')
    parser.add_argument('--cells', type=int, default=10000, help='the number of cells to be computed in this run')
    parser.add_argument('--num_cpus', type=int, default=6, help='num of cores')

    opt = parser.parse_args()
    opt.scenes_name_list = options_utils.build_scenes_name_list_from_opt(opt)
    opt.out_dir = os.path.join(os.path.dirname(opt.scenes_name_list[0]['depth_dir']), 'dist_mat')
    opt.out_path = os.path.join(opt.out_dir, 'dist_mat.npy')
    os.makedirs(opt.out_dir, exist_ok=True)
    if opt.confirm:
        confirm_opt(opt)
    else:
        print_opt(opt)
    main(opt)
