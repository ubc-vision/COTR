'''
COTR two view reconstruction with known extrinsic/intrinsic demo
'''
import argparse
import os
import time

import numpy as np
import torch
import imageio
import open3d as o3d

from COTR.utils import utils, debug_utils
from COTR.models import build_model
from COTR.options.options import *
from COTR.options.options_utils import *
from COTR.inference.sparse_engine import SparseEngine, FasterSparseEngine
from COTR.projector import pcd_projector

utils.fix_randomness(0)
torch.set_grad_enabled(False)


def triangulate_rays_to_pcd(center_a, dir_a, center_b, dir_b):
    A = center_a
    a = dir_a / np.linalg.norm(dir_a, axis=1, keepdims=True)
    B = center_b
    b = dir_b / np.linalg.norm(dir_b, axis=1, keepdims=True)
    c = B - A
    D = A + a * ((-np.sum(a * b, axis=1) * np.sum(b * c, axis=1) + np.sum(a * c, axis=1) * np.sum(b * b, axis=1)) / (np.sum(a * a, axis=1) * np.sum(b * b, axis=1) - np.sum(a * b, axis=1) * np.sum(a * b, axis=1)))[..., None]
    return D


def main(opt):
    model = build_model(opt)
    model = model.cuda()
    weights = torch.load(opt.load_weights_path, map_location='cpu')['model_state_dict']
    utils.safe_load_weights(model, weights)
    model = model.eval()

    img_a = imageio.imread('./sample_data/imgs/img_0.jpg', pilmode='RGB')
    img_b = imageio.imread('./sample_data/imgs/img_1.jpg', pilmode='RGB')

    if opt.faster_infer:
        engine = FasterSparseEngine(model, 32, mode='tile')
    else:
        engine = SparseEngine(model, 32, mode='tile')
    t0 = time.time()
    corrs = engine.cotr_corr_multiscale_with_cycle_consistency(img_a, img_b, np.linspace(0.5, 0.0625, 4), 1, max_corrs=opt.max_corrs, queries_a=None)
    t1 = time.time()
    print(f'spent {t1-t0} seconds for {opt.max_corrs} correspondences.')

    camera_a = np.load('./sample_data/camera_0.npy', allow_pickle=True).item()
    camera_b = np.load('./sample_data/camera_1.npy', allow_pickle=True).item()
    center_a = camera_a['cam_center']
    center_b = camera_b['cam_center']
    rays_a = pcd_projector.PointCloudProjector.pcd_2d_to_pcd_3d_np(corrs[:, :2], np.ones([corrs.shape[0], 1]) * 2, camera_a['intrinsic'], motion=camera_a['c2w'])
    rays_b = pcd_projector.PointCloudProjector.pcd_2d_to_pcd_3d_np(corrs[:, 2:], np.ones([corrs.shape[0], 1]) * 2, camera_b['intrinsic'], motion=camera_b['c2w'])
    dir_a = rays_a - center_a
    dir_b = rays_b - center_b
    center_a = np.array([center_a] * corrs.shape[0])
    center_b = np.array([center_b] * corrs.shape[0])
    points = triangulate_rays_to_pcd(center_a, dir_a, center_b, dir_b)
    colors = (img_a[tuple(np.floor(corrs[:, :2]).astype(int)[:, ::-1].T)] / 255 + img_b[tuple(np.floor(corrs[:, 2:]).astype(int)[:, ::-1].T)] / 255) / 2
    colors = np.array(colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_COTR_arguments(parser)
    parser.add_argument('--out_dir', type=str, default=general_config['out'], help='out directory')
    parser.add_argument('--load_weights', type=str, default=None, help='load a pretrained set of weights, you need to provide the model id')
    parser.add_argument('--max_corrs', type=int, default=2048, help='number of correspondences')
    parser.add_argument('--faster_infer', type=str2bool, default=False, help='use fatser inference')

    opt = parser.parse_args()
    opt.command = ' '.join(sys.argv)

    layer_2_channels = {'layer1': 256,
                        'layer2': 512,
                        'layer3': 1024,
                        'layer4': 2048, }
    opt.dim_feedforward = layer_2_channels[opt.layer]
    if opt.load_weights:
        opt.load_weights_path = os.path.join(opt.out_dir, opt.load_weights, 'checkpoint.pth.tar')
    print_opt(opt)
    main(opt)
