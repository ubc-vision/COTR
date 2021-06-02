'''
Feature-free COTR guided matching for keypoints.
We use DISK(https://github.com/cvlab-epfl/disk) keypoints location.
We apply RANSAC + F matrix to further prune outliers.
Note: This script doesn't use descriptors.
'''
import argparse
import os
import time

import cv2
import numpy as np
import torch
import imageio
from scipy.spatial import distance_matrix

from COTR.utils import utils, debug_utils
from COTR.models import build_model
from COTR.options.options import *
from COTR.options.options_utils import *
from COTR.inference.sparse_engine import SparseEngine, FasterSparseEngine

utils.fix_randomness(0)
torch.set_grad_enabled(False)


def main(opt):
    model = build_model(opt)
    model = model.cuda()
    weights = torch.load(opt.load_weights_path)['model_state_dict']
    utils.safe_load_weights(model, weights)
    model = model.eval()

    img_a = imageio.imread('./sample_data/imgs/21526113_4379776807.jpg')
    img_b = imageio.imread('./sample_data/imgs/21126421_4537535153.jpg')
    kp_a = np.load('./sample_data/21526113_4379776807.jpg.disk.kpts.npy')
    kp_b = np.load('./sample_data/21126421_4537535153.jpg.disk.kpts.npy')

    if opt.faster_infer:
        engine = FasterSparseEngine(model, 32, mode='tile')
    else:
        engine = SparseEngine(model, 32, mode='tile')
    t0 = time.time()
    corrs_a_b = engine.cotr_corr_multiscale(img_a, img_b, np.linspace(0.5, 0.0625, 4), 1, max_corrs=kp_a.shape[0], queries_a=kp_a, force=True)
    corrs_b_a = engine.cotr_corr_multiscale(img_b, img_a, np.linspace(0.5, 0.0625, 4), 1, max_corrs=kp_b.shape[0], queries_a=kp_b, force=True)
    t1 = time.time()
    print(f'COTR spent {t1-t0} seconds.')
    inds_a_b = np.argmin(distance_matrix(corrs_a_b[:, 2:], kp_b), axis=1)
    matched_a_b = np.stack([np.arange(kp_a.shape[0]), inds_a_b]).T
    inds_b_a = np.argmin(distance_matrix(corrs_b_a[:, 2:], kp_a), axis=1)
    matched_b_a = np.stack([np.arange(kp_b.shape[0]), inds_b_a]).T

    good = 0
    final_matches = []
    for m_ab in matched_a_b:
        for m_ba in matched_b_a:
            if (m_ab == m_ba[::-1]).all():
                good += 1
                final_matches.append(m_ab)
                break
    final_matches = np.array(final_matches)
    final_corrs = np.concatenate([kp_a[final_matches[:, 0]], kp_b[final_matches[:, 1]]], axis=1)
    _, mask = cv2.findFundamentalMat(final_corrs[:, :2], final_corrs[:, 2:], cv2.FM_RANSAC, ransacReprojThreshold=5, confidence=0.999999)
    utils.visualize_corrs(img_a, img_b, final_corrs[np.where(mask[:, 0])])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_COTR_arguments(parser)
    parser.add_argument('--out_dir', type=str, default=general_config['out'], help='out directory')
    parser.add_argument('--load_weights', type=str, default=None, help='load a pretrained set of weights, you need to provide the model id')
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
