'''
Manually passing scale to COTR, skip the scale difference estimation.
'''
import argparse
import os
import time

import cv2
import numpy as np
import torch
import imageio
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

from COTR.utils import utils, debug_utils
from COTR.models import build_model
from COTR.options.options import *
from COTR.options.options_utils import *
from COTR.inference.sparse_engine import SparseEngine

utils.fix_randomness(0)
torch.set_grad_enabled(False)


def main(opt):
    model = build_model(opt)
    model = model.cuda()
    weights = torch.load(opt.load_weights_path)['model_state_dict']
    utils.safe_load_weights(model, weights)
    model = model.eval()

    img_a = imageio.imread('./sample_data/imgs/petrzin_01.png')
    img_b = imageio.imread('./sample_data/imgs/petrzin_02.png')
    img_a_area = 1.0
    img_b_area = 1.0
    gt_corrs = np.loadtxt('./sample_data/petrzin_pts.txt')
    kp_a = gt_corrs[:, :2]
    kp_b = gt_corrs[:, 2:]

    engine = SparseEngine(model, 32, mode='tile')
    t0 = time.time()
    corrs = engine.cotr_corr_multiscale(img_a, img_b, np.linspace(0.75, 0.1, 4), 1, max_corrs=kp_a.shape[0], queries_a=kp_a, force=True, areas=[img_a_area, img_b_area])
    t1 = time.time()
    print(f'COTR spent {t1-t0} seconds.')

    utils.visualize_corrs(img_a, img_b, corrs)
    plt.imshow(img_b)
    plt.scatter(kp_b[:,0], kp_b[:,1])
    plt.scatter(corrs[:,2], corrs[:,3])
    plt.plot(np.stack([kp_b[:,0], corrs[:,2]], axis=1).T, np.stack([kp_b[:,1], corrs[:,3]], axis=1).T, color=[1,0,0])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_COTR_arguments(parser)
    parser.add_argument('--out_dir', type=str, default=general_config['out'], help='out directory')
    parser.add_argument('--load_weights', type=str, default=None, help='load a pretrained set of weights, you need to provide the model id')

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
