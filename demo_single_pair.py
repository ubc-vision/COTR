'''
COTR demo for a single image pair
'''
import argparse
import os
import time

import cv2
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt

from COTR.utils import utils, debug_utils
from COTR.models import build_model
from COTR.options.options import *
from COTR.options.options_utils import *
from COTR.inference.inference_helper import triangulate_corr
from COTR.inference.sparse_engine import SparseEngine

utils.fix_randomness(0)
torch.set_grad_enabled(False)


def main(opt):
    model = build_model(opt)
    model = model.cuda()
    weights = torch.load(opt.load_weights_path, map_location='cpu')['model_state_dict']
    utils.safe_load_weights(model, weights)
    model = model.eval()

    img_a = imageio.imread('./sample_data/imgs/cathedral_1.jpg', pilmode='RGB')
    img_b = imageio.imread('./sample_data/imgs/cathedral_2.jpg', pilmode='RGB')

    engine = SparseEngine(model, 32, mode='tile')
    t0 = time.time()
    corrs = engine.cotr_corr_multiscale_with_cycle_consistency(img_a, img_b, np.linspace(0.5, 0.0625, 4), 1, max_corrs=opt.max_corrs, queries_a=None)
    t1 = time.time()

    utils.visualize_corrs(img_a, img_b, corrs)
    print(f'spent {t1-t0} seconds for {opt.max_corrs} correspondences.')
    dense = triangulate_corr(corrs, img_a.shape, img_b.shape)
    warped = cv2.remap(img_b, dense[..., 0].astype(np.float32), dense[..., 1].astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    plt.imshow(warped / 255 * 0.5 + img_a / 255 * 0.5)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_COTR_arguments(parser)
    parser.add_argument('--out_dir', type=str, default=general_config['out'], help='out directory')
    parser.add_argument('--load_weights', type=str, default=None, help='load a pretrained set of weights, you need to provide the model id')
    parser.add_argument('--max_corrs', type=int, default=100, help='number of correspondences')

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
