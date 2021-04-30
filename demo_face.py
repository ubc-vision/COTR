'''
COTR demo for human face
We use an off-the-shelf face landmarks detector: https://github.com/1adrianb/face-alignment
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

    img_a = imageio.imread('./sample_data/imgs/face_1.png', pilmode='RGB')
    img_b = imageio.imread('./sample_data/imgs/face_2.png', pilmode='RGB')
    queries = np.load('./sample_data/face_landmarks.npy')[0]

    engine = SparseEngine(model, 32, mode='stretching')
    corrs = engine.cotr_corr_multiscale(img_a, img_b, np.linspace(0.5, 0.0625, 4), 1, queries_a=queries, force=False)

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img_a)
    axarr[0].scatter(*queries.T, s=1)
    axarr[0].title.set_text('Reference Face')
    axarr[0].axis('off')
    axarr[1].imshow(img_b)
    axarr[1].scatter(*corrs[:, 2:].T, s=1)
    axarr[1].title.set_text('Target Face')
    axarr[1].axis('off')
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
