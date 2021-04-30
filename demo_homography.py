'''
COTR demo for homography estimation
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

    img_a = imageio.imread('./sample_data/imgs/paint_1.JPG', pilmode='RGB')
    img_b = imageio.imread('./sample_data/imgs/paint_2.jpg', pilmode='RGB')
    rep_img = imageio.imread('./sample_data/imgs/Meisje_met_de_parel.jpg', pilmode='RGB')
    rep_mask = np.ones(rep_img.shape[:2])
    lu_corner = [932, 1025]
    ru_corner = [2469, 901]
    lb_corner = [908, 2927]
    rb_corner = [2436, 3080]
    queries = np.array([lu_corner, ru_corner, lb_corner, rb_corner]).astype(np.float32)
    rep_coord = np.array([[0, 0], [rep_img.shape[1], 0], [0, rep_img.shape[0]], [rep_img.shape[1], rep_img.shape[0]]]).astype(np.float32)

    engine = SparseEngine(model, 32, mode='stretching')
    corrs = engine.cotr_corr_multiscale(img_a, img_b, np.linspace(0.5, 0.0625, 4), 1, queries_a=queries, force=True)

    T = cv2.getPerspectiveTransform(rep_coord, corrs[:, 2:].astype(np.float32))
    vmask = cv2.warpPerspective(rep_mask, T, (img_b.shape[1], img_b.shape[0])) > 0
    warped = cv2.warpPerspective(rep_img, T, (img_b.shape[1], img_b.shape[0]))
    out = warped * vmask[..., None] + img_b * (~vmask[..., None])

    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(rep_img)
    axarr[0].title.set_text('Virtual Paint')
    axarr[0].axis('off')
    axarr[1].imshow(img_a)
    axarr[1].title.set_text('Annotated Frame')
    axarr[1].axis('off')
    axarr[2].imshow(img_b)
    axarr[2].title.set_text('Target Frame')
    axarr[2].axis('off')
    axarr[3].imshow(out)
    axarr[3].title.set_text('Overlay')
    axarr[3].axis('off')
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
