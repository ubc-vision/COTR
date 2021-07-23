import sys
import argparse
import json
import os


from COTR.options.options_utils import str2bool
from COTR.options import options_utils
from COTR.global_configs import general_config, dataset_config
from COTR.utils import debug_utils


def set_general_arguments(parser):
    general_arg = parser.add_argument_group('General')
    general_arg.add_argument('--confirm', type=str2bool,
                             default=True, help='promote confirmation for user')
    general_arg.add_argument('--use_cuda', type=str2bool,
                             default=True, help='use cuda')
    general_arg.add_argument('--use_cc', type=str2bool,
                             default=False, help='use computecanada')


def set_dataset_arguments(parser):
    data_arg = parser.add_argument_group('Data')
    data_arg.add_argument('--dataset_name', type=str, default='megadepth', help='dataset name')
    data_arg.add_argument('--shuffle_data', type=str2bool, default=True, help='use sequence dataset or shuffled dataset')
    data_arg.add_argument('--use_ram', type=str2bool, default=False, help='load image/depth/pcd to ram')
    data_arg.add_argument('--info_level', choices=['rgb', 'rgbd'], type=str, default='rgbd', help='the information level of dataset')
    data_arg.add_argument('--scene_file', type=str, default=None, required=False, help='what scene/seq want to use')
    data_arg.add_argument('--workers', type=int, default=0, help='worker for loading data')
    data_arg.add_argument('--crop_cam', choices=['no_crop', 'crop_center', 'crop_center_and_resize'], type=str, default='crop_center_and_resize', help='crop the center of image to avoid changing aspect ratio, resize to make the operations batch-able.')


def set_nn_arguments(parser):
    nn_arg = parser.add_argument_group('Nearest neighbors')
    nn_arg.add_argument('--nn_method', choices=['netvlad', 'overlapping'], type=str, default='overlapping', help='how to select nearest neighbors')
    nn_arg.add_argument('--pool_size', type=int, default=20, help='a pool of sorted nn candidates')
    nn_arg.add_argument('--k_size', type=int, default=1, help='select the nn randomly from pool')


def set_COTR_arguments(parser):
    cotr_arg = parser.add_argument_group('COTR model')
    cotr_arg.add_argument('--backbone', type=str, default='resnet50')
    cotr_arg.add_argument('--hidden_dim', type=int, default=256)
    cotr_arg.add_argument('--dilation', type=str2bool, default=False)
    cotr_arg.add_argument('--dropout', type=float, default=0.1)
    cotr_arg.add_argument('--nheads', type=int, default=8)
    cotr_arg.add_argument('--layer', type=str, default='layer3', help='which layer from resnet')
    cotr_arg.add_argument('--enc_layers', type=int, default=6)
    cotr_arg.add_argument('--dec_layers', type=int, default=6)
    cotr_arg.add_argument('--position_embedding', type=str, default='lin_sine', help='sine wave type')

