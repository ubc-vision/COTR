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

