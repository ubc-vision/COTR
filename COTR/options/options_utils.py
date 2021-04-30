'''utils for argparse
'''

import sys
import os
from os import path
import time
import json

from COTR.utils import utils, debug_utils
from COTR.global_configs import general_config, dataset_config


def str2bool(v: str) -> bool:
    return v.lower() in ('true', '1', 'yes', 'y', 't')


def print_opt(opt):
    content_list = []
    args = list(vars(opt))
    args.sort()
    for arg in args:
        content_list += [arg.rjust(25, ' ') + '  ' + str(getattr(opt, arg))]
    utils.print_notification(content_list, 'OPTIONS')
