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


def get_compact_naming_cotr(opt) -> str:
    base_str = 'model:cotr_{0}_{1}_{2}_dset:{3}_bs:{4}_pe:{5}_lrbackbone:{6}'
    result = base_str.format(opt.backbone,
                             opt.layer,
                             opt.dim_feedforward,
                             opt.dataset_name,
                             opt.batch_size,
                             opt.position_embedding,
                             opt.lr_backbone,
                             )
    if opt.suffix:
        result = result + '_suffix:{0}'.format(opt.suffix)
    return result


def print_opt(opt):
    content_list = []
    args = list(vars(opt))
    args.sort()
    for arg in args:
        content_list += [arg.rjust(25, ' ') + '  ' + str(getattr(opt, arg))]
    utils.print_notification(content_list, 'OPTIONS')


def confirm_opt(opt):
    print_opt(opt)
    if opt.use_cc == False:
        if not utils.confirm():
            exit(1)


def opt_to_string(opt) -> str:
    string = '\n\n'
    string += 'python ' + ' '.join(sys.argv)
    string += '\n\n'
    # string += '---------------------- CONFIG ----------------------\n'
    args = list(vars(opt))
    args.sort()
    for arg in args:
        string += arg.rjust(25, ' ') + '  ' + str(getattr(opt, arg)) + '\n\n'
    # string += '----------------------------------------------------\n'
    return string


def save_opt(opt):
    '''save options to a json file
    '''
    if not os.path.exists(opt.out):
        os.makedirs(opt.out)
    json_path = os.path.join(opt.out, 'params.json')
    if 'debug' not in opt.suffix and path.isfile(json_path):
        assert opt.resume, 'You are trying to modify a model without resuming: {0}'.format(opt.out)
        old_dict = json.load(open(json_path))
        new_dict = vars(opt)
        # assert old_dict.keys() == new_dict.keys(), 'New configuration keys is different from old one.\nold: {0}\nnew: {1}'.format(old_dict.keys(), new_dict.keys())
        if new_dict != old_dict:
            exception_keys = ['command']
            for key in set(old_dict.keys()).union(set(new_dict.keys())):
                if key not in exception_keys:
                    old_val = old_dict[key] if key in old_dict else 'not exists(old)'
                    new_val = new_dict[key] if key in old_dict else 'not exists(new)'
                    if old_val != new_val:
                        print('key: {0}, old_val: {1}, new_val: {2}'.format(key, old_val, new_val))
            if opt.use_cc == False:
                if not utils.confirm('Please manually confirm'):
                    exit(1)
    with open(json_path, 'w') as fp:
        json.dump(vars(opt), fp, indent=0, sort_keys=True)


def build_scenes_name_list_from_opt(opt):
    if hasattr(opt, 'scene_file') and opt.scene_file is not None:
        assert os.path.isfile(opt.scene_file), opt.scene_file
        with open(opt.scene_file, 'r') as f:
                scenes_list = json.load(f)
    else:
        scenes_list = [{'scene': opt.scene, 'seq': opt.seq}]
    if 'megadepth' in opt.dataset_name:
        assert opt.info_level in ['rgb', 'rgbd']
        scenes_name_list = []
        if opt.info_level == 'rgb':
            dir_list = ['scene_dir', 'image_dir']
        elif opt.info_level == 'rgbd':
            dir_list = ['scene_dir', 'image_dir', 'depth_dir']
        dir_list = {dir_name: dataset_config[opt.dataset_name][dir_name] for dir_name in dir_list}
        for item in scenes_list:
            cur_scene = {key: val.format(item['scene'], item['seq']) for key, val in dir_list.items()}
            scenes_name_list.append(cur_scene)
    else:
        raise NotImplementedError()
    return scenes_name_list
