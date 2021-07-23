import os
import json

import tables
from tqdm import tqdm
import numpy as np


def read_all_imgs(base_dir):
    all_imgs = []
    for cur, dirs, files in os.walk(base_dir):
        if 'imgs' in cur:
            all_imgs += [os.path.join(cur, f) for f in files]
    all_imgs.sort()
    return all_imgs


def filter_semantic_depth(imgs):
    valid_imgs = []
    for item in tqdm(imgs):
        f_name = os.path.splitext(os.path.basename(item))[0] + '.h5'
        depth_dir = os.path.abspath(os.path.join(os.path.dirname(item), '../depths'))
        depth_path = os.path.join(depth_dir, f_name)
        depth_h5 = tables.open_file(depth_path, mode='r')
        _depth = np.array(depth_h5.root.depth)
        if _depth.min() >= 0:
            prefix = os.path.abspath(os.path.join(item, '../../../../')) + '/'
            rel_image_path = item.replace(prefix, '')
            valid_imgs.append(rel_image_path)
        depth_h5.close()
    valid_imgs.sort()
    return valid_imgs


if __name__ == "__main__":
    MegaDepth_v1 = '/media/jiangwei/data_ssd/MegaDepth_v1/'
    assert os.path.isdir(MegaDepth_v1), 'Change to your local path'
    all_imgs = read_all_imgs(MegaDepth_v1)
    valid_imgs = filter_semantic_depth(all_imgs)
    with open('megadepth_valid_list.json', 'w') as outfile:
        json.dump(valid_imgs, outfile, indent=4)
