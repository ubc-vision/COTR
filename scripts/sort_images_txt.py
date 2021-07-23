import sys
assert sys.version_info >= (3, 7), 'ordered dict is required'
import os
import argparse
import re

from tqdm import tqdm


def read_images_meta(images_txt_path):
    images_meta = {}
    with open(images_txt_path, "r") as fid:
        line = fid.readline()
        assert line == '# Image list with two lines of data per image:\n'
        line = fid.readline()
        assert line == '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n'
        line = fid.readline()
        assert line == '#   POINTS2D[] as (X, Y, POINT3D_ID)\n'
        line = fid.readline()
        assert re.search('^# Number of images: \d+, mean observations per image: [-+]?\d*\.\d+|\d+\n$', line)
        num_images, mean_ob_per_img = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        num_images = int(num_images)
        mean_ob_per_img = float(mean_ob_per_img)

        for _ in tqdm(range(num_images), desc='reading images meta'):
            l = fid.readline()
            elems = l.split()
            image_id = int(elems[0])
            l2 = fid.readline()
            images_meta[image_id] = [l, l2]
    return images_meta


def read_header(images_txt_path):
    header = []
    with open(images_txt_path, "r") as fid:
        line = fid.readline()
        assert line == '# Image list with two lines of data per image:\n'
        header.append(line)
        line = fid.readline()
        assert line == '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n'
        header.append(line)
        line = fid.readline()
        assert line == '#   POINTS2D[] as (X, Y, POINT3D_ID)\n'
        header.append(line)
        line = fid.readline()
        assert re.search('^# Number of images: \d+, mean observations per image: [-+]?\d*\.\d+|\d+\n$', line)
        header.append(line)
    return header


def export_images_txt(save_to, header, content):
    assert not os.path.isfile(save_to), 'you are overriding existing files'
    with open(save_to, "w") as fid:
        for l in header:
            fid.write(l)
        for k, item in content.items():
            for l in item:
                fid.write(l)


def main(opt):
    reference = read_images_meta(opt.reference_images_txt)
    unordered = read_images_meta(opt.unordered_images_txt)
    ordered = {}
    for k in reference.keys():
        ordered[k] = unordered[k]
    header = read_header(opt.unordered_images_txt)
    export_images_txt(opt.save_to, header, ordered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_images_txt', type=str, default=None, required=True)
    parser.add_argument('--unordered_images_txt', type=str, default=None, required=True)
    parser.add_argument('--save_to', type=str, default=None, required=True)
    opt = parser.parse_args()
    main(opt)
