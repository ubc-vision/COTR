import random
import smtplib
import ssl
from collections import namedtuple

from COTR.utils import debug_utils

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import PIL


'''
ImagePatch: patch: patch content, np array or None
            x: left bound in original resolution
            y: upper bound in original resolution
            w: width of patch
            h: height of patch
            ow: width of original resolution
            oh: height of original resolution
'''
ImagePatch = namedtuple('ImagePatch', ['patch', 'x', 'y', 'w', 'h', 'ow', 'oh'])
Point3D = namedtuple("Point3D", ["id", "arr_idx", "image_ids"])
Point2D = namedtuple("Point2D", ["id_3d", "xy"])


class CropCamConfig():
    def __init__(self, x, y, w, h, out_w, out_h, orig_w, orig_h):
        '''
        xy: left upper corner
        '''
        # assert x > 0 and x < orig_w
        # assert y > 0 and y < orig_h
        # assert w < orig_w and h < orig_h
        # assert x - w / 2 > 0 and x + w / 2 < orig_w
        # assert y - h / 2 > 0 and y + h / 2 < orig_h
        # assert h / w == out_h / out_w
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.out_w = out_w
        self.out_h = out_h
        self.orig_w = orig_w
        self.orig_h = orig_h

    def __str__(self):
        out = f'original image size(h,w): [{self.orig_h}, {self.orig_w}]\n'
        out += f'crop at(x,y):             [{self.x}, {self.y}]\n'
        out += f'crop size(h,w):           [{self.h}, {self.w}]\n'
        out += f'resize crop to(h,w):      [{self.out_h}, {self.out_w}]'
        return out


def fix_randomness(seed=42):
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def float_image_resize(img, shape, interp=PIL.Image.BILINEAR):
    missing_channel = False
    if len(img.shape) == 2:
        missing_channel = True
        img = img[..., None]
    layers = []
    img = img.transpose(2, 0, 1)
    for l in img:
        l = np.array(PIL.Image.fromarray(l).resize(shape[::-1], resample=interp))
        assert l.shape[:2] == shape
        layers.append(l)
    if missing_channel:
        return np.stack(layers, axis=-1)[..., 0]
    else:
        return np.stack(layers, axis=-1)


def is_nan(x):
    """
    get mask of nan values.
    :param x: torch or numpy var.
    :return: a N-D array of bool. True -> nan, False -> ok.
    """
    return x != x


def has_nan(x) -> bool:
    """
    check whether x contains nan.
    :param x: torch or numpy var.
    :return: single bool, True -> x containing nan, False -> ok.
    """
    if x is None:
        return False
    return is_nan(x).any()


def confirm(question='OK to continue?'):
    """
    Ask user to enter Y or N (case-insensitive).
    :return: True if the answer is Y.
    :rtype: bool
    """
    answer = ""
    while answer not in ["y", "n"]:
        answer = input(question + ' [y/n] ').lower()
    return answer == "y"


def print_notification(content_list, notification_type='NOTIFICATION'):
    print('---------------------- {0} ----------------------'.format(notification_type))
    print()
    for content in content_list:
        print(content)
    print()
    print('----------------------------------------------------')


def torch_img_to_np_img(torch_img):
    '''convert a torch image to matplotlib-able numpy image
    torch use Channels x Height x Width
    numpy use Height x Width x Channels
    Arguments:
        torch_img {[type]} -- [description]
    '''
    assert isinstance(torch_img, torch.Tensor), 'cannot process data type: {0}'.format(type(torch_img))
    if len(torch_img.shape) == 4 and (torch_img.shape[1] == 3 or torch_img.shape[1] == 1):
        return np.transpose(torch_img.detach().cpu().numpy(), (0, 2, 3, 1))
    if len(torch_img.shape) == 3 and (torch_img.shape[0] == 3 or torch_img.shape[0] == 1):
        return np.transpose(torch_img.detach().cpu().numpy(), (1, 2, 0))
    elif len(torch_img.shape) == 2:
        return torch_img.detach().cpu().numpy()
    else:
        raise ValueError('cannot process this image')


def np_img_to_torch_img(np_img):
    """convert a numpy image to torch image
    numpy use Height x Width x Channels
    torch use Channels x Height x Width

    Arguments:
        np_img {[type]} -- [description]
    """
    assert isinstance(np_img, np.ndarray), 'cannot process data type: {0}'.format(type(np_img))
    if len(np_img.shape) == 4 and (np_img.shape[3] == 3 or np_img.shape[3] == 1):
        return torch.from_numpy(np.transpose(np_img, (0, 3, 1, 2)))
    if len(np_img.shape) == 3 and (np_img.shape[2] == 3 or np_img.shape[2] == 1):
        return torch.from_numpy(np.transpose(np_img, (2, 0, 1)))
    elif len(np_img.shape) == 2:
        return torch.from_numpy(np_img)
    else:
        raise ValueError('cannot process this image with shape: {0}'.format(np_img.shape))


def safe_load_weights(model, saved_weights):
    try:
        model.load_state_dict(saved_weights)
    except RuntimeError:
        try:
            weights = saved_weights
            weights = {k.replace('module.', ''): v for k, v in weights.items()}
            model.load_state_dict(weights)
        except RuntimeError:
            try:
                weights = saved_weights
                weights = {'module.' + k: v for k, v in weights.items()}
                model.load_state_dict(weights)
            except RuntimeError:
                try:
                    pretrained_dict = saved_weights
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if ((k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape))}
                    assert len(pretrained_dict) != 0
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                    non_match_keys = set(model.state_dict().keys()) - set(pretrained_dict.keys())
                    notification = []
                    notification += ['pretrained weights PARTIALLY loaded, following are missing:']
                    notification += [str(non_match_keys)]
                    print_notification(notification, 'WARNING')
                except Exception as e:
                    print(f'pretrained weights loading failed {e}')
                    exit()
    print('weights safely loaded')


def visualize_corrs(img1, img2, corrs, mask=None):
    if mask is None:
        mask = np.ones(len(corrs)).astype(bool)

    scale1 = 1.0
    scale2 = 1.0
    if img1.shape[1] > img2.shape[1]:
        scale2 = img1.shape[1] / img2.shape[1]
        w = img1.shape[1]
    else:
        scale1 = img2.shape[1] / img1.shape[1]
        w = img2.shape[1]
    # Resize if too big
    max_w = 400
    if w > max_w:
        scale1 *= max_w / w
        scale2 *= max_w / w
    img1 = cv2.resize(img1, (0, 0), fx=scale1, fy=scale1)
    img2 = cv2.resize(img2, (0, 0), fx=scale2, fy=scale2)

    x1, x2 = corrs[:, :2], corrs[:, 2:]
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img = np.zeros((h1 + h2, max(w1, w2), 3), dtype=img1.dtype)
    img[:h1, :w1] = img1
    img[h1:, :w2] = img2
    # Move keypoints to coordinates to image coordinates
    x1 = x1 * scale1
    x2 = x2 * scale2
    # recompute the coordinates for the second image
    x2p = x2 + np.array([[0, h1]])
    fig = plt.figure(frameon=False)
    fig = plt.imshow(img)

    cols = [
        [0.0, 0.67, 0.0],
        [0.9, 0.1, 0.1],
    ]
    lw = .5
    alpha = 1

    # Draw outliers
    _x1 = x1[~mask]
    _x2p = x2p[~mask]
    xs = np.stack([_x1[:, 0], _x2p[:, 0]], axis=1).T
    ys = np.stack([_x1[:, 1], _x2p[:, 1]], axis=1).T
    plt.plot(
        xs, ys,
        alpha=alpha,
        linestyle="-",
        linewidth=lw,
        aa=False,
        color=cols[1],
    )
    

    # Draw Inliers
    _x1 = x1[mask]
    _x2p = x2p[mask]
    xs = np.stack([_x1[:, 0], _x2p[:, 0]], axis=1).T
    ys = np.stack([_x1[:, 1], _x2p[:, 1]], axis=1).T
    plt.plot(
        xs, ys,
        alpha=alpha,
        linestyle="-",
        linewidth=lw,
        aa=False,
        color=cols[0],
    )
    plt.scatter(xs, ys)

    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()
