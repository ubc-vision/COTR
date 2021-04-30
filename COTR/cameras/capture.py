'''
Capture from a pinhole camera
Separate the captured content and the camera...
'''

import os
import time
import abc
import copy

import cv2
import torch
import numpy as np
import imageio
from PIL import Image
from scipy import misc


def crop_center_max(img):
    if isinstance(img, torch.Tensor):
        return crop_center_max_torch(img)
    elif isinstance(img, np.ndarray):
        return crop_center_max_np(img)
    else:
        raise ValueError


def crop_center_max_torch(img):
    if len(img.shape) == 2:
        h, w = img.shape
    elif len(img.shape) == 3:
        c, h, w = img.shape
    elif len(img.shape) == 4:
        b, c, h, w = img.shape
    else:
        raise ValueError
    crop_x = min(h, w)
    crop_y = crop_x
    start_x = w // 2 - crop_x // 2
    start_y = h // 2 - crop_y // 2
    if len(img.shape) == 2:
        return img[start_y:start_y + crop_y, start_x:start_x + crop_x]
    elif len(img.shape) in [3, 4]:
        return img[..., start_y:start_y + crop_y, start_x:start_x + crop_x]


def crop_center_max_np(img, return_starts=False):
    if len(img.shape) == 2:
        h, w = img.shape
    elif len(img.shape) == 3:
        h, w, c = img.shape
    elif len(img.shape) == 4:
        b, h, w, c = img.shape
    else:
        raise ValueError
    crop_x = min(h, w)
    crop_y = crop_x
    start_x = w // 2 - crop_x // 2
    start_y = h // 2 - crop_y // 2
    if len(img.shape) == 2:
        canvas = img[start_y:start_y + crop_y, start_x:start_x + crop_x]
    elif len(img.shape) == 3:
        canvas = img[start_y:start_y + crop_y, start_x:start_x + crop_x, :]
    elif len(img.shape) == 4:
        canvas = img[:, start_y:start_y + crop_y, start_x:start_x + crop_x, :]
    if return_starts:
        return canvas, -start_x, -start_y
    else:
        return canvas


def pad_to_square_np(img, till_divisible_by=1, return_starts=False):
    if len(img.shape) == 2:
        h, w = img.shape
    elif len(img.shape) == 3:
        h, w, c = img.shape
    elif len(img.shape) == 4:
        b, h, w, c = img.shape
    else:
        raise ValueError
    if till_divisible_by == 1:
        size = max(h, w)
    else:
        size = (max(h, w) + till_divisible_by) - (max(h, w) % till_divisible_by)
    start_x = size // 2 - w // 2
    start_y = size // 2 - h // 2
    if len(img.shape) == 2:
        canvas = np.zeros([size, size], dtype=img.dtype)
        canvas[start_y:start_y + h, start_x:start_x + w] = img
    elif len(img.shape) == 3:
        canvas = np.zeros([size, size, c], dtype=img.dtype)
        canvas[start_y:start_y + h, start_x:start_x + w, :] = img
    elif len(img.shape) == 4:
        canvas = np.zeros([b, size, size, c], dtype=img.dtype)
        canvas[:, start_y:start_y + h, start_x:start_x + w, :] = img
    if return_starts:
        return canvas, start_x, start_y
    else:
        return canvas


def stretch_to_square_np(img):
    size = max(*img.shape[:2])
    return misc.imresize(img, (size, size), interp='bilinear')
