"""
Static pinhole camera
"""

import copy

import numpy as np

from COTR.utils import constants
from COTR.utils.constants import MAX_SIZE
from COTR.utils.utils import CropCamConfig


class PinholeCamera():
    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = int(width)
        self.height = int(height)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def __str__(self):
        string = 'width: {0}, height: {1}, fx: {2}, fy: {3}, cx: {4}, cy: {5}'.format(self.width, self.height, self.fx, self.fy, self.cx, self.cy)
        return string

    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def intrinsic_mat(self):
        mat = np.array([[self.fx, 0.0, self.cx],
                        [0.0, self.fy, self.cy],
                        [0.0, 0.0, 1.0]], dtype=constants.DEFAULT_PRECISION)
        return mat


def rotate_pinhole_camera(cam, rot):
    assert 0, 'TODO: Camera should stay the same while rotation'
    assert rot in [0, 90, 180, 270], 'only support 0/90/180/270 degrees rotation'
    if rot in [0, 180]:
        return copy.deepcopy(cam)
    elif rot in [90, 270]:
        return PinholeCamera(width=cam.height, height=cam.width, fx=cam.fy, fy=cam.fx, cx=cam.cy, cy=cam.cx)
    else:
        raise NotImplementedError


def crop_pinhole_camera(pinhole_cam, crop_cam):
    if crop_cam == 'no_crop':
        cropped_pinhole_cam = pinhole_cam
    elif crop_cam == 'crop_center':
        _h = _w = min(*pinhole_cam.shape)
        _cx = _cy = _h / 2
        cropped_pinhole_cam = PinholeCamera(_w, _h, pinhole_cam.fx, pinhole_cam.fy, _cx, _cy)
    elif crop_cam == 'crop_center_and_resize':
        _h = _w = MAX_SIZE
        _cx = _cy = MAX_SIZE / 2
        scale = MAX_SIZE / min(*pinhole_cam.shape)
        cropped_pinhole_cam = PinholeCamera(_w, _h, pinhole_cam.fx * scale, pinhole_cam.fy * scale, _cx, _cy)
    elif isinstance(crop_cam, CropCamConfig):
        scale = crop_cam.out_h / crop_cam.h
        cropped_pinhole_cam = PinholeCamera(crop_cam.out_w,
                                            crop_cam.out_h,
                                            pinhole_cam.fx * scale,
                                            pinhole_cam.fy * scale,
                                            (pinhole_cam.cx - crop_cam.x) * scale,
                                            (pinhole_cam.cy - crop_cam.y) * scale
                                            )
    else:
        raise ValueError
    return cropped_pinhole_cam
