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
import PIL
from PIL import Image
import tables

from COTR.cameras.camera_pose import CameraPose, rotate_camera_pose
from COTR.cameras.pinhole_camera import PinholeCamera, rotate_pinhole_camera, crop_pinhole_camera
from COTR.utils import debug_utils, utils, constants
from COTR.utils.utils import Point2D
from COTR.projector import pcd_projector
from COTR.utils.constants import MAX_SIZE
from COTR.utils.utils import CropCamConfig


def crop_center_max_xy(p2d, shape):
    h, w = shape
    crop_x = min(h, w)
    crop_y = crop_x
    start_x = w // 2 - crop_x // 2
    start_y = h // 2 - crop_y // 2
    mask = (p2d.xy[:, 0] > start_x) & (p2d.xy[:, 0] < start_x + crop_x) & (p2d.xy[:, 1] > start_y) & (p2d.xy[:, 1] < start_y + crop_y)
    out_xy = (p2d.xy - [start_x, start_y])[mask]
    out = Point2D(p2d.id_3d[mask], out_xy)
    return out


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
    return np.array(PIL.Image.fromarray(img).resize((size, size), resample=PIL.Image.BILINEAR))


def rotate_image(image, angle, interpolation=cv2.INTER_LINEAR):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=interpolation)
    return result


def read_array(path):
    '''
    https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py
    '''
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


################ Content ################


class CapturedContent(abc.ABC):
    def __init__(self):
        self._rotation = 0

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, rot):
        self._rotation = rot


class CapturedImage(CapturedContent):
    def __init__(self, img_path, crop_cam, pinhole_cam_before=None):
        super(CapturedImage, self).__init__()
        assert os.path.isfile(img_path), 'file does not exist: {0}'.format(img_path)
        self.crop_cam = crop_cam
        self._image = None
        self.img_path = img_path
        self.pinhole_cam_before = pinhole_cam_before
        self._p2d = None

    def read_image_to_ram(self) -> int:
        # raise NotImplementedError
        assert self._image is None
        _image = self.image
        self._image = _image
        return self._image.nbytes

    @property
    def image(self):
        if self._image is not None:
            _image = self._image
        else:
            _image = imageio.imread(self.img_path, pilmode='RGB')
            if self.rotation != 0:
                _image = rotate_image(_image, self.rotation)
            if _image.shape[:2] != self.pinhole_cam_before.shape:
                _image = np.array(PIL.Image.fromarray(_image).resize(self.pinhole_cam_before.shape[::-1], resample=PIL.Image.BILINEAR))
                assert _image.shape[:2] == self.pinhole_cam_before.shape
            if self.crop_cam == 'no_crop':
                pass
            elif self.crop_cam == 'crop_center':
                _image = crop_center_max(_image)
            elif self.crop_cam == 'crop_center_and_resize':
                _image = crop_center_max(_image)
                _image = np.array(PIL.Image.fromarray(_image).resize((MAX_SIZE, MAX_SIZE), resample=PIL.Image.BILINEAR))
            elif isinstance(self.crop_cam, CropCamConfig):
                assert _image.shape[0] == self.crop_cam.orig_h
                assert _image.shape[1] == self.crop_cam.orig_w
                _image = _image[self.crop_cam.y:self.crop_cam.y + self.crop_cam.h,
                                self.crop_cam.x:self.crop_cam.x + self.crop_cam.w, ]
                _image = np.array(PIL.Image.fromarray(_image).resize((self.crop_cam.out_w, self.crop_cam.out_h), resample=PIL.Image.BILINEAR))
                assert _image.shape[:2] == (self.crop_cam.out_h, self.crop_cam.out_w)
            else:
                raise ValueError()
        return _image

    @property
    def p2d(self):
        if self._p2d is None:
            return self._p2d
        else:
            _p2d = self._p2d
            if self.crop_cam == 'no_crop':
                pass
            elif self.crop_cam == 'crop_center':
                _p2d = crop_center_max_xy(_p2d, self.pinhole_cam_before.shape)
            else:
                raise ValueError()
        return _p2d

    @p2d.setter
    def p2d(self, value):
        if value is not None:
            assert isinstance(value, Point2D)
        self._p2d = value


class CapturedDepth(CapturedContent):
    def __init__(self, depth_path, crop_cam, pinhole_cam_before=None):
        super(CapturedDepth, self).__init__()
        if not depth_path.endswith('dummy'):
            assert os.path.isfile(depth_path), 'file does not exist: {0}'.format(depth_path)
        self.crop_cam = crop_cam
        self._depth = None
        self.depth_path = depth_path
        self.pinhole_cam_before = pinhole_cam_before

    def read_depth(self):
        if self.depth_path.endswith('dummy'):
            image_path = self.depth_path[:-5]
            w, h = Image.open(image_path).size
            _depth = np.zeros([h, w], dtype=np.float32)
        elif self.depth_path.endswith('.h5'):
            depth_h5 = tables.open_file(self.depth_path, mode='r')
            _depth = np.array(depth_h5.root.depth)
            depth_h5.close()
        else:
            raise ValueError
        return _depth.astype(np.float32)

    def read_depth_to_ram(self) -> int:
        # raise NotImplementedError
        assert self._depth is None
        _depth = self.depth_map
        self._depth = _depth
        return self._depth.nbytes

    @property
    def depth_map(self):
        if self._depth is not None:
            _depth = self._depth
        else:
            _depth = self.read_depth()
            if self.rotation != 0:
                _depth = rotate_image(_depth, self.rotation, interpolation=cv2.INTER_NEAREST)
            if _depth.shape != self.pinhole_cam_before.shape:
                _depth = np.array(PIL.Image.fromarray(_depth).resize(self.pinhole_cam_before.shape[::-1], resample=PIL.Image.NEAREST))
                assert _depth.shape[:2] == self.pinhole_cam_before.shape
            if self.crop_cam == 'no_crop':
                pass
            elif self.crop_cam == 'crop_center':
                _depth = crop_center_max(_depth)
            elif self.crop_cam == 'crop_center_and_resize':
                _depth = crop_center_max(_depth)
                _depth = np.array(PIL.Image.fromarray(_depth).resize((MAX_SIZE, MAX_SIZE), resample=PIL.Image.NEAREST))
            elif isinstance(self.crop_cam, CropCamConfig):
                assert _depth.shape[0] == self.crop_cam.orig_h
                assert _depth.shape[1] == self.crop_cam.orig_w
                _depth = _depth[self.crop_cam.y:self.crop_cam.y + self.crop_cam.h,
                                self.crop_cam.x:self.crop_cam.x + self.crop_cam.w, ]
                _depth = np.array(PIL.Image.fromarray(_depth).resize((self.crop_cam.out_w, self.crop_cam.out_h), resample=PIL.Image.NEAREST))
                assert _depth.shape[:2] == (self.crop_cam.out_h, self.crop_cam.out_w)
            else:
                raise ValueError()
        assert (_depth >= 0).all()
        return _depth


################ Pinhole Capture ################
class BasePinholeCapture():
    def __init__(self, pinhole_cam, cam_pose, crop_cam):
        self.crop_cam = crop_cam
        self.cam_pose = cam_pose
        # modify the camera instrinsics
        self.pinhole_cam = crop_pinhole_camera(pinhole_cam, crop_cam)
        self.pinhole_cam_before = pinhole_cam

    def __str__(self):
        string = 'pinhole camera: {0}\ncamera pose: {1}'.format(self.pinhole_cam, self.cam_pose)
        return string

    @property
    def intrinsic_mat(self):
        return self.pinhole_cam.intrinsic_mat

    @property
    def extrinsic_mat(self):
        return self.cam_pose.extrinsic_mat

    @property
    def shape(self):
        return self.pinhole_cam.shape

    @property
    def size(self):
        return self.shape

    @property
    def mvp_mat(self):
        '''
        model-view-projection matrix (naming from opengl)
        '''
        return np.matmul(self.pinhole_cam.intrinsic_mat, self.cam_pose.world_to_camera_3x4)


class RGBPinholeCapture(BasePinholeCapture):
    def __init__(self, img_path, pinhole_cam, cam_pose, crop_cam):
        BasePinholeCapture.__init__(self, pinhole_cam, cam_pose, crop_cam)
        self.captured_image = CapturedImage(img_path, crop_cam, self.pinhole_cam_before)

    def read_image_to_ram(self) -> int:
        return self.captured_image.read_image_to_ram()

    @property
    def img_path(self):
        return self.captured_image.img_path

    @property
    def image(self):
        _image = self.captured_image.image
        assert _image.shape[0:2] == self.pinhole_cam.shape, 'image shape: {0}, pinhole camera: {1}'.format(_image.shape, self.pinhole_cam)
        return _image

    @property
    def seq_id(self):
        return os.path.dirname(self.captured_image.img_path)

    @property
    def p2d(self):
        return self.captured_image.p2d

    @p2d.setter
    def p2d(self, value):
        self.captured_image.p2d = value


class DepthPinholeCapture(BasePinholeCapture):
    def __init__(self, depth_path, pinhole_cam, cam_pose, crop_cam):
        BasePinholeCapture.__init__(self, pinhole_cam, cam_pose, crop_cam)
        self.captured_depth = CapturedDepth(depth_path, crop_cam, self.pinhole_cam_before)

    def read_depth_to_ram(self) -> int:
        return self.captured_depth.read_depth_to_ram()

    @property
    def depth_path(self):
        return self.captured_depth.depth_path

    @property
    def depth_map(self):
        _depth = self.captured_depth.depth_map
        # if self.pinhole_cam.shape != _depth.shape:
        #     _depth = misc.imresize(_depth, self.pinhole_cam.shape, interp='nearest', mode='F')
        assert (_depth >= 0).all()
        return _depth

    @property
    def point_cloud_world(self):
        return self.get_point_cloud_world_from_depth(feat_map=None)

    def get_point_cloud_world_from_depth(self, feat_map=None):
        _pcd = pcd_projector.PointCloudProjector.img_2d_to_pcd_3d_np(self.depth_map, self.pinhole_cam.intrinsic_mat, img=feat_map, motion=self.cam_pose.camera_to_world).astype(constants.DEFAULT_PRECISION)
        return _pcd


class RGBDPinholeCapture(RGBPinholeCapture, DepthPinholeCapture):
    def __init__(self, img_path, depth_path, pinhole_cam, cam_pose, crop_cam):
        RGBPinholeCapture.__init__(self, img_path, pinhole_cam, cam_pose, crop_cam)
        DepthPinholeCapture.__init__(self, depth_path, pinhole_cam, cam_pose, crop_cam)

    @property
    def point_cloud_w_rgb_world(self):
        return self.get_point_cloud_world_from_depth(feat_map=self.image)


def rotate_capture(cap, rot):
    if rot == 0:
        return copy.deepcopy(cap)
    else:
        rot_pose = rotate_camera_pose(cap.cam_pose, rot)
        rot_cap = copy.deepcopy(cap)
        rot_cap.cam_pose = rot_pose
        if hasattr(rot_cap, 'captured_image'):
            rot_cap.captured_image.rotation = rot
        if hasattr(rot_cap, 'captured_depth'):
            rot_cap.captured_depth.rotation = rot
        return rot_cap


def crop_capture(cap, crop_cam):
    if isinstance(cap, RGBDPinholeCapture):
        cropped_cap = RGBDPinholeCapture(cap.img_path, cap.depth_path, cap.pinhole_cam, cap.cam_pose, crop_cam)
    elif isinstance(cap, RGBPinholeCapture):
        cropped_cap = RGBPinholeCapture(cap.img_path, cap.pinhole_cam, cap.cam_pose, crop_cam)
    else:
        raise ValueError
    if hasattr(cropped_cap, 'captured_image'):
        cropped_cap.captured_image.rotation = cap.captured_image.rotation
    if hasattr(cropped_cap, 'captured_depth'):
        cropped_cap.captured_depth.rotation = cap.captured_depth.rotation
    return cropped_cap
