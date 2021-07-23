import sys
assert sys.version_info >= (3, 7), 'ordered dict is required'
import os
import re
from collections import namedtuple
import json

import numpy as np
from tqdm import tqdm

from COTR.utils import debug_utils
from COTR.cameras.pinhole_camera import PinholeCamera
from COTR.cameras.camera_pose import CameraPose
from COTR.cameras.capture import RGBPinholeCapture, RGBDPinholeCapture
from COTR.cameras import capture
from COTR.transformations import transformations
from COTR.transformations.transform_basics import Translation, Rotation
from COTR.sfm_scenes import sfm_scenes
from COTR.global_configs import dataset_config
from COTR.utils.utils import Point2D, Point3D

ImageMeta = namedtuple('ImageMeta', ['image_id', 'r', 't', 'camera_id', 'image_path', 'point3d_id', 'p2d'])
COVISIBILITY_CHECK = False
LOAD_PCD = False


class ColmapAsciiReader():
    def __init__(self):
        pass

    @classmethod
    def read_sfm_scene(cls, scene_dir, images_dir, crop_cam):
        point_cloud_path = os.path.join(scene_dir, 'points3D.txt')
        cameras_path = os.path.join(scene_dir, 'cameras.txt')
        images_path = os.path.join(scene_dir, 'images.txt')
        captures = cls.read_captures(images_path, cameras_path, images_dir, crop_cam)
        if LOAD_PCD:
            point_cloud = cls.read_point_cloud(point_cloud_path)
        else:
            point_cloud = None
        sfm_scene = sfm_scenes.SfmScene(captures, point_cloud)
        return sfm_scene

    @staticmethod
    def read_point_cloud(points_txt_path):
        with open(points_txt_path, "r") as fid:
            line = fid.readline()
            assert line == '# 3D point list with one line of data per point:\n'
            line = fid.readline()
            assert line == '#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n'
            line = fid.readline()
            assert re.search('^# Number of points: \d+, mean track length: [-+]?\d*\.\d+|\d+\n$', line)
            num_points, mean_track_length = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            num_points = int(num_points)
            mean_track_length = float(mean_track_length)

            xyz = np.zeros((num_points, 3), dtype=np.float32)
            rgb = np.zeros((num_points, 3), dtype=np.float32)
            if COVISIBILITY_CHECK:
                point_meta = {}

            for i in tqdm(range(num_points), desc='reading point cloud'):
                elems = fid.readline().split()
                xyz[i] = list(map(float, elems[1:4]))
                rgb[i] = list(map(int, elems[4:7]))
                if COVISIBILITY_CHECK:
                    point_id = int(elems[0])
                    image_ids = np.array(tuple(map(int, elems[8::2])))
                    point_meta[point_id] = Point3D(id=point_id,
                                                   arr_idx=i,
                                                   image_ids=image_ids)
            pcd = np.concatenate([xyz, rgb], axis=1)
        if COVISIBILITY_CHECK:
            return pcd, point_meta
        else:
            return pcd

    @classmethod
    def read_captures(cls, images_txt_path, cameras_txt_path, images_dir, crop_cam):
        captures = []
        cameras = cls.read_cameras(cameras_txt_path)
        images_meta = cls.read_images_meta(images_txt_path, images_dir)
        for key in images_meta.keys():
            cur_cam_id = images_meta[key].camera_id
            cur_cam = cameras[cur_cam_id]
            cur_camera_pose = CameraPose(images_meta[key].t, images_meta[key].r)
            cur_image_path = images_meta[key].image_path
            cap = RGBPinholeCapture(cur_image_path, cur_cam, cur_camera_pose, crop_cam)
            captures.append(cap)
        return captures

    @classmethod
    def read_cameras(cls, cameras_txt_path):
        cameras = {}
        with open(cameras_txt_path, "r") as fid:
            line = fid.readline()
            assert line == '# Camera list with one line of data per camera:\n'
            line = fid.readline()
            assert line == '#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n'
            line = fid.readline()
            assert re.search('^# Number of cameras: \d+\n$', line)
            num_cams = int(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])

            for _ in tqdm(range(num_cams), desc='reading cameras'):
                elems = fid.readline().split()
                camera_id = int(elems[0])
                camera_type = elems[1]
                if camera_type == "PINHOLE":
                    width, height, focal_length_x, focal_length_y, cx, cy = list(map(float, elems[2:8]))
                else:
                    raise ValueError('Please rectify the 3D model to pinhole camera.')
                cur_cam = PinholeCamera(width, height, focal_length_x, focal_length_y, cx, cy)
                assert camera_id not in cameras
                cameras[camera_id] = cur_cam
        return cameras

    @classmethod
    def read_images_meta(cls, images_txt_path, images_dir):
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
                elems = fid.readline().split()
                assert len(elems) == 10

                image_path = os.path.join(images_dir, elems[9])
                assert os.path.isfile(image_path)
                image_id = int(elems[0])
                qw, qx, qy, qz, tx, ty, tz = list(map(float, elems[1:8]))
                t = Translation(np.array([tx, ty, tz], dtype=np.float32))
                r = Rotation(np.array([qw, qx, qy, qz], dtype=np.float32))
                camera_id = int(elems[8])
                assert image_id not in images_meta

                line = fid.readline()
                if COVISIBILITY_CHECK:
                    elems = line.split()
                    elems = list(map(float, elems))
                    elems = np.array(elems).reshape(-1, 3)
                    point3d_id = set(elems[elems[:, 2] != -1][:, 2].astype(np.int))
                    point3d_id = np.sort(np.array(list(point3d_id)))
                    xyi = elems[elems[:, 2] != -1]
                    xy = xyi[:, :2]
                    idx = xyi[:, 2].astype(np.int)
                    p2d = Point2D(idx, xy)
                else:
                    point3d_id = None
                    p2d = None

                images_meta[image_id] = ImageMeta(image_id, r, t, camera_id, image_path, point3d_id, p2d)
        return images_meta


class ColmapWithDepthAsciiReader(ColmapAsciiReader):
    '''
    Not all images have usable depth estimate from colmap.
    A valid list is needed.
    '''

    @classmethod
    def read_sfm_scene(cls, scene_dir, images_dir, depth_dir, crop_cam):
        point_cloud_path = os.path.join(scene_dir, 'points3D.txt')
        cameras_path = os.path.join(scene_dir, 'cameras.txt')
        images_path = os.path.join(scene_dir, 'images.txt')
        captures = cls.read_captures(images_path, cameras_path, images_dir, depth_dir, crop_cam)
        if LOAD_PCD:
            point_cloud = cls.read_point_cloud(point_cloud_path)
        else:
            point_cloud = None
        sfm_scene = sfm_scenes.SfmScene(captures, point_cloud)
        return sfm_scene

    @classmethod
    def read_sfm_scene_given_valid_list_path(cls, scene_dir, images_dir, depth_dir, valid_list_json_path, crop_cam):
        point_cloud_path = os.path.join(scene_dir, 'points3D.txt')
        cameras_path = os.path.join(scene_dir, 'cameras.txt')
        images_path = os.path.join(scene_dir, 'images.txt')
        valid_list = cls.read_valid_list(valid_list_json_path)
        captures = cls.read_captures_with_depth_given_valid_list(images_path, cameras_path, images_dir, depth_dir, valid_list, crop_cam)
        if LOAD_PCD:
            point_cloud = cls.read_point_cloud(point_cloud_path)
        else:
            point_cloud = None
        sfm_scene = sfm_scenes.SfmScene(captures, point_cloud)
        return sfm_scene

    @classmethod
    def read_captures(cls, images_txt_path, cameras_txt_path, images_dir, depth_dir, crop_cam):
        captures = []
        cameras = cls.read_cameras(cameras_txt_path)
        images_meta = cls.read_images_meta(images_txt_path, images_dir)
        for key in images_meta.keys():
            cur_cam_id = images_meta[key].camera_id
            cur_cam = cameras[cur_cam_id]
            cur_camera_pose = CameraPose(images_meta[key].t, images_meta[key].r)
            cur_image_path = images_meta[key].image_path
            try:
                cur_depth_path = cls.image_path_2_depth_path(cur_image_path[len(images_dir) + 1:], depth_dir)
            except:
                print('{0} does not have depth at {1}'.format(cur_image_path, depth_dir))
                # TODO
                # continue
                # exec(debug_utils.embed_breakpoint())
                cur_depth_path = f'{cur_image_path}dummy'

            cap = RGBDPinholeCapture(cur_image_path, cur_depth_path, cur_cam, cur_camera_pose, crop_cam)
            cap.point3d_id = images_meta[key].point3d_id
            cap.p2d = images_meta[key].p2d
            cap.image_id = key
            captures.append(cap)
        return captures

    @classmethod
    def read_captures_with_depth_given_valid_list(cls, images_txt_path, cameras_txt_path, images_dir, depth_dir, valid_list, crop_cam):
        captures = []
        cameras = cls.read_cameras(cameras_txt_path)
        images_meta = cls.read_images_meta_given_valid_list(images_txt_path, images_dir, valid_list)
        for key in images_meta.keys():
            cur_cam_id = images_meta[key].camera_id
            cur_cam = cameras[cur_cam_id]
            cur_camera_pose = CameraPose(images_meta[key].t, images_meta[key].r)
            cur_image_path = images_meta[key].image_path
            try:
                cur_depth_path = cls.image_path_2_depth_path(cur_image_path, depth_dir)
            except:
                print('{0} does not have depth at {1}'.format(cur_image_path, depth_dir))
                continue
            cap = RGBDPinholeCapture(cur_image_path, cur_depth_path, cur_cam, cur_camera_pose, crop_cam)
            cap.point3d_id = images_meta[key].point3d_id
            cap.p2d = images_meta[key].p2d
            cap.image_id = key
            captures.append(cap)
        return captures

    @classmethod
    def read_images_meta_given_valid_list(cls, images_txt_path, images_dir, valid_list):
        images_meta = {}
        with open(images_txt_path, "r") as fid:
            line = fid.readline()
            assert line == '# Image list with two lines of data per image:\n'
            line = fid.readline()
            assert line == '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n'
            line = fid.readline()
            assert line == '#   POINTS2D[] as (X, Y, POINT3D_ID)\n'
            line = fid.readline()
            assert re.search('^# Number of images: \d+, mean observations per image:[-+]?\d*\.\d+|\d+\n$', line), line
            num_images, mean_ob_per_img = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            num_images = int(num_images)
            mean_ob_per_img = float(mean_ob_per_img)

            for _ in tqdm(range(num_images), desc='reading images meta'):
                elems = fid.readline().split()
                assert len(elems) == 10
                line = fid.readline()
                image_path = os.path.join(images_dir, elems[9])
                prefix = os.path.abspath(os.path.join(image_path, '../../../../')) + '/'
                rel_image_path = image_path.replace(prefix, '')
                if rel_image_path not in valid_list:
                    continue
                assert os.path.isfile(image_path), '{0} is not existing'.format(image_path)
                image_id = int(elems[0])
                qw, qx, qy, qz, tx, ty, tz = list(map(float, elems[1:8]))
                t = Translation(np.array([tx, ty, tz], dtype=np.float32))
                r = Rotation(np.array([qw, qx, qy, qz], dtype=np.float32))
                camera_id = int(elems[8])
                assert image_id not in images_meta

                if COVISIBILITY_CHECK:
                    elems = line.split()
                    elems = list(map(float, elems))
                    elems = np.array(elems).reshape(-1, 3)
                    point3d_id = set(elems[elems[:, 2] != -1][:, 2].astype(np.int))
                    point3d_id = np.sort(np.array(list(point3d_id)))
                    xyi = elems[elems[:, 2] != -1]
                    xy = xyi[:, :2]
                    idx = xyi[:, 2].astype(np.int)
                    p2d = Point2D(idx, xy)
                else:
                    point3d_id = None
                    p2d = None
                images_meta[image_id] = ImageMeta(image_id, r, t, camera_id, image_path, point3d_id, p2d)
        return images_meta

    @classmethod
    def read_valid_list(cls, valid_list_json_path):
        assert os.path.isfile(valid_list_json_path), valid_list_json_path
        with open(valid_list_json_path, 'r') as f:
            valid_list = json.load(f)
        assert len(valid_list) == len(set(valid_list))
        return set(valid_list)

    @classmethod
    def image_path_2_depth_path(cls, image_path, depth_dir):
        depth_file = os.path.splitext(os.path.basename(image_path))[0] + '.h5'
        depth_path = os.path.join(depth_dir, depth_file)
        if not os.path.isfile(depth_path):
            # depth_file = image_path + '.photometric.bin'
            depth_file = image_path + '.geometric.bin'
            depth_path = os.path.join(depth_dir, depth_file)
        assert os.path.isfile(depth_path), '{0} is not file'.format(depth_path)
        return depth_path
