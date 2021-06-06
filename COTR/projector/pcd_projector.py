'''
a point cloud projector based on np
'''

import numpy as np

from COTR.utils import debug_utils, utils


def render_point_cloud_at_capture(point_cloud, capture, render_type='rgb', return_pcd=False):
    assert render_type in ['rgb', 'bw', 'depth']
    if render_type == 'rgb':
        assert point_cloud.shape[1] == 6
    else:
        point_cloud = point_cloud[:, :3]
        assert point_cloud.shape[1] == 3
    if render_type in ['bw', 'rgb']:
        keep_z = False
    else:
        keep_z = True

    pcd_2d = PointCloudProjector.pcd_3d_to_pcd_2d_np(point_cloud,
                                                     capture.intrinsic_mat,
                                                     capture.extrinsic_mat,
                                                     capture.size,
                                                     keep_z=True,
                                                     crop=True,
                                                     filter_neg=True,
                                                     norm_coord=False,
                                                     return_index=False)
    reproj = PointCloudProjector.pcd_2d_to_img_2d_np(pcd_2d,
                                                     capture.size,
                                                     has_z=True,
                                                     keep_z=keep_z)
    if return_pcd:
        return reproj, pcd_2d
    else:
        return reproj


def optical_flow_from_a_to_b(cap_a, cap_b):
    cap_a_intrinsic = cap_a.pinhole_cam.intrinsic_mat
    cap_a_img_size = cap_a.pinhole_cam.shape[:2]
    _h, _w = cap_b.pinhole_cam.shape[:2]
    x, y = np.meshgrid(
        np.linspace(0, _w - 1, num=_w),
        np.linspace(0, _h - 1, num=_h),
    )
    coord_map = np.concatenate([np.expand_dims(x, 2), np.expand_dims(y, 2)], axis=2)
    pcd_from_cap_b = cap_b.get_point_cloud_world_from_depth(coord_map)
    # pcd_from_cap_b = cap_b.point_cloud_world_w_feat(['pos', 'coord'])
    optical_flow = PointCloudProjector.pcd_2d_to_img_2d_np(PointCloudProjector.pcd_3d_to_pcd_2d_np(pcd_from_cap_b, cap_a_intrinsic, cap_a.cam_pose.world_to_camera[0:3, :], cap_a_img_size, keep_z=True, crop=True, filter_neg=True, norm_coord=False), cap_a_img_size, has_z=True, keep_z=False)
    return optical_flow


class PointCloudProjector():
    def __init__(self):
        pass

    @staticmethod
    def pcd_2d_to_pcd_3d_np(pcd, depth, intrinsic, motion=None, return_index=False):
        assert isinstance(pcd, np.ndarray), 'cannot process data type: {0}'.format(type(pcd))
        assert isinstance(intrinsic, np.ndarray), 'cannot process data type: {0}'.format(type(intrinsic))
        assert len(pcd.shape) == 2 and pcd.shape[1] >= 2
        assert len(depth.shape) == 2 and depth.shape[1] == 1
        assert intrinsic.shape == (3, 3)
        if motion is not None:
            assert isinstance(motion, np.ndarray), 'cannot process data type: {0}'.format(type(motion))
            assert motion.shape == (4, 4)
        # exec(debug_utils.embed_breakpoint())
        x, y, z = pcd[:, 0], pcd[:, 1], depth[:, 0]
        append_ones = np.ones_like(x)
        xyz = np.stack([x, y, append_ones], axis=1)  # shape: [num_points, 3]
        inv_intrinsic_mat = np.linalg.inv(intrinsic)
        xyz = np.matmul(inv_intrinsic_mat, xyz.T).T * z[..., None]
        valid_mask_1 = np.where(xyz[:, 2] > 0)
        xyz = xyz[valid_mask_1]

        if motion is not None:
            append_ones = np.ones_like(xyz[:, 0:1])
            xyzw = np.concatenate([xyz, append_ones], axis=1)
            xyzw = np.matmul(motion, xyzw.T).T
            valid_mask_2 = np.where(xyzw[:, 3] != 0)
            xyzw = xyzw[valid_mask_2]
            xyzw /= xyzw[:, 3:4]
            xyz = xyzw[:, 0:3]

        if pcd.shape[1] > 2:
            features = pcd[:, 2:]
            try:
                features = features[valid_mask_1][valid_mask_2]
            except UnboundLocalError:
                features = features[valid_mask_1]
            assert xyz.shape[0] == features.shape[0]
            xyz = np.concatenate([xyz, features], axis=1)
        if return_index:
            points_index = np.arange(pcd.shape[0])[valid_mask_1][valid_mask_2]
            return xyz, points_index
        return xyz

    @staticmethod
    def img_2d_to_pcd_3d_np(depth, intrinsic, img=None, motion=None):
        '''
        the function signature is not fully correct, because img is an optional
        if motion is None, the output pcd is in camera space
        if motion is camera_to_world, the out pcd is in world space.
        here the output is pure np array
        '''

        assert isinstance(depth, np.ndarray), 'cannot process data type: {0}'.format(type(depth))
        assert isinstance(intrinsic, np.ndarray), 'cannot process data type: {0}'.format(type(intrinsic))
        assert len(depth.shape) == 2
        assert intrinsic.shape == (3, 3)
        if img is not None:
            assert isinstance(img, np.ndarray), 'cannot process data type: {0}'.format(type(img))
            assert len(img.shape) == 3
            assert img.shape[:2] == depth.shape[:2], 'feature should have the same resolution as the depth'
        if motion is not None:
            assert isinstance(motion, np.ndarray), 'cannot process data type: {0}'.format(type(motion))
            assert motion.shape == (4, 4)

        pcd_image_space = PointCloudProjector.img_2d_to_pcd_2d_np(depth[..., None], norm_coord=False)
        valid_mask_1 = np.where(pcd_image_space[:, 2] > 0)
        pcd_image_space = pcd_image_space[valid_mask_1]
        xy = pcd_image_space[:, :2]
        z = pcd_image_space[:, 2:3]
        if img is not None:
            _c = img.shape[-1]
            feat = img.reshape(-1, _c)
            feat = feat[valid_mask_1]
            xy = np.concatenate([xy, feat], axis=1)
        pcd_3d = PointCloudProjector.pcd_2d_to_pcd_3d_np(xy, z, intrinsic, motion=motion)
        return pcd_3d

    @staticmethod
    def pcd_3d_to_pcd_2d_np(pcd, intrinsic, extrinsic, size, keep_z: bool, crop: bool = True, filter_neg: bool = True, norm_coord: bool = True, return_index: bool = False):
        assert isinstance(pcd, np.ndarray), 'cannot process data type: {0}'.format(type(pcd))
        assert isinstance(intrinsic, np.ndarray), 'cannot process data type: {0}'.format(type(intrinsic))
        assert isinstance(extrinsic, np.ndarray), 'cannot process data type: {0}'.format(type(extrinsic))
        assert len(pcd.shape) == 2 and pcd.shape[1] >= 3, 'seems the input pcd is not a valid 3d point cloud: {0}'.format(pcd.shape)

        xyzw = np.concatenate([pcd[:, 0:3], np.ones_like(pcd[:, 0:1])], axis=1)
        mvp_mat = np.matmul(intrinsic, extrinsic)
        camera_points = np.matmul(mvp_mat, xyzw.T).T
        if filter_neg:
            valid_mask_1 = camera_points[:, 2] > 0.0
        else:
            valid_mask_1 = np.ones_like(camera_points[:, 2], dtype=bool)
        camera_points = camera_points[valid_mask_1]
        image_points = camera_points / camera_points[:, 2:3]
        image_points = image_points[:, :2]
        if crop:
            valid_mask_2 = (image_points[:, 0] >= 0) * (image_points[:, 0] < size[1] - 1) * (image_points[:, 1] >= 0) * (image_points[:, 1] < size[0] - 1)
        else:
            valid_mask_2 = np.ones_like(image_points[:, 0], dtype=bool)
        if norm_coord:
            image_points = ((image_points / size[::-1]) * 2) - 1

        if keep_z:
            image_points = np.concatenate([image_points[valid_mask_2], camera_points[valid_mask_2][:, 2:3], pcd[valid_mask_1][:, 3:][valid_mask_2]], axis=1)
        else:
            image_points = np.concatenate([image_points[valid_mask_2], pcd[valid_mask_1][:, 3:][valid_mask_2]], axis=1)
        # if filter_neg and crop:
        #     exec(debug_utils.embed_breakpoint('pcd_3d_to_pcd_2d_np'))
        if return_index:
            points_index = np.arange(pcd.shape[0])[valid_mask_1][valid_mask_2]
            return image_points, points_index
        return image_points

    @staticmethod
    def pcd_2d_to_img_2d_np(pcd, size, has_z=False, keep_z=False):
        assert len(pcd.shape) == 2 and pcd.shape[-1] >= 2, 'seems the input pcd is not a valid point cloud: {0}'.format(pcd.shape)
        # assert 0, 'pass Z values in'
        if has_z:
            pcd = pcd[pcd[:, 2].argsort()[::-1]]
            if not keep_z:
                pcd = np.delete(pcd, [2], axis=1)
        index_list = np.round(pcd[:, 0:2]).astype(np.int32)
        index_list[:, 0] = np.clip(index_list[:, 0], 0, size[1] - 1)
        index_list[:, 1] = np.clip(index_list[:, 1], 0, size[0] - 1)
        _h, _w, _c = *size, pcd.shape[-1] - 2
        if _c == 0:
            canvas = np.zeros((_h, _w, 1))
            canvas[index_list[:, 1], index_list[:, 0]] = 1.0
        else:
            canvas = np.zeros((_h, _w, _c))
            canvas[index_list[:, 1], index_list[:, 0]] = pcd[:, 2:]

        return canvas

    @staticmethod
    def img_2d_to_pcd_2d_np(img, norm_coord=True):
        assert isinstance(img, np.ndarray), 'cannot process data type: {0}'.format(type(img))
        assert len(img.shape) == 3

        _h, _w, _c = img.shape
        if norm_coord:
            x, y = np.meshgrid(
                np.linspace(-1, 1, num=_w),
                np.linspace(-1, 1, num=_h),
            )
        else:
            x, y = np.meshgrid(
                np.linspace(0, _w - 1, num=_w),
                np.linspace(0, _h - 1, num=_h),
            )
        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        feat = img.reshape(-1, _c)
        pcd_2d = np.concatenate([x, y, feat], axis=1)
        return pcd_2d
