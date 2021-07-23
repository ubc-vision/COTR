'''
Extrinsic camera pose
'''
import math
import copy

import numpy as np

from COTR.transformations import transformations
from COTR.transformations.transform_basics import Translation, Rotation, UnstableRotation


class CameraPose():
    def __init__(self, t: Translation, r: Rotation):
        '''
        WARN: World 2 cam
        Translation and rotation are world to camera
        translation_vector is not the coordinate of the camera in world space.
        '''
        assert isinstance(t, Translation)
        assert isinstance(r, Rotation) or isinstance(r, UnstableRotation)
        self.t = t
        self.r = r

    def __str__(self):
        string = f'center in world: {self.camera_center_in_world}, translation(w2c): {self.t}, rotation(w2c): {self.r}'
        return string

    @classmethod
    def from_world_to_camera(cls, world_to_camera, unstable=False):
        assert isinstance(world_to_camera, np.ndarray)
        assert world_to_camera.shape == (4, 4)
        vec = transformations.translation_from_matrix(world_to_camera).astype(np.float32)
        t = Translation(vec)
        if unstable:
            r = UnstableRotation(world_to_camera)
        else:
            quat = transformations.quaternion_from_matrix(world_to_camera).astype(np.float32)
            r = Rotation(quat)
        return cls(t, r)

    @classmethod
    def from_camera_to_world(cls, camera_to_world, unstable=False):
        assert isinstance(camera_to_world, np.ndarray)
        assert camera_to_world.shape == (4, 4)
        world_to_camera = np.linalg.inv(camera_to_world)
        world_to_camera /= world_to_camera[3, 3]
        return cls.from_world_to_camera(world_to_camera, unstable)

    @classmethod
    def from_pose_vector(cls, pose_vector):
        t = Translation(pose_vector[:3])
        r = Rotation(pose_vector[3:])
        return cls(t, r)

    @property
    def translation_vector(self):
        return self.t.translation_vector

    @property
    def translation_matrix(self):
        return self.t.translation_matrix

    @property
    def quaternion(self):
        '''
        quaternion format (w, x, y, z)
        '''
        return self.r.quaternion

    @property
    def rotation_matrix(self):
        return self.r.rotation_matrix

    @property
    def pose_vector(self):
        '''
        Pose vector is a concat of translation vector and quaternion vector
        (X, Y, Z, w, x, y, z)
        w2c
        '''
        return np.concatenate([self.translation_vector, self.quaternion])

    @property
    def inv_pose_vector(self):
        inv_quat = transformations.quaternion_inverse(self.quaternion)
        return np.concatenate([self.camera_center_in_world, inv_quat])

    @property
    def pose_vector_6_dof(self):
        '''
        Here we assuming the quaternion is normalized and we remove the W component
        (X, Y, Z, x, y, z)
        '''
        return np.concatenate([self.translation_vector, self.quaternion[1:]])

    @property
    def world_to_camera(self):
        M = np.matmul(self.translation_matrix, self.rotation_matrix)
        M /= M[3, 3]
        return M

    @property
    def world_to_camera_3x4(self):
        M = self.world_to_camera
        M = M[0:3, 0:4]
        return M

    @property
    def extrinsic_mat(self):
        return self.world_to_camera_3x4

    @property
    def camera_to_world(self):
        M = np.linalg.inv(self.world_to_camera)
        M /= M[3, 3]
        return M

    @property
    def camera_to_world_3x4(self):
        M = self.camera_to_world
        M = M[0:3, 0:4]
        return M

    @property
    def camera_center_in_world(self):
        return self.camera_to_world[:3, 3]

    @property
    def forward(self):
        return self.camera_to_world[:3, 2]

    @property
    def up(self):
        return self.camera_to_world[:3, 1]

    @property
    def right(self):
        return self.camera_to_world[:3, 0]

    @property
    def essential_matrix(self):
        E = np.cross(self.rotation_matrix[:3, :3], self.camera_center_in_world)
        return E / np.linalg.norm(E)


def inverse_camera_pose(cam_pose: CameraPose):
    return CameraPose.from_world_to_camera(np.linalg.inv(cam_pose.world_to_camera))


def rotate_camera_pose(cam_pose, rot):
    if rot == 0:
        return copy.deepcopy(cam_pose)
    else:
        rot = rot / 180 * np.pi
        sin_rot = np.sin(rot)
        cos_rot = np.cos(rot)

        rot_mat = np.stack([np.stack([cos_rot, -sin_rot, 0, 0], axis=-1),
                            np.stack([sin_rot, cos_rot, 0, 0], axis=-1),
                            np.stack([0, 0, 1, 0], axis=-1),
                            np.stack([0, 0, 0, 1], axis=-1)], axis=1)
        new_world2cam = np.matmul(rot_mat, cam_pose.world_to_camera)
        return CameraPose.from_world_to_camera(new_world2cam)
