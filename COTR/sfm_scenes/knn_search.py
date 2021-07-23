'''
Given one capture in a scene, search for its KNN captures
'''

import os

import numpy as np

from COTR.utils import debug_utils
from COTR.utils.constants import VALID_NN_OVERLAPPING_THRESH


class ReprojRatioKnnSearch():
    def __init__(self, scene):
        self.scene = scene
        self.distance_mat = None
        self.nn_index = None
        self._read_dist_mat()
        self._build_nn_index()

    def _read_dist_mat(self):
        dist_mat_path = os.path.join(os.path.dirname(os.path.dirname(self.scene.captures[0].depth_path)), 'dist_mat/dist_mat.npy')
        self.distance_mat = np.load(dist_mat_path)

    def _build_nn_index(self):
        # argsort is in ascending order, so we take negative
        self.nn_index = (-1 * self.distance_mat).argsort(axis=1)

    def get_knn(self, query, k, db_mask=None):
        query_index = self.scene.img_path_to_index_dict[query.img_path]
        if db_mask is not None:
            query_mask = np.setdiff1d(np.arange(self.distance_mat[query_index].shape[0]), db_mask)
        num_pos = (self.distance_mat[query_index] > VALID_NN_OVERLAPPING_THRESH).sum() if db_mask is None else (self.distance_mat[query_index][db_mask] > VALID_NN_OVERLAPPING_THRESH).sum()
        # we have enough valid NN or not
        if num_pos > k:
            if db_mask is None:
                ind = self.nn_index[query_index][:k + 1]
            else:
                temp_dist = self.distance_mat[query_index].copy()
                temp_dist[query_mask] = -1
                ind = (-1 * temp_dist).argsort(axis=0)[:k + 1]
            # remove self
            if query_index in ind:
                ind = np.delete(ind, np.argwhere(ind == query_index))
            else:
                ind = ind[:k]
            assert ind.shape[0] <= k, ind.shape[0] > 0
        else:
            k = num_pos
            if db_mask is None:
                ind = self.nn_index[query_index][:max(k, 1)]
            else:
                temp_dist = self.distance_mat[query_index].copy()
                temp_dist[query_mask] = -1
                ind = (-1 * temp_dist).argsort(axis=0)[:max(k, 1)]
        return self.scene.get_captures_given_index_list(ind)
