'''
Inference engine for sparse image pair correspondences
'''

import time
import random

import numpy as np
import torch

from COTR.inference.inference_helper import THRESHOLD_SPARSE, THRESHOLD_AREA, cotr_flow
from COTR.inference.refinement_task import RefinementTask
from COTR.utils import debug_utils, utils
from COTR.cameras.capture import stretch_to_square_np


class SparseEngine():
    def __init__(self, model, batch_size, mode='stretching'):
        assert mode in ['stretching', 'tile']
        self.model = model
        self.batch_size = batch_size
        self.total_tasks = 0
        self.mode = mode

    def form_batch(self, tasks, zoom=None):
        counter = 0
        task_ref = []
        img_batch = []
        query_batch = []
        for t in tasks:
            if t.status == 'unfinished' and t.submitted == False:
                if zoom is not None and t.cur_zoom != zoom:
                    continue
                task_ref.append(t)
                img, query = t.get_task()
                img_batch.append(img)
                query_batch.append(query)
                counter += 1
                if counter >= self.batch_size:
                    break
        if len(task_ref) == 0:
            return [], [], []
        img_batch = torch.stack(img_batch)
        query_batch = torch.stack(query_batch)
        return task_ref, img_batch, query_batch

    def infer_batch(self, img_batch, query_batch):
        self.total_tasks += img_batch.shape[0]
        device = next(self.model.parameters()).device
        img_batch = img_batch.to(device)
        query_batch = query_batch.to(device)
        out = self.model(img_batch, query_batch)['pred_corrs'].clone().detach()
        out = out.cpu().numpy()[:, 0, :]
        if utils.has_nan(out):
            raise ValueError('NaN in prediction')
        return out

    def conclude_tasks(self, tasks, return_idx=False, force=False,
                       offset_x_from=0,
                       offset_y_from=0,
                       offset_x_to=0,
                       offset_y_to=0,
                       img_a_shape=None,
                       img_b_shape=None):
        corrs = []
        idx = []
        for t in tasks:
            if t.status == 'finished':
                out = t.conclude(force)
                if out is not None:
                    corrs.append(np.array(out))
                    idx.append(t.identifier)
        corrs = np.array(corrs)
        idx = np.array(idx)
        if corrs.shape[0] > 0:
            corrs -= np.array([offset_x_from, offset_y_from, offset_x_to, offset_y_to])
            if img_a_shape is not None and img_b_shape is not None:
                border_mask = np.prod(corrs < np.concatenate([img_a_shape[::-1], img_b_shape[::-1]]), axis=1)
                border_mask = (np.prod(corrs > np.array([0, 0, 0, 0]), axis=1) * border_mask).astype(np.bool)
                corrs = corrs[border_mask]
                idx = idx[border_mask]
        if return_idx:
            return corrs, idx
        return corrs

    def num_finished_tasks(self, tasks):
        counter = 0
        for t in tasks:
            if t.status == 'finished':
                counter += 1
        return counter

    def num_good_tasks(self, tasks):
        counter = 0
        for t in tasks:
            if t.result == 'good':
                counter += 1
        return counter

    def gen_tasks(self, img_a, img_b, zoom_ins=[1.0], converge_iters=1, max_corrs=1000, queries_a=None, force=False):
        if self.mode == 'stretching':
            if img_a.shape[0] != img_a.shape[1] or img_b.shape[0] != img_b.shape[1]:
                img_a_shape = img_a.shape
                img_b_shape = img_b.shape
                img_a_sq = stretch_to_square_np(img_a.copy())
                img_b_sq = stretch_to_square_np(img_b.copy())
                corr_a, con_a, resample_a, corr_b, con_b, resample_b = cotr_flow(self.model,
                                                                                 img_a_sq,
                                                                                 img_b_sq
                                                                                 )
                corr_a = utils.float_image_resize(corr_a, img_a_shape[:2])
                con_a = utils.float_image_resize(con_a, img_a_shape[:2])
                resample_a = utils.float_image_resize(resample_a, img_a_shape[:2])
                corr_b = utils.float_image_resize(corr_b, img_b_shape[:2])
                con_b = utils.float_image_resize(con_b, img_b_shape[:2])
                resample_b = utils.float_image_resize(resample_b, img_b_shape[:2])
            else:
                corr_a, con_a, resample_a, corr_b, con_b, resample_b = cotr_flow(self.model,
                                                                                 img_a,
                                                                                 img_b
                                                                                 )
        elif self.mode == 'tile':
            corr_a, con_a, resample_a, corr_b, con_b, resample_b = cotr_flow(self.model,
                                                                             img_a,
                                                                             img_b
                                                                             )
        else:
            raise ValueError(f'unsupported mode: {self.mode}')
        mask_a = con_a < THRESHOLD_SPARSE
        mask_b = con_b < THRESHOLD_SPARSE
        area_a = (con_a < THRESHOLD_AREA).sum() / mask_a.size
        area_b = (con_b < THRESHOLD_AREA).sum() / mask_b.size
        tasks = []

        if queries_a is None:
            index_a = np.where(mask_a)
            index_a = np.array(index_a).T
            index_a = index_a[np.random.choice(len(index_a), min(max_corrs, len(index_a)))]
            index_b = np.where(mask_b)
            index_b = np.array(index_b).T
            index_b = index_b[np.random.choice(len(index_b), min(max_corrs, len(index_b)))]
            for pos in index_a:
                loc_from = pos[::-1]
                loc_to = (corr_a[tuple(np.floor(pos).astype('int'))].copy() * 0.5 + 0.5) * img_b.shape[:2][::-1]
                tasks.append(RefinementTask(img_a, img_b, loc_from, loc_to, area_a, area_b, converge_iters, zoom_ins))
            for pos in index_b:
                '''
                trick: suppose to fix the query point location(loc_from),
                but here it fixes the first guess(loc_to).
                '''
                loc_from = pos[::-1]
                loc_to = (corr_b[tuple(np.floor(pos).astype('int'))].copy() * 0.5 + 0.5) * img_a.shape[:2][::-1]
                tasks.append(RefinementTask(img_a, img_b, loc_to, loc_from, area_a, area_b, converge_iters, zoom_ins))
        else:
            if force:
                for i, loc_from in enumerate(queries_a):
                    pos = loc_from[::-1]
                    pos = np.array([np.clip(pos[0], 0, corr_a.shape[0] - 1), np.clip(pos[1], 0, corr_a.shape[1] - 1)], dtype=np.int)
                    loc_to = (corr_a[tuple(pos)].copy() * 0.5 + 0.5) * img_b.shape[:2][::-1]
                    tasks.append(RefinementTask(img_a, img_b, loc_from, loc_to, area_a, area_b, converge_iters, zoom_ins, identifier=i))
            else:
                for i, loc_from in enumerate(queries_a):
                    pos = loc_from[::-1]
                    if (pos > np.array(img_a.shape[:2]) - 1).any() or (pos < 0).any():
                        continue
                    if mask_a[tuple(np.floor(pos).astype('int'))]:
                        loc_to = (corr_a[tuple(np.floor(pos).astype('int'))].copy() * 0.5 + 0.5) * img_b.shape[:2][::-1]
                        tasks.append(RefinementTask(img_a, img_b, loc_from, loc_to, area_a, area_b, converge_iters, zoom_ins, identifier=i))
                if len(tasks) < max_corrs:
                    extra = max_corrs - len(tasks)
                    counter = 0
                    for i, loc_from in enumerate(queries_a):
                        if counter >= extra:
                            break
                        pos = loc_from[::-1]
                        if (pos > np.array(img_a.shape[:2]) - 1).any() or (pos < 0).any():
                            continue
                        if mask_a[tuple(np.floor(pos).astype('int'))] == False:
                            loc_to = (corr_a[tuple(np.floor(pos).astype('int'))].copy() * 0.5 + 0.5) * img_b.shape[:2][::-1]
                            tasks.append(RefinementTask(img_a, img_b, loc_from, loc_to, area_a, area_b, converge_iters, zoom_ins, identifier=i))
                            counter += 1
        return tasks

    def cotr_corr_multiscale(self, img_a, img_b, zoom_ins=[1.0], converge_iters=1, max_corrs=1000, queries_a=None, return_idx=False, force=False, return_tasks_only=False):
        '''
        currently only support fixed queries_a
        '''
        img_a = img_a.copy()
        img_b = img_b.copy()
        img_a_shape = img_a.shape[:2]
        img_b_shape = img_b.shape[:2]
        if queries_a is not None:
            queries_a = queries_a.copy()
        tasks = self.gen_tasks(img_a, img_b, zoom_ins, converge_iters, max_corrs, queries_a, force)
        while True:
            num_g = self.num_good_tasks(tasks)
            print(f'{num_g} / {max_corrs} | {self.num_finished_tasks(tasks)} / {len(tasks)}')
            task_ref, img_batch, query_batch = self.form_batch(tasks)
            if len(task_ref) == 0:
                break
            if num_g >= max_corrs:
                break
            out = self.infer_batch(img_batch, query_batch)
            for t, o in zip(task_ref, out):
                t.step(o)
        if return_tasks_only:
            return tasks
        if return_idx:
            corrs, idx = self.conclude_tasks(tasks, return_idx=True, force=force,
                                             img_a_shape=img_a_shape,
                                             img_b_shape=img_b_shape,)
            corrs = corrs[:max_corrs]
            idx = idx[:max_corrs]
            return corrs, idx
        else:
            corrs = self.conclude_tasks(tasks, force=force,
                                        img_a_shape=img_a_shape,
                                        img_b_shape=img_b_shape,)
            corrs = corrs[:max_corrs]
            return corrs

    def cotr_corr_multiscale_with_cycle_consistency(self, img_a, img_b, zoom_ins=[1.0], converge_iters=1, max_corrs=1000, queries_a=None, return_idx=False, return_cycle_error=False):
        EXTRACTION_RATE = 0.3
        temp_max_corrs = int(max_corrs / EXTRACTION_RATE)
        if queries_a is not None:
            temp_max_corrs = min(temp_max_corrs, queries_a.shape[0])
            queries_a = queries_a.copy()
        corr_f, idx_f = self.cotr_corr_multiscale(img_a.copy(), img_b.copy(),
                                                  zoom_ins=zoom_ins,
                                                  converge_iters=converge_iters,
                                                  max_corrs=temp_max_corrs,
                                                  queries_a=queries_a,
                                                  return_idx=True)
        assert corr_f.shape[0] > 0
        corr_b, idx_b = self.cotr_corr_multiscale(img_b.copy(), img_a.copy(),
                                                  zoom_ins=zoom_ins,
                                                  converge_iters=converge_iters,
                                                  max_corrs=corr_f.shape[0],
                                                  queries_a=corr_f[:, 2:].copy(),
                                                  return_idx=True)
        assert corr_b.shape[0] > 0
        cycle_errors = np.linalg.norm(corr_f[idx_b][:, :2] - corr_b[:, 2:], axis=1)
        order = np.argsort(cycle_errors)
        out = [corr_f[idx_b][order][:max_corrs]]
        if return_idx:
            out.append(idx_f[idx_b][order][:max_corrs])
        if return_cycle_error:
            out.append(cycle_errors[order][:max_corrs])
        if len(out) == 1:
            out = out[0]
        return out
