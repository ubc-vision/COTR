import time

import numpy as np
import torch
from torchvision.transforms import functional as tvtf
import imageio
import PIL

from COTR.inference.inference_helper import BASE_ZOOM, THRESHOLD_PIXELS_RELATIVE, get_patch_centered_at, two_images_side_by_side, find_prediction_loop
from COTR.utils import debug_utils, utils
from COTR.utils.constants import MAX_SIZE
from COTR.utils.utils import ImagePatch


class RefinementTask():
    def __init__(self, image_from, image_to, loc_from, loc_to, area_from, area_to, converge_iters, zoom_ins, identifier=None):
        self.identifier = identifier
        self.image_from = image_from
        self.image_to = image_to
        self.loc_from = loc_from
        self.best_loc_to = loc_to
        self.cur_loc_to = loc_to
        self.area_from = area_from
        self.area_to = area_to
        if self.area_from < self.area_to:
            self.s_from = BASE_ZOOM
            self.s_to = BASE_ZOOM * np.sqrt(self.area_to / self.area_from)
        else:
            self.s_to = BASE_ZOOM
            self.s_from = BASE_ZOOM * np.sqrt(self.area_from / self.area_to)

        self.cur_job = {}
        self.status = 'unfinished'
        self.result = 'unknown'

        self.converge_iters = converge_iters
        self.zoom_ins = zoom_ins
        self.cur_zoom_idx = 0
        self.cur_iter = 0
        self.total_iter = 0

        self.loc_to_at_zoom = []
        self.loc_history = [loc_to]
        self.all_loc_to_dict = {}
        self.job_history = []
        self.submitted = False

    @property
    def cur_zoom(self):
        return self.zoom_ins[self.cur_zoom_idx]

    @property
    def confidence_scaling_factor(self):
        if self.cur_zoom_idx > 0:
            conf_scaling = float(self.cur_zoom) / float(self.zoom_ins[0])
        else:
            conf_scaling = 1.0
        return conf_scaling

    def peek(self):
        assert self.status == 'unfinished'
        patch_from = get_patch_centered_at(None, self.loc_from, scale=self.s_from * self.cur_zoom, return_content=False, img_shape=self.image_from.shape)
        patch_to = get_patch_centered_at(None, self.cur_loc_to, scale=self.s_to * self.cur_zoom, return_content=False, img_shape=self.image_to.shape)
        top_job = {'patch_from': patch_from,
                    'patch_to': patch_to,
                    'loc_from': self.loc_from,
                    'loc_to': self.cur_loc_to,
                    }
        return top_job

    def get_task_pilot(self, pilot):
        assert self.status == 'unfinished'
        patch_from = ImagePatch(None, pilot.cur_job['patch_from'].x, pilot.cur_job['patch_from'].y, pilot.cur_job['patch_from'].w, pilot.cur_job['patch_from'].h, pilot.cur_job['patch_from'].ow, pilot.cur_job['patch_from'].oh)
        patch_to   = ImagePatch(None, pilot.cur_job['patch_to'].x, pilot.cur_job['patch_to'].y, pilot.cur_job['patch_to'].w, pilot.cur_job['patch_to'].h, pilot.cur_job['patch_to'].ow, pilot.cur_job['patch_to'].oh)
        query = torch.from_numpy((np.array(self.loc_from) - np.array([patch_from.x, patch_from.y])) / np.array([patch_from.w * 2, patch_from.h]))[None].float()
        self.cur_job = {'patch_from': patch_from,
                        'patch_to': patch_to,
                        'loc_from': self.loc_from,
                        'loc_to': self.cur_loc_to,
                        'img': None,
                        }
        self.job_history.append((patch_from.h, patch_from.w, patch_to.h, patch_to.w))
        assert self.submitted == False
        self.submitted = True
        return None, query

    def get_task_fast(self):
        assert self.status == 'unfinished'
        patch_from = get_patch_centered_at(self.image_from, self.loc_from, scale=self.s_from * self.cur_zoom, return_content=False)
        patch_to = get_patch_centered_at(self.image_to, self.cur_loc_to, scale=self.s_to * self.cur_zoom, return_content=False)
        query = torch.from_numpy((np.array(self.loc_from) - np.array([patch_from.x, patch_from.y])) / np.array([patch_from.w * 2, patch_from.h]))[None].float()
        self.cur_job = {'patch_from': patch_from,
                        'patch_to': patch_to,
                        'loc_from': self.loc_from,
                        'loc_to': self.cur_loc_to,
                        'img': None,
                        }

        self.job_history.append((patch_from.h, patch_from.w, patch_to.h, patch_to.w))
        assert self.submitted == False
        self.submitted = True

        return None, query

    def get_task(self):
        assert self.status == 'unfinished'
        patch_from = get_patch_centered_at(self.image_from, self.loc_from, scale=self.s_from * self.cur_zoom)
        patch_to = get_patch_centered_at(self.image_to, self.cur_loc_to, scale=self.s_to * self.cur_zoom)

        query = torch.from_numpy((np.array(self.loc_from) - np.array([patch_from.x, patch_from.y])) / np.array([patch_from.w * 2, patch_from.h]))[None].float()

        img_from = patch_from.patch
        img_to = patch_to.patch
        assert img_from.shape[0] == img_from.shape[1]
        assert img_to.shape[0] == img_to.shape[1]

        img_from = np.array(PIL.Image.fromarray(img_from).resize((MAX_SIZE, MAX_SIZE), resample=PIL.Image.BILINEAR))
        img_to = np.array(PIL.Image.fromarray(img_to).resize((MAX_SIZE, MAX_SIZE), resample=PIL.Image.BILINEAR))
        img = two_images_side_by_side(img_from, img_to)
        img = tvtf.normalize(tvtf.to_tensor(img), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).float()

        self.cur_job = {'patch_from': ImagePatch(None, patch_from.x, patch_from.y, patch_from.w, patch_from.h, patch_from.ow, patch_from.oh),
                        'patch_to': ImagePatch(None, patch_to.x, patch_to.y, patch_to.w, patch_to.h, patch_to.ow, patch_to.oh),
                        'loc_from': self.loc_from,
                        'loc_to': self.cur_loc_to,
                        }

        self.job_history.append((patch_from.h, patch_from.w, patch_to.h, patch_to.w))
        assert self.submitted == False
        self.submitted = True

        return img, query

    def next_zoom(self):
        if self.cur_zoom_idx >= len(self.zoom_ins) - 1:
            self.status = 'finished'
            if self.conclude() is None:
                self.result = 'bad'
            else:
                self.result = 'good'
        self.cur_zoom_idx += 1
        self.cur_iter = 0
        self.loc_to_at_zoom = []

    def scale_to_loc(self, raw_to_loc):
        raw_to_loc = raw_to_loc.copy()
        patch_b = self.cur_job['patch_to']
        raw_to_loc[0] = (raw_to_loc[0] - 0.5) * 2
        loc_to = raw_to_loc * np.array([patch_b.w, patch_b.h])
        loc_to = loc_to + np.array([patch_b.x, patch_b.y])
        return loc_to

    def step(self, raw_to_loc):
        assert self.submitted == True
        self.submitted = False
        loc_to = self.scale_to_loc(raw_to_loc)
        self.total_iter += 1
        self.loc_to_at_zoom.append(loc_to)
        self.cur_loc_to = loc_to
        zoom_finished = False
        if self.cur_zoom_idx == len(self.zoom_ins) - 1:
            # converge at the last level
            if len(self.loc_to_at_zoom) >= 2:
                zoom_finished = np.prod(self.loc_to_at_zoom[:-1] == loc_to, axis=1, keepdims=True).any()
            if self.cur_iter >= self.converge_iters - 1:
                zoom_finished = True
            self.cur_iter += 1
        else:
            # finish immediately for other levels
            zoom_finished = True
        if zoom_finished:
            self.all_loc_to_dict[self.cur_zoom] = np.array(self.loc_to_at_zoom).copy()
            last_level_loc_to = self.all_loc_to_dict[self.cur_zoom]
            if len(last_level_loc_to) >= 2:
                has_loop = np.prod(last_level_loc_to[:-1] == last_level_loc_to[-1], axis=1, keepdims=True).any()
                if has_loop:
                    loop = find_prediction_loop(last_level_loc_to)
                    loc_to = loop.mean(axis=0)
            self.loc_history.append(loc_to)
            self.best_loc_to = loc_to
            self.cur_loc_to = self.best_loc_to
            self.next_zoom()

    def conclude(self, force=False):
        loc_history = np.array(self.loc_history)
        if (force == False) and (max(loc_history.std(axis=0)) >= THRESHOLD_PIXELS_RELATIVE * max(*self.image_to.shape)):
            return None
        return np.concatenate([self.loc_from, self.best_loc_to])

    def conclude_intermedia(self):
        return np.concatenate([np.array(self.loc_history), np.array(self.job_history)], axis=1)
