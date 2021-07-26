import warnings

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as tvtf
from tqdm import tqdm
import PIL

from COTR.utils import utils, debug_utils
from COTR.utils.constants import MAX_SIZE
from COTR.cameras.capture import crop_center_max_np, pad_to_square_np
from COTR.utils.utils import ImagePatch

THRESHOLD_SPARSE = 0.02
THRESHOLD_PIXELS_RELATIVE = 0.02
BASE_ZOOM = 1.0
THRESHOLD_AREA = 0.02
LARGE_GPU = True


def find_prediction_loop(arr):
    '''
    loop ends at last element
    '''
    assert arr.shape[1] == 2, 'requires shape (N, 2)'
    start_index = np.where(np.prod(arr[:-1] == arr[-1], axis=1))[0][0]
    return arr[start_index:-1]


def two_images_side_by_side(img_a, img_b):
    assert img_a.shape == img_b.shape, f'{img_a.shape} vs {img_b.shape}'
    assert img_a.dtype == img_b.dtype
    h, w, c = img_a.shape
    canvas = np.zeros((h, 2 * w, c), dtype=img_a.dtype)
    canvas[:, 0 * w:1 * w, :] = img_a
    canvas[:, 1 * w:2 * w, :] = img_b
    return canvas


def to_square_patches(img):
    patches = []
    h, w, _ = img.shape
    short = size = min(h, w)
    long = max(h, w)
    if long == short:
        patch_0 = ImagePatch(img[:size, :size], 0, 0, size, size, w, h)
        patches = [patch_0]
    elif long <= size * 2:
        warnings.warn('Spatial smoothness in dense optical flow is lost, but sparse matching and triangulation should be fine')
        patch_0 = ImagePatch(img[:size, :size], 0, 0, size, size, w, h)
        patch_1 = ImagePatch(img[-size:, -size:], w - size, h - size, size, size, w, h)
        patches = [patch_0, patch_1]
        # patches += subdivide_patch(patch_0)
        # patches += subdivide_patch(patch_1)
    else:
        raise NotImplementedError
    return patches


def merge_flow_patches(corrs):
    confidence = np.ones([corrs[0].oh, corrs[0].ow]) * 100
    flow = np.zeros([corrs[0].oh, corrs[0].ow, 2])
    cmap = np.ones([corrs[0].oh, corrs[0].ow]) * -1
    for i, c in enumerate(corrs):
        temp = np.ones([c.oh, c.ow]) * 100
        temp[c.y:c.y + c.h, c.x:c.x + c.w] = c.patch[..., 2]
        tempf = np.zeros([c.oh, c.ow, 2])
        tempf[c.y:c.y + c.h, c.x:c.x + c.w] = c.patch[..., :2]
        min_ind = np.stack([temp, confidence], axis=-1).argmin(axis=-1)
        min_ind = min_ind == 0
        confidence[min_ind] = temp[min_ind]
        flow[min_ind] = tempf[min_ind]
        cmap[min_ind] = i
    return flow, confidence, cmap


def get_patch_centered_at(img, pos, scale=1.0, return_content=True, img_shape=None):
    '''
    pos - [x, y]
    '''
    if img_shape is None:
        img_shape = img.shape
    h, w, _ = img_shape
    short = min(h, w)
    scale = np.clip(scale, 0.0, 1.0)
    size = short * scale
    size = int((size // 2) * 2)
    lu_y = int(pos[1] - size // 2)
    lu_x = int(pos[0] - size // 2)
    if lu_y < 0:
        lu_y -= lu_y
    if lu_x < 0:
        lu_x -= lu_x
    if lu_y + size > h:
        lu_y -= (lu_y + size) - (h)
    if lu_x + size > w:
        lu_x -= (lu_x + size) - (w)
    if return_content:
        return ImagePatch(img[lu_y:lu_y + size, lu_x:lu_x + size], lu_x, lu_y, size, size, w, h)
    else:
        return ImagePatch(None, lu_x, lu_y, size, size, w, h)


def cotr_patch_flow_exhaustive(model, patches_a, patches_b):
    def one_pass(model, img_a, img_b):
        device = next(model.parameters()).device
        assert img_a.shape[0] == img_a.shape[1]
        assert img_b.shape[0] == img_b.shape[1]
        img_a = np.array(PIL.Image.fromarray(img_a).resize((MAX_SIZE, MAX_SIZE), resample=PIL.Image.BILINEAR))
        img_b = np.array(PIL.Image.fromarray(img_b).resize((MAX_SIZE, MAX_SIZE), resample=PIL.Image.BILINEAR))
        img = two_images_side_by_side(img_a, img_b)
        img = tvtf.normalize(tvtf.to_tensor(img), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).float()[None]
        img = img.to(device)

        q_list = []
        for i in range(MAX_SIZE):
            queries = []
            for j in range(MAX_SIZE * 2):
                queries.append([(j) / (MAX_SIZE * 2), i / MAX_SIZE])
            queries = np.array(queries)
            q_list.append(queries)
        if LARGE_GPU:
            try:
                queries = torch.from_numpy(np.concatenate(q_list))[None].float().to(device)
                out = model.forward(img, queries)['pred_corrs'].detach().cpu().numpy()[0]
                out_list = out.reshape(MAX_SIZE, MAX_SIZE * 2, -1)
            except:
                assert 0, 'set LARGE_GPU to False'
        else:
            out_list = []
            for q in q_list:
                queries = torch.from_numpy(q)[None].float().to(device)
                out = model.forward(img, queries)['pred_corrs'].detach().cpu().numpy()[0]
                out_list.append(out)
            out_list = np.array(out_list)
        in_grid = torch.from_numpy(np.array(q_list)).float()[None] * 2 - 1
        out_grid = torch.from_numpy(out_list).float()[None] * 2 - 1
        cycle_grid = torch.nn.functional.grid_sample(out_grid.permute(0, 3, 1, 2), out_grid).permute(0, 2, 3, 1)
        confidence = torch.norm(cycle_grid[0, ...] - in_grid[0, ...], dim=-1)
        corr = out_grid[0].clone()
        corr[:, :MAX_SIZE, 0] = corr[:, :MAX_SIZE, 0] * 2 - 1
        corr[:, MAX_SIZE:, 0] = corr[:, MAX_SIZE:, 0] * 2 + 1
        corr = torch.cat([corr, confidence[..., None]], dim=-1).numpy()
        return corr[:, :MAX_SIZE, :], corr[:, MAX_SIZE:, :]
    corrs_a = []
    corrs_b = []

    for p_i in patches_a:
        for p_j in patches_b:
            c_i, c_j = one_pass(model, p_i.patch, p_j.patch)
            base_corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
            real_corners_j = (np.array([[p_j.x, p_j.y], [p_j.x + p_j.w, p_j.y], [p_j.x + p_j.w, p_j.y + p_j.h], [p_j.x, p_j.y + p_j.h]]) / np.array([p_j.ow, p_j.oh])) * 2 + np.array([-1, -1])
            real_corners_i = (np.array([[p_i.x, p_i.y], [p_i.x + p_i.w, p_i.y], [p_i.x + p_i.w, p_i.y + p_i.h], [p_i.x, p_i.y + p_i.h]]) / np.array([p_i.ow, p_i.oh])) * 2 + np.array([-1, -1])
            T_i = cv2.getAffineTransform(base_corners[:3].astype(np.float32), real_corners_j[:3].astype(np.float32))
            T_j = cv2.getAffineTransform(base_corners[:3].astype(np.float32), real_corners_i[:3].astype(np.float32))
            c_i[..., :2] = c_i[..., :2] @ T_i[:2, :2] + T_i[:, 2]
            c_j[..., :2] = c_j[..., :2] @ T_j[:2, :2] + T_j[:, 2]
            c_i = utils.float_image_resize(c_i, (p_i.h, p_i.w))
            c_j = utils.float_image_resize(c_j, (p_j.h, p_j.w))
            c_i = ImagePatch(c_i, p_i.x, p_i.y, p_i.w, p_i.h, p_i.ow, p_i.oh)
            c_j = ImagePatch(c_j, p_j.x, p_j.y, p_j.w, p_j.h, p_j.ow, p_j.oh)
            corrs_a.append(c_i)
            corrs_b.append(c_j)
    return corrs_a, corrs_b


def cotr_flow(model, img_a, img_b):
    # assert img_a.shape[0] == img_a.shape[1]
    # assert img_b.shape[0] == img_b.shape[1]
    patches_a = to_square_patches(img_a)
    patches_b = to_square_patches(img_b)

    corrs_a, corrs_b = cotr_patch_flow_exhaustive(model, patches_a, patches_b)
    corr_a, con_a, cmap_a = merge_flow_patches(corrs_a)
    corr_b, con_b, cmap_b = merge_flow_patches(corrs_b)

    resample_a = utils.torch_img_to_np_img(torch.nn.functional.grid_sample(utils.np_img_to_torch_img(img_b)[None].float(),
                                                                           torch.from_numpy(corr_a)[None].float())[0])
    resample_b = utils.torch_img_to_np_img(torch.nn.functional.grid_sample(utils.np_img_to_torch_img(img_a)[None].float(),
                                                                           torch.from_numpy(corr_b)[None].float())[0])
    return corr_a, con_a, resample_a, corr_b, con_b, resample_b


def cotr_corr_base(model, img_a, img_b, queries_a):
    def one_pass(model, img_a, img_b, queries):
        device = next(model.parameters()).device
        assert img_a.shape[0] == img_a.shape[1]
        assert img_b.shape[0] == img_b.shape[1]
        img_a = np.array(PIL.Image.fromarray(img_a).resize((MAX_SIZE, MAX_SIZE), resample=PIL.Image.BILINEAR))
        img_b = np.array(PIL.Image.fromarray(img_b).resize((MAX_SIZE, MAX_SIZE), resample=PIL.Image.BILINEAR))
        img = two_images_side_by_side(img_a, img_b)
        img = tvtf.normalize(tvtf.to_tensor(img), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).float()[None]
        img = img.to(device)

        queries = torch.from_numpy(queries)[None].float().to(device)
        out = model.forward(img, queries)['pred_corrs'].clone().detach()
        cycle = model.forward(img, out)['pred_corrs'].clone().detach()

        queries = queries.cpu().numpy()[0]
        out = out.cpu().numpy()[0]
        cycle = cycle.cpu().numpy()[0]
        conf = np.linalg.norm(queries - cycle, axis=1, keepdims=True)
        return np.concatenate([out, conf], axis=1)

    patches_a = to_square_patches(img_a)
    patches_b = to_square_patches(img_b)
    pred_list = []

    for p_i in patches_a:
        for p_j in patches_b:
            normalized_queries_a = queries_a.copy()
            mask = (normalized_queries_a[:, 0] >= p_i.x) & (normalized_queries_a[:, 1] >= p_i.y) & (normalized_queries_a[:, 0] <= p_i.x + p_i.w) & (normalized_queries_a[:, 1] <= p_i.y + p_i.h)
            normalized_queries_a[:, 0] -= p_i.x
            normalized_queries_a[:, 1] -= p_i.y
            normalized_queries_a[:, 0] /= 2 * p_i.w
            normalized_queries_a[:, 1] /= p_i.h
            pred = one_pass(model, p_i.patch, p_j.patch, normalized_queries_a)
            pred[~mask, 2] = np.inf
            pred[:, 0] -= 0.5
            pred[:, 0] *= 2 * p_j.w
            pred[:, 0] += p_j.x
            pred[:, 1] *= p_j.h
            pred[:, 1] += p_j.y
            pred_list.append(pred)

    pred_list = np.stack(pred_list).transpose(1, 0, 2)
    out = []
    for item in pred_list:
        out.append(item[np.argmin(item[..., 2], axis=0)])
    out = np.array(out)[..., :2]
    return np.concatenate([queries_a, out], axis=1)


try:
    from vispy import gloo
    from vispy import app
    from vispy.util.ptime import time
    from scipy.spatial import Delaunay
    from vispy.gloo.wrappers import read_pixels

    app.use_app('glfw')


    vertex_shader = """
        attribute vec4 color;
        attribute vec2 position;
        varying vec4 v_color;
        void main()
        {
            gl_Position = vec4(position, 0.0, 1.0);
            v_color = color;
        } """

    fragment_shader = """
        varying vec4 v_color;
        void main()
        {
            gl_FragColor = v_color;
        } """


    class Canvas(app.Canvas):
        def __init__(self, mesh, color, size):
            # We hide the canvas upon creation.
            app.Canvas.__init__(self, show=False, size=size)
            self._t0 = time()
            # Texture where we render the scene.
            self._rendertex = gloo.Texture2D(shape=self.size[::-1] + (4,), internalformat='rgba32f')
            # FBO.
            self._fbo = gloo.FrameBuffer(self._rendertex,
                                        gloo.RenderBuffer(self.size[::-1]))
            # Regular program that will be rendered to the FBO.
            self.program = gloo.Program(vertex_shader, fragment_shader)
            self.program["position"] = mesh
            self.program['color'] = color
            # We manually draw the hidden canvas.
            self.update()

        def on_draw(self, event):
            # Render in the FBO.
            with self._fbo:
                gloo.clear('black')
                gloo.set_viewport(0, 0, *self.size)
                self.program.draw()
                # Retrieve the contents of the FBO texture.
                self.im = read_pixels((0, 0, self.size[0], self.size[1]), True, out_type='float')
            self._time = time() - self._t0
            # Immediately exit the application.
            app.quit()


    def triangulate_corr(corr, from_shape, to_shape):
        corr = corr.copy()
        to_shape = to_shape[:2]
        from_shape = from_shape[:2]
        corr = corr / np.concatenate([from_shape[::-1], to_shape[::-1]])
        tri = Delaunay(corr[:, :2])
        mesh = corr[:, :2][tri.simplices].astype(np.float32) * 2 - 1
        mesh[..., 1] *= -1
        color = corr[:, 2:][tri.simplices].astype(np.float32)
        color = np.concatenate([color, np.ones_like(color[..., 0:2])], axis=-1)
        c = Canvas(mesh.reshape(-1, 2), color.reshape(-1, 4), size=(from_shape[::-1]))
        app.run()
        render = c.im.copy()
        render = render[..., :2]
        render *= np.array(to_shape[::-1])
        return render
except:
    print('cannot use vispy, setting triangulate_corr as None')
    triangulate_corr = None
