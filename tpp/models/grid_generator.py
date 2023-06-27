import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import Registry
from ..utils.vis import vis_batched_imgs


from mmcv.cnn import MODELS as MMCV_MODELS
GRID_GENERATORS = Registry('models', parent=MMCV_MODELS)

import os, json
from .homography_layers import HomographyLayer, HomographySaliencyParamsNet, HomographyLayerGlobal, CuboidLayerGlobal

def build_grid_generator(cfg):
    """Build grid generator."""
    return GRID_GENERATORS.build(cfg)

def make1DGaussian(size, fwhm=3, center=None):
    """ Make a 1D gaussian kernel.

    size is the length of the kernel,
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, size, 1, dtype=float)

    if center is None:
        center = size // 2

    return np.exp(-4*np.log(2) * (x-center)**2 / fwhm**2)


def make2DGaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def unwarp_bboxes(bboxes, grid, output_shape):
    """Unwarps a tensor of bboxes of shape (n, 4) or (n, 5) according to the grid \
    of shape (h, w, 2) used to warp the corresponding image and the \
    output_shape (H, W, ...)."""
    bboxes = bboxes.clone()
    # image map of unwarped (x,y) coordinates
    img = grid.permute(2, 0, 1).unsqueeze(0)

    warped_height, warped_width = grid.shape[0:2]
    xgrid = 2 * (bboxes[:, 0:4:2] / warped_width) - 1
    ygrid = 2 * (bboxes[:, 1:4:2] / warped_height) - 1
    grid = torch.stack((xgrid, ygrid), dim=2).unsqueeze(0)

    # warped_bboxes has shape (2, num_bboxes, 2)
    warped_bboxes = F.grid_sample(
        img, grid, align_corners=True, padding_mode="border").squeeze(0)
    bboxes[:, 0:4:2] = (warped_bboxes[0] + 1) / 2 * output_shape[1]
    bboxes[:, 1:4:2] = (warped_bboxes[1] + 1) / 2 * output_shape[0]

    return bboxes


class RecasensSaliencyToGridMixin(object):
    """Grid generator based on 'Learning to Zoom: a Saliency-Based Sampling \
    Layer for Neural Networks' [https://arxiv.org/pdf/1809.03355.pdf]."""

    def __init__(self, output_shape, grid_shape=(31, 51), separable=True,
                 attraction_fwhm=13, anti_crop=True, **kwargs):
        super(RecasensSaliencyToGridMixin, self).__init__()
        self.output_shape = output_shape
        self.output_height, self.output_width = output_shape
        self.grid_shape = grid_shape
        self.padding_size = min(self.grid_shape)-1
        self.total_shape = tuple(
            dim+2*self.padding_size
            for dim in self.grid_shape
        )
        self.padding_mode = 'reflect' if anti_crop else 'replicate'
        self.separable = separable

        if self.separable:
            self.filter = make1DGaussian(
                2*self.padding_size+1, fwhm=attraction_fwhm)
            self.filter = torch.FloatTensor(self.filter).unsqueeze(0) \
                                                        .unsqueeze(0).cuda()

            self.P_basis_x = torch.zeros(self.total_shape[1])
            for i in range(self.total_shape[1]):
                self.P_basis_x[i] = \
                    (i-self.padding_size)/(self.grid_shape[1]-1.0)
            self.P_basis_y = torch.zeros(self.total_shape[0])
            for i in range(self.total_shape[0]):
                self.P_basis_y[i] = \
                    (i-self.padding_size)/(self.grid_shape[0]-1.0)
        else:
            self.filter = make2DGaussian(
                2*self.padding_size+1, fwhm=attraction_fwhm)
            self.filter = torch.FloatTensor(self.filter) \
                               .unsqueeze(0).unsqueeze(0).cuda()

            self.P_basis = torch.zeros(2, *self.total_shape)
            for k in range(2):
                for i in range(self.total_shape[0]):
                    for j in range(self.total_shape[1]):
                        self.P_basis[k, i, j] = k*(i-self.padding_size)/(self.grid_shape[0]-1.0)+(1.0-k)*(j-self.padding_size)/(self.grid_shape[1]-1.0)  # noqa: E501

    def separable_saliency_to_grid(self, imgs, img_metas, x_saliency,
                                   y_saliency, device):
        assert self.separable
        x_saliency = F.pad(x_saliency, (self.padding_size, self.padding_size),
                           mode=self.padding_mode)
        y_saliency = F.pad(y_saliency, (self.padding_size, self.padding_size),
                           mode=self.padding_mode)

        N = imgs.shape[0]
        P_x = torch.zeros(1, 1, self.total_shape[1], device=device)
        P_x[0, 0, :] = self.P_basis_x
        P_x = P_x.expand(N, 1, self.total_shape[1])
        P_y = torch.zeros(1, 1, self.total_shape[0], device=device)
        P_y[0, 0, :] = self.P_basis_y
        P_y = P_y.expand(N, 1, self.total_shape[0])

        weights = F.conv1d(x_saliency, self.filter)
        weighted_offsets = torch.mul(P_x, x_saliency)
        weighted_offsets = F.conv1d(weighted_offsets, self.filter)
        xgrid = weighted_offsets/weights
        xgrid = torch.clamp(xgrid*2-1, min=-1, max=1)
        xgrid = xgrid.view(-1, 1, 1, self.grid_shape[1])
        xgrid = xgrid.expand(-1, 1, *self.grid_shape)

        weights = F.conv1d(y_saliency, self.filter)
        weighted_offsets = F.conv1d(torch.mul(P_y, y_saliency), self.filter)
        ygrid = weighted_offsets/weights
        ygrid = torch.clamp(ygrid*2-1, min=-1, max=1)
        ygrid = ygrid.view(-1, 1, self.grid_shape[0], 1)
        ygrid = ygrid.expand(-1, 1, *self.grid_shape)

        grid = torch.cat((xgrid, ygrid), 1)
        grid = F.interpolate(grid, size=self.output_shape, mode='bilinear',
                             align_corners=True)
        return grid.permute(0, 2, 3, 1)

    def nonseparable_saliency_to_grid(self, imgs, img_metas, saliency, device):
        assert not self.separable
        p = self.padding_size
        saliency = F.pad(saliency, (p, p, p, p), mode=self.padding_mode)

        N = imgs.shape[0]
        P = torch.zeros(1, 2, *self.total_shape, device=device)
        P[0, :, :, :] = self.P_basis
        P = P.expand(N, 2, *self.total_shape)

        saliency_cat = torch.cat((saliency, saliency), 1)
        weights = F.conv2d(saliency, self.filter)
        weighted_offsets = torch.mul(P, saliency_cat) \
                                .view(-1, 1, *self.total_shape)
        weighted_offsets = F.conv2d(weighted_offsets, self.filter) \
                            .view(-1, 2, *self.grid_shape)

        weighted_offsets_x = weighted_offsets[:, 0, :, :] \
            .contiguous().view(-1, 1, *self.grid_shape)
        xgrid = weighted_offsets_x/weights
        xgrid = torch.clamp(xgrid*2-1, min=-1, max=1)
        xgrid = xgrid.view(-1, 1, *self.grid_shape)

        weighted_offsets_y = weighted_offsets[:, 1, :, :] \
            .contiguous().view(-1, 1, *self.grid_shape)
        ygrid = weighted_offsets_y/weights
        ygrid = torch.clamp(ygrid*2-1, min=-1, max=1)
        ygrid = ygrid.view(-1, 1, *self.grid_shape)

        grid = torch.cat((xgrid, ygrid), 1)
        grid = F.interpolate(grid, size=self.output_shape, mode='bilinear',
                             align_corners=True)
        return grid.permute(0, 2, 3, 1)

@GRID_GENERATORS.register_module()
class FixedKDEGrid(nn.Module, RecasensSaliencyToGridMixin):
    """ Grid generator that uses a fixed saliency map -- KDE SD.
        
        If the vanishing point is fixed, this class can be instead 
        used to load saliency during inference.
    """

    def __init__(self, saliency_file, **kwargs):
        super(FixedKDEGrid, self).__init__()
        RecasensSaliencyToGridMixin.__init__(self, **kwargs)
        self.saliency = pickle.load(open(saliency_file, 'rb'))

    def forward(self, imgs, img_metas, **kwargs):
        vis_options = kwargs.get('vis_options', {})
        device = imgs.device
        self.saliency = self.saliency.to(device)

        if 'saliency' in vis_options:
            h, w, _ = img_metas[0]['pad_shape']
            show_saliency = F.interpolate(self.saliency, size=(h, w),
                                          mode='bilinear', align_corners=True)
            show_saliency = 255*(show_saliency/show_saliency.max())
            show_saliency = show_saliency.expand(
                show_saliency.size(0), 3, h, w)
            vis_batched_imgs(vis_options['saliency'], show_saliency,
                             img_metas, denorm=False)
            vis_batched_imgs(vis_options['saliency']+'_no_box', show_saliency,
                             img_metas, bboxes=None, denorm=False)

        if self.separable:
            x_saliency = self.saliency.sum(dim=2)
            y_saliency = self.saliency.sum(dim=3)
            grid = self.separable_saliency_to_grid(imgs, img_metas, x_saliency,
                                                   y_saliency, device)
        else:
            grid = self.nonseparable_saliency_to_grid(imgs, img_metas,
                                                      self.saliency, device)

        return grid


@GRID_GENERATORS.register_module()
class HomographyGlobalKDEGrid(nn.Module, RecasensSaliencyToGridMixin):
    """
        Grid generator that uses a homography based saliency map 
        which has a fixed parameter set we learn.
    """

    def __init__(self, **kwargs):
        super(HomographyGlobalKDEGrid, self).__init__()
        RecasensSaliencyToGridMixin.__init__(self, **kwargs)
        self.im_shape = kwargs.get('input_shape')
        self.homo = HomographyLayerGlobal(self.im_shape)

        ### Injecting VP from file
        bdir = "data/vps/" 
        with open(os.path.join(bdir, "vanishing_pts_argoverse_train.json"), "r") as f:
            x = json.load(f)
        with open(os.path.join(bdir + "vanishing_pts_argoverse_val.json"), "r") as f:
            y = json.load(f)
        self.v_pts_dict = {**x, **y}

    def forward(self, imgs, img_metas,
                **kwargs):
        vis_options = kwargs.get('vis_options', {})
        device = imgs.device
        v_pts_arr = [ self.v_pts_dict[x['ori_filename']] for x in img_metas ]

        for i, x in enumerate(img_metas):
            if x['flip'] == True:
                v_pts_arr[i][0] = x['img_shape'][2] - v_pts_arr[i][0]
        v_pts = torch.tensor(v_pts_arr, device=device)

        self.saliency = self.homo.forward(imgs, v_pts)
        self.saliency = F.interpolate(self.saliency, (31, 51))

        if 'saliency' in vis_options:
            h, w, _ = img_metas[0]['pad_shape']
            show_saliency = F.interpolate(self.saliency, size=(h, w),
                                          mode='bilinear', align_corners=True)
            show_saliency = 255*(show_saliency/show_saliency.max())
            show_saliency = show_saliency.expand(
                show_saliency.size(0), 3, h, w)
            vis_batched_imgs(vis_options['saliency'], show_saliency,
                             img_metas, denorm=False)
            vis_batched_imgs(vis_options['saliency']+'_no_box', show_saliency,
                             img_metas, bboxes=None, denorm=False)

        if self.separable:
            x_saliency = self.saliency.sum(dim=2)
            y_saliency = self.saliency.sum(dim=3)
            grid = self.separable_saliency_to_grid(imgs, img_metas, x_saliency,
                                                   y_saliency, device)
        else:
            grid = self.nonseparable_saliency_to_grid(imgs, img_metas,
                                                      self.saliency, device)
        return grid

@GRID_GENERATORS.register_module()
class CuboidGlobalKDEGrid(nn.Module, RecasensSaliencyToGridMixin):
    """
        Grid generator that uses a two-plane based saliency map 
        which has a fixed parameter set we learn.
    """

    def __init__(self, **kwargs):
        super(CuboidGlobalKDEGrid, self).__init__()
        RecasensSaliencyToGridMixin.__init__(self, **kwargs)
        self.im_shape = kwargs.get('input_shape')
        self.homo = CuboidLayerGlobal(self.im_shape)
        
        ### Injecting VP from file
        bdir = "data/vps/" 
        with open(os.path.join(bdir, "vanishing_pts_argoverse_train.json"), "r") as f:
            x = json.load(f)
        with open(os.path.join(bdir + "vanishing_pts_argoverse_val.json"), "r") as f:
            y = json.load(f)
        self.v_pts_dict = {**x, **y}

    def forward(self, imgs, img_metas,
                **kwargs):
        vis_options = kwargs.get('vis_options', {})
        device = imgs.device
        v_pts_arr = [ self.v_pts_dict[x['ori_filename']] for x in img_metas ]

        for i, x in enumerate(img_metas):
            if x['flip'] == True:
                v_pts_arr[i][0] = x['img_shape'][2] - v_pts_arr[i][0]
        v_pts = torch.tensor(v_pts_arr, device=device)

        self.saliency = self.homo.forward(imgs, v_pts)

        self.saliency = F.interpolate(self.saliency, (31, 51))

        if 'saliency' in vis_options:
            h, w, _ = img_metas[0]['pad_shape']
            show_saliency = F.interpolate(self.saliency, size=(h, w),
                                          mode='bilinear', align_corners=True)
            show_saliency = 255*(show_saliency/show_saliency.max())
            show_saliency = show_saliency.expand(
                show_saliency.size(0), 3, h, w)
            vis_batched_imgs(vis_options['saliency'], show_saliency,
                             img_metas, denorm=False)
            vis_batched_imgs(vis_options['saliency']+'_no_box', show_saliency,
                             img_metas, bboxes=None, denorm=False)

        if self.separable:
            x_saliency = self.saliency.sum(dim=2)
            y_saliency = self.saliency.sum(dim=3)
            grid = self.separable_saliency_to_grid(imgs, img_metas, x_saliency,
                                                   y_saliency, device)
        else:
            grid = self.nonseparable_saliency_to_grid(imgs, img_metas,
                                                      self.saliency, device)
        return grid