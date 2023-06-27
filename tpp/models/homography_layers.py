import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import kornia as K

class HomographySaliencyParamsNet(nn.Module):
    def __init__(self, 
            backbone='resnet18', 
            scale_factor=0.2,
            min_theta=110,
            max_theta=120,
            min_alpha=0.2,
            max_alpha=0.4,
            min_p=1, max_p=4
            ):
        self.min_theta = torch.deg2rad(torch.tensor([min_theta])).item()        
        self.max_theta = torch.deg2rad(torch.tensor([max_theta])).item()        
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha       
        self.min_p = min_p
        self.max_p = max_p
        super(HomographySaliencyParamsNet, self).__init__()
        self.scale_factor = scale_factor
        if backbone == "resnet18":
            self.backbone = torchvision.models.resnet18(pretrained=True)
            self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))
        elif backbone == "resnet50":
            self.backbone = torchvision.models.resnet50(pretrained=True)
            self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))
        self.predictor = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 4),
            nn.LeakyReLU()
        )
        
    def forward(self, imgs):
        device = imgs.device
        self.backbone.to(device)
        self.predictor.to(device)
        scaled_imgs = F.interpolate(imgs, scale_factor=self.scale_factor)
        feats = self.backbone(scaled_imgs).squeeze(-1).squeeze(-1)
        outs = self.predictor(feats)
        thetas_l = torch.clamp(outs[:, 0], min=self.min_theta, max=self.max_theta)
        thetas_r = torch.clamp(outs[:, 1], min=self.min_theta, max=self.max_theta)
        alphas = torch.clamp(outs[:, 2], min=self.min_alpha, max=self.max_alpha)
        ps = torch.clamp(outs[:, 3], min=self.min_p, max=self.max_p)                
        return thetas_l, thetas_r, alphas, ps

class HomographyLayer(nn.Module):

    def __init__(self, im_shape):
        super(HomographyLayer, self).__init__()
        self.im_shape = im_shape
        self.init_map = torch.zeros(self.im_shape)
        for r in range(self.init_map.shape[0]):
            self.init_map[r, :] = (self.init_map.shape[0]*1.0 - r) / self.init_map.shape[0]*1.0
        self.init_map = self.init_map.unsqueeze(0).unsqueeze(0)
        self.init_map = self.init_map - 1

    def parametric_homography(self, v_pts, thetas_l, thetas_r, alphas):
        h, w = self.im_shape
        B = v_pts.shape[0]
        p_l = torch.zeros(B, 2, device=self.device)
        p_l[:, 1] = v_pts[:, 1] + torch.mul(v_pts[:, 0], 1./torch.tan(thetas_l))   
        p_r = torch.zeros(B, 2, device=self.device)
        p_r[:, 0] += w - 1
        p_r[:, 1] = v_pts[:, 1] + torch.mul(w - 1 - v_pts[:, 0], 1./torch.tan(thetas_r)) 
        p1 = alphas[:, None]*p_l + (1 - alphas[:, None])*v_pts
        p2 = alphas[:, None]*p_r + (1 - alphas[:, None])*v_pts
        pt_src = torch.zeros(B, 4, 2, device=self.device)
        pt_src[:, 0, :] = p1
        pt_src[:, 1, :] = p2
        pt_src[:, 2, :] = torch.tensor([w - 1, h - 1], device=self.device).repeat(B, 1)
        pt_src[:, 3, :] = torch.tensor([0, h - 1], device=self.device).repeat(B, 1)
        pt_dst = torch.tensor([[
            [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
        ]], dtype=torch.float32, device=self.device).repeat(B, 1, 1)
        M = K.geometry.get_perspective_transform(pt_dst, pt_src)
        return pt_src, M
    
    def forward(self, imgs, v_pts, thetas_l, thetas_r, alphas, ps, return_homo=False):
        self.device = imgs.device
        B = imgs.shape[0]
        points_src, M = self.parametric_homography(v_pts, thetas_l, thetas_r, alphas)
        ps = ps.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        ps = ps.expand(B, 1, self.init_map.shape[2], self.init_map.shape[3])
        init_map = self.init_map.to(self.device).repeat(imgs.shape[0], 1, 1, 1)        
        init_map = torch.exp( torch.mul(ps, init_map) )
        map_warp: torch.tensor = K.geometry.warp_perspective(init_map.float(), M, 
                                    dsize=( round(self.im_shape[0]), round(self.im_shape[1]) ))
        if return_homo == True:
            return map_warp, (points_src, M)
        return map_warp

class HomographyLayerGlobal(nn.Module):

    def __init__(self, im_shape,
            min_theta=110,
            max_theta=120,
            min_alpha=0.2,
            max_alpha=0.4,
            min_p=1, max_p=5
    ):
        super(HomographyLayerGlobal, self).__init__()
        self.im_shape = im_shape
        self.init_map = torch.zeros(self.im_shape)
        for r in range(self.init_map.shape[0]):
            self.init_map[r, :] = (self.init_map.shape[0]*1.0 - r) / self.init_map.shape[0]*1.0
        self.init_map = self.init_map.unsqueeze(0).unsqueeze(0)
        self.init_map = self.init_map - 1
        
        min_theta = np.deg2rad(min_theta)
        max_theta = np.deg2rad(max_theta)

        self.theta_l = nn.Parameter(torch.Tensor([1])*(min_theta + max_theta)/2, requires_grad=True)
        self.theta_r = nn.Parameter(torch.Tensor([1])*(min_theta + max_theta)/2, requires_grad=True)
        self.alpha = nn.Parameter(torch.Tensor([1])*(min_alpha + max_alpha)/2, requires_grad=True)
        self.p = nn.Parameter(torch.Tensor([1])*(min_p + max_p)/2, requires_grad=True)

    def parametric_homography(self, v_pts, thetas_l, thetas_r, alphas):
        h, w = self.im_shape
        B = v_pts.shape[0]
        p_l = torch.zeros(B, 2, device=self.device)
        p_l[:, 1] = v_pts[:, 1] + torch.mul(v_pts[:, 0], 1./torch.tan(thetas_l))   
        p_r = torch.zeros(B, 2, device=self.device)
        p_r[:, 0] += w - 1
        p_r[:, 1] = v_pts[:, 1] + torch.mul(w - 1 - v_pts[:, 0], 1./torch.tan(thetas_r))
        p1 = alphas[:, None]*p_l + (1 - alphas[:, None])*v_pts
        p2 = alphas[:, None]*p_r + (1 - alphas[:, None])*v_pts
        pt_src = torch.zeros(B, 4, 2, device=self.device)
        pt_src[:, 0, :] = p1
        pt_src[:, 1, :] = p2
        pt_src[:, 2, :] = torch.tensor([w - 1, h - 1], device=self.device).repeat(B, 1)
        pt_src[:, 3, :] = torch.tensor([0, h - 1], device=self.device).repeat(B, 1)
        pt_dst = torch.tensor([[
            [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
        ]], dtype=torch.float32, device=self.device).repeat(B, 1, 1)
        M = K.geometry.get_perspective_transform(pt_dst, pt_src)
        return pt_src, M
    
    def forward(self, imgs, v_pts):
        self.device = imgs.device
        B = imgs.shape[0]
        thetas_l = self.theta_l.expand(B).to(self.device)
        thetas_r = self.theta_r.expand(B).to(self.device)
        alphas = self.alpha.expand(B).to(self.device)
        points_src, M = self.parametric_homography(v_pts, thetas_l, thetas_r, alphas)
        ps = self.p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(self.device)
        ps = ps.expand(B, 1, self.init_map.shape[2], self.init_map.shape[3])
        init_map = self.init_map.to(self.device).repeat(imgs.shape[0], 1, 1, 1)        
        init_map = torch.exp( torch.mul(ps, init_map) )
        map_warp: torch.tensor = K.geometry.warp_perspective(init_map.float(), M, 
                                    dsize=( round(self.im_shape[0]), round(self.im_shape[1]) ))
        return map_warp

class CuboidLayerGlobal(nn.Module):
    def __init__(self, im_shape,
            min_theta=110,
            max_theta=120,           
            min_alpha=0.2,
            max_alpha=0.4,            
            min_p=1, 
            max_p=5,
            lambd=0.97
    ):
        super(CuboidLayerGlobal, self).__init__()
        self.im_shape = im_shape
        self.init_map = torch.zeros(self.im_shape)
        for r in range(self.init_map.shape[0]):
            self.init_map[r, :] = (self.init_map.shape[0]*1.0 - r) / self.init_map.shape[0]*1.0
        self.init_map = self.init_map.unsqueeze(0).unsqueeze(0)
        self.init_map = self.init_map - 1
        
        min_theta = np.deg2rad(min_theta)
        max_theta = np.deg2rad(max_theta)

        self.theta_l = nn.Parameter(torch.Tensor([1])*(min_theta + max_theta)/2, requires_grad=True)
        self.theta_r = nn.Parameter(torch.Tensor([1])*(min_theta + max_theta)/2, requires_grad=True)
        self.alpha_1 = nn.Parameter(torch.Tensor([1])*(min_alpha + max_alpha)/2, requires_grad=True)
        self.alpha_2 = nn.Parameter(torch.Tensor([1])*(min_alpha + max_alpha)/2, requires_grad=True)
        self.p = nn.Parameter(torch.Tensor([1])*(min_p + max_p)/2, requires_grad=True)

        self.lambd = nn.Parameter(torch.Tensor([1])*lambd, requires_grad=True)

        self.theta_top_l = nn.Parameter(torch.Tensor([1])*(min_theta + max_theta)/2, requires_grad=True)
        self.theta_top_r = nn.Parameter(torch.Tensor([1])*(min_theta + max_theta)/2, requires_grad=True)
        self.alpha_top_1 = nn.Parameter(torch.Tensor([1])*(min_alpha + max_alpha)/2, requires_grad=True)
        self.alpha_top_2 = nn.Parameter(torch.Tensor([1])*(min_alpha + max_alpha)/2, requires_grad=True)
        self.p_top = nn.Parameter(torch.Tensor([1])*(min_p + max_p)/2, requires_grad=True)

    def parametric_homography(self, v_pts, thetas_l, thetas_r, alphas_1, alphas_2, bottom):
        h, w = self.im_shape
        B = v_pts.shape[0]
        p_l = torch.zeros(B, 2, device=self.device)
        p_l[:, 1] = v_pts[:, 1] + torch.mul(v_pts[:, 0], 1./torch.tan(thetas_l))   
        p_r = torch.zeros(B, 2, device=self.device)
        p_r[:, 0] += w - 1
        p_r[:, 1] = v_pts[:, 1] + torch.mul(w - 1 - v_pts[:, 0], 1./torch.tan(thetas_r))
        p1 = alphas_1[:, None]*p_l + (1 - alphas_1[:, None])*v_pts
        p2 = alphas_2[:, None]*p_r + (1 - alphas_2[:, None])*v_pts
        pt_src = torch.zeros(B, 4, 2, device=self.device)
        if bottom:
            pt_src[:, 0, :] = p1
            pt_src[:, 1, :] = p2
            pt_src[:, 2, :] = torch.tensor([w - 1, h - 1], device=self.device).repeat(B, 1)
            pt_src[:, 3, :] = torch.tensor([0, h - 1], device=self.device).repeat(B, 1)
        else:
            pt_src[:, 0, :] = torch.tensor([0, 0], device=self.device).repeat(B, 1)
            pt_src[:, 1, :] = torch.tensor([w - 1, 0], device=self.device).repeat(B, 1)
            pt_src[:, 2, :] = p2
            pt_src[:, 3, :] = p1
        pt_dst = torch.tensor([[
            [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
        ]], dtype=torch.float32, device=self.device).repeat(B, 1, 1)
        M = K.geometry.get_perspective_transform(pt_dst, pt_src)
        return pt_src, M
    
    def map_warp(self, B, v_pts, thetas_l, thetas_r, alphas_1, alphas_2, ps, bottom):
        points_src, M = self.parametric_homography(v_pts, thetas_l, thetas_r, alphas_1, alphas_2, bottom)
        init_map = self.init_map.to(self.device).repeat(B, 1, 1, 1)        
        init_map = torch.exp( torch.mul(ps, init_map) )
        map_warp: torch.tensor = K.geometry.warp_perspective(init_map.float(), M, 
                                    dsize=( round(self.im_shape[0]), round(self.im_shape[1]) ))
        return map_warp

    def forward(self, imgs, v_pts):
        self.device = imgs.device
        B = imgs.shape[0]
        thetas_l = self.theta_l.expand(B).to(self.device)
        thetas_r = self.theta_r.expand(B).to(self.device)
        alphas_1 = self.alpha_1.expand(B).to(self.device)
        alphas_2 = self.alpha_2.expand(B).to(self.device)

        ps = self.p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(self.device)
        ps = ps.expand(B, 1, self.init_map.shape[2], self.init_map.shape[3])        
        bottom = self.map_warp(B, v_pts, thetas_l, thetas_r, alphas_1, alphas_2, ps, bottom=True)

        thetas_top_l = self.theta_l.expand(B).to(self.device)
        thetas_top_r = self.theta_r.expand(B).to(self.device)
        alphas_top_1 = self.alpha_top_1.expand(B).to(self.device)
        alphas_top_2 = self.alpha_top_2.expand(B).to(self.device)
        ps_top = self.p_top.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(self.device)
        ps_top = ps_top.expand(B, 1, self.init_map.shape[2], self.init_map.shape[3])        
        top = self.map_warp(B, v_pts, thetas_top_l, thetas_top_r, alphas_top_1, alphas_top_2, ps_top, bottom=False)

        lambd = (1.0 - self.lambd).to(self.device)
        map_warp = bottom + lambd * top 

        return map_warp