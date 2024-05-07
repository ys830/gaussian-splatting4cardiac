#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 创建一个与输入点云（高斯模型）大小相同的零张量，用于记录屏幕空间中的点的位置。这个张量将用于计算对于屏幕空间坐标的梯度
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0 #[初始点云数目, 3]
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        image_length=int(viewpoint_camera.image_length),
        # tanfovx=tanfovx,
        # tanfovy=tanfovy,
        # bg=bg_color, # 背景张量 [0,0,0]
        scale_modifier=scaling_modifier, # 缩放修正因子 1.0
        # viewmatrix=viewpoint_camera.world_view_transform, # 视图矩阵 [4,4]
        # projmatrix=viewpoint_camera.full_proj_transform, # 投影矩阵 [4,4]
        R = viewpoint_camera.R,
        T = viewpoint_camera.T,
        # sh_degree=pc.active_sh_degree,  # SH（球谐函数）的阶数
        # campos=viewpoint_camera.camera_center, # 相机位置 [3]
        init_plane = viewpoint_camera.plane,
        # prefiltered=False,
        debug=pipe.debug
    )
  
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz #高斯分布的三维坐标 [点云数量，3]
    means2D = screenspace_points #屏幕空间坐标 [点云数量，3]
    opacity = pc.get_opacity #透明度 [点云数量，1]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else: 
        scales = pc.get_scaling #[点云数量，3]
        rotations = pc.get_rotation #[点云数量，4]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    # colors_precomp = None
    # if override_color is None:
    #     if pipe.convert_SHs_python:
    #         shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    #         dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    #     else:
    #         shs = pc.get_features
    # else:
    #     colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # 调用光栅化器，将高斯分布投影到屏幕上，获得渲染图像和每个高斯分布在屏幕上的半径。
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        # shs = shs,
        # colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
