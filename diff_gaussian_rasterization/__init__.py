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

from typing import NamedTuple
import torch.nn as nn
import torch
import os
from torch.utils.cpp_extension import load
import time

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
extra_opts = ["-I" + os.path.join(base_dir, "third_party/glm/")]
_C = load(name="ray_rasterizer",
            sources=[
            os.path.join(base_dir,"cuda_rasterizer/rasterizer_impl.cu"),
            os.path.join(base_dir,"cuda_rasterizer/forward.cu"),
            os.path.join(base_dir,"cuda_rasterizer/backward.cu"),
            os.path.join(base_dir,"rasterize_points.cu"),
            os.path.join(base_dir,"ext.cu")],
            extra_cuda_cflags=extra_opts)

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    bvh_nodes,
    bvh_aabbs
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        bvh_nodes,
        bvh_aabbs
    )

def compute_ray_bbox_intersection(ray_pos, ray_dir, bbox_min, bbox_max):
    t_min = torch.div((bbox_min - ray_pos), ray_dir)
    t_max = torch.div((bbox_max - ray_pos), ray_dir)

    t1 = torch.minimum(t_min, t_max)
    t2 = torch.maximum(t_min, t_max)

    t_near = torch.max(t1)
    t_far = torch.min(t2)

    if t_far < 0:
        return False, t_near, t_far
    if t_near > t_far:
        return False, t_near, t_far

    return True, t_near, t_far

def find_closest(bvh_nodes, bvh_aabbs, ray_pos, ray_dir, node_addr, means3D):
    if bvh_nodes[node_addr][3] != -1:
        # Is leaf
        hit, t_near, t_far = compute_ray_bbox_intersection(ray_pos, ray_dir, bvh_aabbs[node_addr, :, 0], bvh_aabbs[node_addr, :, 1])
        assert hit
        print(f"Hit leaf for t=({t_near}, {t_far}), node {node_addr} and object {bvh_nodes[node_addr][3]}")
    else:
        h1_hit, h1_t_near, h1_t_far = compute_ray_bbox_intersection(ray_pos, ray_dir,  bvh_aabbs[bvh_nodes[node_addr, 1], :, 0], bvh_aabbs[bvh_nodes[node_addr, 1], :, 1])
        h2_hit, h2_t_near, h2_t_far = compute_ray_bbox_intersection(ray_pos, ray_dir,  bvh_aabbs[bvh_nodes[node_addr, 2], :, 0], bvh_aabbs[bvh_nodes[node_addr, 2], :, 1])
        if h1_hit:
            find_closest(bvh_nodes, bvh_aabbs, ray_pos, ray_dir, bvh_nodes[node_addr, 1], means3D)
        if h2_hit:
            find_closest(bvh_nodes, bvh_aabbs, ray_pos, ray_dir, bvh_nodes[node_addr, 2], means3D)

def python_rasterize(bvh_nodes, bvh_aabbs, viewmatrix, projmatrix, tanfovx, tanfovy, campos, znear, zfar, means3D, cov3Ds_precomp, scales, rotations, image_width, image_height):
    viewmatrix = viewmatrix.T
    projmatrix = projmatrix.T

    top = tanfovy * 1.0
    bottom = -top
    right = tanfovx * 1.0
    left = -right

    print(image_width, image_height)

    for p_x in range(image_width):
        for p_y in range(image_height):

            pixel_view = torch.tensor([right * 2.0 * (p_x / image_width) - right, top * 2.0 * (p_y / image_height) - top, -1.0, 1.0]).cuda()
            pixel_world = viewmatrix.inverse() @pixel_view 
            pixel_world = pixel_world[:3]

            ray_pos = campos
            ray_dir = pixel_world - ray_pos
            ray_dir = ray_dir / torch.norm(ray_dir)
            # print("Ray position", ray_pos)
            # print("Ray direction", ray_dir)
            # print("Pixel coord", pixel_world)

    find_closest(bvh_nodes, bvh_aabbs, ray_pos, ray_dir, 0, means3D)

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        bvh_nodes,
        bvh_aabbs
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.znear,
            raster_settings.zfar,
            raster_settings.viewmatrix,
            raster_settings.viewmatrix_inv,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            bvh_nodes,
            bvh_aabbs,
            raster_settings.debug
        )

        # python_rasterize(bvh_nodes, bvh_aabbs, raster_settings.viewmatrix, raster_settings.projmatrix, raster_settings.tanfovx, raster_settings.tanfovy, raster_settings.campos, raster_settings.znear, raster_settings.zfar, means3D, cov3Ds_precomp, scales, rotations, raster_settings.image_width, raster_settings.image_height)

        # Invoke C++/CUDA rasterizer
        start_time = time.time()
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
        print(color)
        end_time = time.time()
        print(f"Rasterize took {end_time - start_time} seconds")

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    znear : float
    zfar : float
    viewmatrix : torch.Tensor
    viewmatrix_inv : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, bvh_nodes, bvh_aabbs, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
            bvh_nodes,
            bvh_aabbs
        )

