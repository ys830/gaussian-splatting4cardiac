/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* cov3D_precomp,
		const float* R,
		const float* T,
		const float* init_plane,
		float3* convert_plane,
		const int W, int H,
		int* radii,
		float3* points_xyz_image,
		float* cov3Ds,
		const dim3 grid,
		uint32_t* tiles_touched);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		int P,
		int W, int H,
		const float* opacities,
		float* output_opacity);
		// const float2* points_xy_image,
		// const float* features,
		// const float4* conic_opacity,
		// float* final_T,
		// uint32_t* n_contrib,
		// const float* bg_color,
		// float* out_color);
}


#endif