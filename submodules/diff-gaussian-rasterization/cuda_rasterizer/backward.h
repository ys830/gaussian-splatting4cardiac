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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(int P,
		const dim3 grid, dim3 block,
		int W, int H,
		const int* radii,
		const float3* points_xyz_image,
		const float3*  convert_plane,
		const float* opacities,
		const float* weight_sums,
		const float* dL_dpixels,
		float3* dL_dpoints_xyz_image,
		float* dL_dinverses_cov3D,
		float*  dL_dopacities,
		float*  dL_dcov3D);

	void preprocess(
		int P,
		const int* radii,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* dL_dcov3D,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot);
}

#endif