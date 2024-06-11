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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		// static void markVisible(
		// 	int P,
		// 	float* means3D,
		// 	float* viewmatrix,
		// 	float* projmatrix,
		// 	bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P,
			const float* orig_points,
			const int width, int height,
			const float* means3D,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* R,
			const float* T,
			const float* init_plane,
			float* output_opacity,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P, int NR,
			const int width, int height,
			const float* means3D,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* R,
			const float* T,
			const float* init_plane,
			const int* radii,
			char* geom_buffer,
			char* image_buffer,
			const float* dL_dpix,
			float* dL_dmean2D, //對
			float* dL_dinverses_cov3D,
			float* dL_dopacity,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dscale,
			float* dL_drot,
			bool debug);
	};
};

#endif