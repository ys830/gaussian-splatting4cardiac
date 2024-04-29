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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

__device__ void computeEigenvalues(const float* cov3D, float* eigenvalues) {
    cusolverDnHandle_t cusolver_handle;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolver_handle), "Failed to create cusolver handle");

    float* d_cov3D;
    cudaMalloc((void**)&d_cov3D, sizeof(float) * 9);
    cudaMemcpy(d_cov3D, cov3D, sizeof(float) * 9, cudaMemcpyHostToDevice);

    float* d_eigenvalues;
    cudaMalloc((void**)&d_eigenvalues, sizeof(float) * 3);

    int lwork = 0;
    cusolverDnSsyevd_bufferSize(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, 3, d_cov3D, 3, d_eigenvalues, &lwork);
    float* workspace;
    cudaMalloc((void**)&workspace, sizeof(float) * lwork);

    int* devInfo;
    cudaMalloc((void**)&devInfo, sizeof(int));
    cusolverDnSsyevd(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, 3, d_cov3D, 3, d_eigenvalues, workspace, lwork, devInfo);

    cudaMemcpy(eigenvalues, d_eigenvalues, sizeof(float) * 3, cudaMemcpyDeviceToHost);

    cudaFree(d_cov3D);
    cudaFree(d_eigenvalues);
    cudaFree(workspace);
    cudaFree(devInfo);
    cusolverDnDestroy(cusolver_handle);
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(
	int P, //高斯分布的点的数量。
	int D, //高斯分布的维度。
	int M, //点云数量
	const float* orig_points, //点云初始化高斯得到的三维坐标
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
	float3* points_xyz_image //之后要计算的平面坐标数组
	float* cov3Ds,
	const dim3 grid, //CUDA 网格的大小
	uint32_t* tiles_touched) //记录每个高斯覆盖的图像块数量的数组
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	// 首先，初始化了一些变量，包括半径（radii）和触及到的瓦片数量（tiles_touched）
	radii[idx] = 0;
	tiles_touched[idx] = 0;


	// by ys获取目标平面上的坐标（初始平面+RT矩阵）
    int point_idx = idx * 3;
    float init_point[3] = {init_plane[point_idx], init_plane[point_idx + 1], init_plane[point_idx + 2]};
    float3 transformed_point;
    
    transformPoint(init_point, R, T, &transformed_point);

    // 存储变换后的坐标到输出数组
    convert_plane[idx] = transformed_point;


	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	float eigenvalues[3];
	//通过三维协方差矩阵计算特征值
	computeEigenvalues(cov3D,eigenvalues);
	float lambda1 = eigenvalues[0];
	float lambda2 = eigenvalues[1];
	float lambda3 = eigenvalues[2];
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2, lambda3)));
	//不投影，直接判断以三维点云的点为中心，计算出的半径为半径，与3D tile的交点数目
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	get3DRect(p_orig, my_radius, rect_min, rect_max, grid); //判断交叉
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y)*(rect_max.z - rect_min.z) == 0)
		return;


	radii[idx] = my_radius;
	// 直接从原始点数组中获取点的坐标，并存储到输出数组中
    points_xyz_image[idx] = p_orig;
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x) * (rect_max.z-rect.min.z); //3D矩阵的体积
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y * BLOCK_Z)
renderCUDA(
    int P, // 总高斯数目
    int W, int H,  // 体数据的宽度、高度
    const int* __restrict__ radii, // 每个高斯的半径
    const float3* __restrict__ points_xyz_image, // 高斯点的位置信息
    const float3* __restrict__ convert_plane, // 每个体素的位置
    const float* __restrict__ opacities, // 每个高斯的不透明度
    float* __restrict__ output_opacity // 输出的不透明度数组
) {
    auto block = cg::this_thread_block();
    uint3 pix = { block.group_index().x * BLOCK_X + block.thread_index().x,
                  block.group_index().y * BLOCK_Y + block.thread_index().y,
                  block.group_index().z * BLOCK_Z + block.thread_index().z };

    uint idx = pix.z * W * H + pix.y * W + pix.x; // 计算当前处理点的线性索引
    float3 point = convert_plane[idx]; // 获取当前点的三维坐标
    float opacity_acc = 0.0; // 累积不透明度
    float weight_sum = 0.0; // 权重和

    // 直接遍历所有的高斯
    for (int i = 0; i < P; ++i) {
        float3 gauss_point = points_xyz_image[i]; // 高斯中心
        float distance = length(gauss_point - point); // 计算距离

        int radius = radii[i];
        if (distance > 3 * radius) continue; // 如果距离大于半径的三倍，则忽略

        float influence = exp(-distance * distance / (radius * radius));
        float opacity = opacities[i] * influence; // 加权不透明度

        opacity_acc += opacity;
        weight_sum += influence;
    }

    if (weight_sum > 0) {
        output_opacity[idx] = opacity_acc / weight_sum; // 计算加权平均不透明度
    } else {
        output_opacity[idx] = 0.0; // 如果没有任何有效的高斯影响，设置不透明度为0
    }
}

void FORWARD::render(
    const dim3 grid, dim3 block,
    int P, // 总高斯数目
    int W, int H, // 体数据的宽度、高度
    const float* opacities, // 每个高斯的不透明度数组
    float* output_opacity // 输出的不透明度数组
) {
    renderCUDA<NUM_CHANNELS> <<<grid, block>>> (
        P,
        W, H,
        opacities,
        output_opacity);
}


void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* cov3D_precomp,
	const float* R,
	const float* T,
	const float* init_plane,
	const int W, int H,
	int* radii,
	float* cov3Ds,
	const dim3 grid,
	uint32_t* tiles_touched)
{
	//check here
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		cov3D_precomp,
		init_plane,
		W, H,
		radii,
		R, T,
		cov3Ds,
		grid,
		tiles_touched
		);
}