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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.

__device__ void computeGradientForCov3D(
    const float* inversecov3D,
    const float* dL_dinverses_cov3D,
    float* dL_dcov3D) 
{
    glm::mat3 invCov = glm::mat3(
        inversecov3D[0], inversecov3D[1], inversecov3D[2],
        inversecov3D[1], inversecov3D[3], inversecov3D[4],
        inversecov3D[2], inversecov3D[4], inversecov3D[5]
    );

    // 初始化原始矩阵梯度矩阵
    glm::mat3 dL_dCov(0.0);

    // 外两层循环遍历逆矩阵的行和列
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            // 内两层循环遍历原始矩阵的行和列
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    // 应用梯度公式
                    dL_dCov[k][l] -= invCov[i][k] * invCov[j][l] * dL_dinverses_cov3D[i * 3 + j];
                }
            }
        }
    }

    // 展平梯度矩阵到输出数组
    dL_dcov3D[0] = dL_dCov[0][0];
    dL_dcov3D[1] = dL_dCov[0][1];
    dL_dcov3D[2] = dL_dCov[0][2];
    dL_dcov3D[3] = dL_dCov[1][1];
    dL_dcov3D[4] = dL_dCov[1][2];
    dL_dcov3D[5] = dL_dCov[2][2];
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, 
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y * BLOCK_Z)
renderCUDA(
	int P, // 总高斯数目
    int W, int H,  // 体数据的宽度、高度
	const int* __restrict__ radii, // 每个高斯的半径
	const float3* __restrict__ points_xyz_image,
	const float3* __restrict__ convert_plane, // 每个体素的位置
	const float* __restrict__ inverses_cov3Ds,
	const float* __restrict__ opacities,
	const float* weight_sums,
	const float* __restrict__ dL_dpixels, //求导第一步
	float* __restrict__ dL_dinverses_cov3D,
	float3* __restrict__ dL_dpoints_xyz_im age,
	float* __restrict__ dL_dopacities,
	float* __restrict__ dL_dcov3D)
{
	// We rasterize again. Compute necessary block info.
    auto block = cg::this_thread_block();
    const uint3 pix = { block.group_index().x * BLOCK_X + block.thread_index().x,
                  block.group_index().y * BLOCK_Y + block.thread_index().y,
                  block.group_index().z * BLOCK_Z + block.thread_index().z };

    const uint idx = pix.z * W * H + pix.y * W + pix.x; // 计算当前处理点的线性索引
	if (idx >= W * H) return; // 保证不超出范围

    const float3 point = convert_plane[idx]; // 获取当前点的三维坐标
	const float weight_sum = weight_sums[idx];
    float dL_dpixel = dL_dpixels[idx]; // 当前体素的损失函数梯度


	// 直接遍历所有的高斯
    for (int i = 0; i < P; ++i) {
        float3 gauss_point = points_xyz_image[i]; // 高斯中心
        float3 d = { point.x - gauss_point.x, point.y - gauss_point.y, point.z - gauss_point.z }; 
		float distance = length(gauss_point - point); // 计算距离

        int radius = radii[i];
        if (distance > 3 * radius) continue; // 如果距离大于半径的三倍，则忽略

		const float* in_cov3D = &inverses_cov3Ds[i * 6]; // 当前高斯的反协方差矩阵
        float power = -0.5f *(in_cov3D[0] * d.x * d.x + in_cov3D[3] * d.y * d.y + in_cov3D[5] * d.z * d.z) 
			- in_cov3D[1] * d.x * d.y - in_cov3D[2] * d.x * d.z - in_cov3D[4] * d.y * d.z;
		if (power > 0.0f)
			continue;

		float G = exp(power); // 高斯权重
		float opacity = opacities[i] * G * weight_sum // 加权不透明度
        
		float dL_dopacity = dL_dpixel * (G / weight_sum);
		atomicAdd(&dL_dopacities[i], dL_dopacity);
		
		float dL_dG = dL_dpixel * opacities[i] * ((weight_sum-G)/weight_sum*weight_sum);
        float dG_din_cov3D[6] = {
            -0.5f * d.x * d.x * G,
            -d.x * d.y * G,
            -d.x * d.z * G,
            -0.5f * d.y * d.y * G,
            -d.y * d.z * G,
            -0.5f * d.z * d.z * G
        };
		for (int j = 0; j < 6; ++j) {
            atomicAdd(&dL_dinverses_cov3D[i * 6 + j], dG_din_cov3D[j] * dL_dG);
        }
		float* dL_dCov3D = dL_dcov3D + i * 6;
		computeGradientForCov3D(in_cov3D, dL_dinverses_cov3D + i * 6, dL_dCov3D);
		
		float dG_ddx = -(in_cov3D[0] * d.x + in_cov3D[1] * d.y + in_cov3D[2] * d.z) * G;
        float dG_ddy = -(in_cov3D[1] * d.x + in_cov3D[3] * d.y + in_cov3D[4] * d.z) * G;
        float dG_ddz = -(in_cov3D[2] * d.x + in_cov3D[4] * d.y + in_cov3D[5] * d.z) * G;

		atomicAdd(&dL_dpoints_xyz_image[i].x, dL_dG * dG_ddx);
		atomicAdd(&dL_dpoints_xyz_image[i].y, dL_dG * dG_ddy);
		atomicAdd(&dL_dpoints_xyz_image[i].z, dL_dG * dG_ddz);
    }
}

void BACKWARD::preprocess(
	int P, 
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, 
		radii,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		dL_dcov3D,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(int P,
	const dim3 grid, const dim3 block,
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
	float*  dL_dcov3D)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		P,
		W, H,
		radii,
		(float3*)points_xyz_image,
		(float3*)convert_plane,
		opacities,
		weight_sums,
		dL_dpixels,
		dL_dinverses_cov3D,
		dL_dpoints_xyz_image,
		dL_dopacities,
		dL_dcov3D
		);
}