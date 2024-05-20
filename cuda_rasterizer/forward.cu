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

#include <iomanip>
#include <unistd.h>
#include <tuple>
#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <fstream>
#include <vector>

#include <chrono>

#include <sstream>
#include <optix.h>
#include <optix_host.h>
#include <optix_types.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

namespace cg = cooperative_groups;

#define OPTIX_CHECK(call) \
{ \
  const OptixResult result = call; \
  if (result != OPTIX_SUCCESS) \
  { \
    std::ostringstream message; \
    message << "ERROR: " << " (" << result << ")"; \
    throw std::runtime_error(message.str()); \
  } \
}

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

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view)) {
		depths[idx] = p_view.z;
		return;
	}

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

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

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	// printf("IDX:%d Radius:%f, Px:%f, Py:%f, L1:%f, L2:%f, cov00:%f, cov01:%f, cov11:%f\n", idx, my_radius, point_image.x, point_image.y, lambda1, lambda2, cov.x, cov.y, cov.z);
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;

		// if (last_contributor > 0) printf("Pix (%d, %d): %d\n", pix.x, pix.y, last_contributor);
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

char* readFile(const char* filename, size_t* size) {
    // Open the file in binary mode and get its size
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Unable to open file " << filename << std::endl;
        return nullptr;
    }

    // Get the size of the file
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Allocate memory for the C string
    char* buffer = new char[fileSize + 1]; // +1 for the null terminator

    // Read the file content into the buffer
    if (!file.read(buffer, fileSize)) {
        std::cerr << "Error reading file " << filename << std::endl;
        delete[] buffer; // Free allocated memory in case of failure
        return nullptr;
    }
    buffer[fileSize] = '\0'; // Null-terminate the C string

    // Write the file size to the provided size variable
    *size = fileSize;

    return buffer;
}

struct RayGenData
{
    // No data needed
};


struct MissData
{
	// No data needed
};


struct HitGroupData
{
    // No data needed
};

struct CallablesData
{
    // No data needed
};

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenRecord;
typedef SbtRecord<MissData>       MissRecord;
typedef SbtRecord<HitGroupData>   HitGroupRecord;
typedef SbtRecord<CallablesData>  CallablesRecord;

struct Params
{
    CUdeviceptr            gaussians;
    CUdeviceptr            n_gaussians;
    unsigned int           width;
    unsigned int           height;
    float                  tanfovx;
    float                  tanfovy;
    float*                 viewmatrix_inv;
    float*                 cam_pos;
    float*				   depths;
    OptixTraversableHandle handle;
};

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */ )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}

template <uint32_t CHANNELS>
__global__ void composing(
	int W, int H,
	int *d_gaussians,
	float *d_depths,
	int *n_gaussians,
	float2 *means2D,
	const float *bg_color,
	const float* colors_precomp,
	float4 *conic_opacity,
	float *out_color
) {
	int gaussians[1024];
	float depths[1024];

	int pixel_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (pixel_id >= W * H) {
		return;
	}
	int x = pixel_id % W;
	int y = pixel_id / W;

    int N_GAUSSIANS = n_gaussians[pixel_id];

    // Collect
    for (int i = 0; i < N_GAUSSIANS; i++) {
    	gaussians[i] = d_gaussians[i * W * H + pixel_id];
    	depths[i] = d_depths[d_gaussians[i * W * H + pixel_id]];
    }

    // Sort
	// __syncthreads();
    // thrust::sort_by_key(thrust::seq, depths, depths + N_GAUSSIANS, gaussians);

    // Compose
	__syncthreads();
	float accumulated_color[CHANNELS] = { 0.0f };
	float T = 1.0f;  // Start with full transmittance
	// Compute the color of the pixel (cf. "Surface Splatting" by Zwicker et al., 2001)
    for (int i = 0; i < N_GAUSSIANS; i++) {
        int index = gaussians[i];
        float4 con_o = conic_opacity[index];
        float2 gaussian_mean = means2D[index];

        // Compute power using conic matrix
        float2 d = {x - gaussian_mean.x, y - gaussian_mean.y};
        float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
        if (power > 0.0f)
            continue;

        // Compute alpha
        float alpha = min(exp(power) * con_o.w, 0.99f);
        if (alpha < 1.0f / 255.0f)
			continue;	// Skip if alpha is too small

		// Update transmittance
        float test_T = T * (1 - alpha);
        if (test_T < 0.0001f)
            break;  	// Stop if transmittance is negligible

		// Accumulate color
		for (int ch = 0; ch < CHANNELS; ch++) {
			accumulated_color[ch] += colors_precomp[index * CHANNELS + ch] * alpha * T;
		}

        T = test_T;  	// Update the transmittance
    }

	__syncthreads();
    int pix_id = y * W + x;
    for (int ch = 0; ch < CHANNELS; ch++) {
		out_color[ch * H * W + pix_id] = accumulated_color[ch] + bg_color[ch] * T;
    }
}

__device__ float2 intersect_ray_bbox(
	float3 ray_origin, 
	float3 ray_direction, 
	float3 bbox_min, 
	float3 bbox_max
) {
    float3 inv_dir = make_float3(1.0f) / ray_direction;
    float3 t_min = (bbox_min - ray_origin) * inv_dir;
    float3 t_max = (bbox_max - ray_origin) * inv_dir;
    float3 t1 = fminf(t_min, t_max);
    float3 t2 = fmaxf(t_min, t_max);

    float t_near = fmaxf(fmaxf(t1.x, t1.y), t1.z);
    float t_far = fminf(fminf(t2.x, t2.y), t2.z);

    return make_float2(t_near, t_far);
}

template <uint32_t CHANNELS>
__device__ void composing_3D(
    const int H, const int W,
    // float3 ray_origin,
    // float3 ray_direction,
    // float2 *t_bounds,
	float *d_aabbBuffer,
    float *viewmatrix_inv,
    float *cam_pos,
	int *d_gaussians,
    int *n_gaussians,
    float3 *mean3D,
    float *cov3D,
    const float *bg_color,
    const float* colors_precomp,
    float4* conic_opacity,
    float *out_color
) {
    int pixel_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_id >= W * H) return;

    int x = pixel_id % W;
    int y = pixel_id / W;

    int N_GAUSSIANS = n_gaussians[pixel_id];
    if (N_GAUSSIANS == 0) return;  // Skip processing if no Gaussians

	// Compute ray origin
	float3 ray_origin = make_float3(cam_pos[0], cam_pos[1], cam_pos[2]);

	// Compute ray direction
    float3 ray_direction = normalize(make_float3(
        (x - W / 2.0f) * tanfovx,
        (y - H / 2.0f) * tanfovy,
        -1.0f
    ));
	ray_direction = multiply_direction(viewmatrix_inv, ray_direction);
    ray_direction = normalize(ray_direction);

	// Compute intersections
	float2 *t_bounds = new float2[n_gaussians[pixel_id]];
	for (int i = 0; i < N_GAUSSIANS; i++) {
		int idx = d_gaussians[i * W * H + pixel_id];
		float3 bbox_min = make_float3(d_aabbBuffer[idx * 6 + 0], d_aabbBuffer[idx * 6 + 1], d_aabbBuffer[idx * 6 + 2]);
		float3 bbox_max = make_float3(d_aabbBuffer[idx * 6 + 3], d_aabbBuffer[idx * 6 + 4], d_aabbBuffer[idx * 6 + 5]);

		t_bounds[i] = intersect_ray_bbox(ray_origin, ray_direction, bbox_min, bbox_max);
	}

    // Temporary arrays for sorting
    float *t_near = new float[N_GAUSSIANS];
    int *gaussians = new int[N_GAUSSIANS];

    // Collect data for sorting
    for (int i = 0; i < N_GAUSSIANS; i++) {
        int idx = d_gaussians[i * W * H + pixel_id];
        gaussians[i] = idx;
        t_near[i] = t_bounds[idx].x;
    }

    // Sort Gaussians by t_near
	__syncthreads();
    thrust::device_ptr<int> dev_gaussians(gaussians);
    thrust::device_ptr<float> dev_t_near(t_near);
    thrust::sort_by_key(thrust::device, dev_t_near, dev_t_near + N_GAUSSIANS, dev_gaussians);

	// Compose
	__syncthreads();
	float accumulated_color[CHANNELS] = {0.0f};
    float T = 1.0f;  // Full transmittance initially
    for (int i = 0; i < N_GAUSSIANS; i++) {
        int idx = gaussians[i];
        float3 gaussian_mean = mean3D[idx];
        float covariance = cov3D[idx];  // Assuming isotropic for simplification
        float4 color_and_opacity = conic_opacity[idx];
        float alpha = color_and_opacity.w;

        // Ray-Gaussian intersection bounds
        float t_start = t_bounds[idx].x;
        float t_end = t_bounds[idx].y;
        if (t_start > t_end) continue;

		// Discretization steps
        float dt = (t_end - t_start) / 10.0f;

        // March along the ray within the bounds of the current Gaussian
        for (float t_curr = t_start; t_curr <= t_end; t_curr += dt) {
			float3 sample_point;
            sample_point.x = ray_origin.x + t_curr * ray_direction.x;
            sample_point.y = ray_origin.y + t_curr * ray_direction.y;
            sample_point.z = ray_origin.z + t_curr * ray_direction.z;

			float3 diff;
            diff.x = sample_point.x - gaussian_mean.x;
            diff.y = sample_point.y - gaussian_mean.y;
            diff.z = sample_point.z - gaussian_mean.z;

			float distance2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
			float density = expf(-0.5f * distance2 / covariance);

            float sample_alpha = density * alpha * dt;
            if (sample_alpha < 0.0001f) continue; 		// Skip negligible contributions
            sample_alpha = min(sample_alpha, 1.0f - T); // Clamp to remaining transmittance

            // Blend colors
            for (int ch = 0; ch < CHANNELS; ch++) {
                accumulated_color[ch] += colors_precomp[idx * CHANNELS + ch] * sample_alpha;
            }

            T *= (1.0f - sample_alpha); // Update remaining transmittance
            if (T < 0.01f) break; 		// Early termination if opaque
        }

        if (T < 0.01f) break; // Early termination if opaque
    }

	__syncthreads();
    int pix_id = y * W + x;
    for (int ch = 0; ch < CHANNELS; ch++) {
        out_color[pix_id * CHANNELS + ch] = accumulated_color[ch] + bg_color[ch] * T;
    }
}

template <uint32_t CHANNELS>
void build_optix_bvh(const int W, const int H, const int P, float *d_aabbBuffer, float tanfovx, float tanfovy, float *viewmatrix_inv, float *cam_pos, float *depths,
									float2 *means2D,
									const float *bg_color,
									const float* colors_precomp,
									float4 *conic_opacity,
									float *out_color) {
	auto start = std::chrono::high_resolution_clock::now();
	printf("Building Optix BVH\n");

	// Create cuda stream
	cudaStream_t cuStream;
    CHECK_CUDA( cudaStreamCreate(&cuStream), true );

	// Create Optix context
	OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
	OptixDeviceContext optixContext;
	cudaFree(0);
	CUcontext cuCtx = 0;
	OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &optixContext ) );

	// Create Module, ProgramGroup, and Pipeline

	// First create module
	OptixPipelineCompileOptions pipeline_compile_options;
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT; 
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;

    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipeline_compile_options.numPayloadValues = 6; // FIXME
    pipeline_compile_options.numAttributeValues = 0; // FIXME
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.usesPrimitiveTypeFlags = 0;
    pipeline_compile_options.allowOpacityMicromaps = false;

    // OptixModuleCompileBoundValueEntry boundValue = {};
    // boundValue.pipelineParamOffsetInBytes = offsetof( Params, ao );
    // boundValue.sizeInBytes                = sizeof( Params::ao );
    // boundValue.boundValuePtr              = &state.renderAO;
    // boundValue.annotation                 = "ao";
    // module_compile_options.boundValues    = &boundValue;
    module_compile_options.numBoundValues = 0;

    size_t      inputSize = 0;
    const char* input = readFile("submodules/diff-gaussian-rasterization/cuda_rasterizer/optix.ptx", &inputSize);

    size_t logStringSize = 1024;
    char logString[logStringSize];

    OptixModule ptx_module;

    OPTIX_CHECK( optixModuleCreate(
        optixContext,
        &module_compile_options,
        &pipeline_compile_options,
        input,
        inputSize,
        logString, &logStringSize,
        &ptx_module
    ) );

    printf("Module compile %d: %s", logStringSize, logString);

	// Then create program groups
    OptixProgramGroup raygen_prog_group;
    OptixProgramGroup hit_prog_group;
    OptixProgramGroup miss_prog_group;
    OptixProgramGroup callables_prog_group;

    OptixProgramGroupOptions  program_group_options = {};

    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

		logStringSize = 1024;
		logString[0] = 0;
        OPTIX_CHECK( optixProgramGroupCreate(
            optixContext, &raygen_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            logString, &logStringSize,
            &raygen_prog_group
        ) );
    }

    printf("Raygen Program Group %d: %s", logStringSize, logString);

    {
        // OptixBuiltinISOptions builtin_is_options = {};
        // builtin_is_options.usesMotionBlur      = false;
        // builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
        // OptixModule sphere_module;
        // OPTIX_CHECK(optixBuiltinISModuleGet(optixContext, &module_compile_options, &pipeline_compile_options, &builtin_is_options, &sphere_module));

        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleAH = ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
        // hit_prog_group_desc.hitgroup.moduleCH = ptx_module;
        // hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        hit_prog_group_desc.hitgroup.moduleIS            = ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__is";

		logStringSize = 1024;
		logString[0] = 0;
        OPTIX_CHECK( optixProgramGroupCreate(
            optixContext,
            &hit_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            logString, &logStringSize,
            &hit_prog_group
        ) );
    }

    printf("Hit Program Group %d: %s", logStringSize, logString);

    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
		logStringSize = 1024;
		logString[0] = 0;
        OPTIX_CHECK( optixProgramGroupCreate(
            optixContext, &miss_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            logString, &logStringSize,
            &miss_prog_group
        ) );
    }

    printf("Miss Program Group %d: %s", logStringSize, logString);

    {
        OptixProgramGroupDesc callables_prog_group_desc = {};
        callables_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        callables_prog_group_desc.callables.moduleDC = ptx_module;
        callables_prog_group_desc.callables.entryFunctionNameDC = "__direct_callable__dc";
		logStringSize = 1024;
		logString[0] = 0;
        OPTIX_CHECK( optixProgramGroupCreate(
            optixContext, &callables_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            logString, &logStringSize,
            &callables_prog_group
        ) );
    }

    printf("Callables Group %d: %s", logStringSize, logString);

	// Finally create pipeline
	OptixPipeline pipeline;

    OptixProgramGroup program_groups[] =
    {
        raygen_prog_group,
        miss_prog_group,
        hit_prog_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = 1;

	logStringSize = 1024;
	logString[0] = 0;
    optixPipelineCreate(
        optixContext,
        &pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof( program_groups ) / sizeof( program_groups[0] ),
        logString, &logStringSize,
        &pipeline
    );

    printf("Pipeline %d: ", logStringSize);
    std::cout.write(logString, logStringSize);
    printf("\n");

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK( optixUtilAccumulateStackSizes( raygen_prog_group, &stack_sizes, pipeline ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( hit_prog_group, &stack_sizes, pipeline ) );

    uint32_t max_trace_depth = pipeline_link_options.maxTraceDepth;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ) );

    // This is 4 since the largest depth is IAS->MT->MT->GAS
    const uint32_t max_traversable_graph_depth = 1;

    OPTIX_CHECK( optixPipelineSetStackSize(
        pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversable_graph_depth
    ) );

	// Create acceleration structure
	OptixAccelBuildOptions accelOptions = {};
	OptixBuildInput buildInputs[1];

	CUdeviceptr tempBuffer, outputBuffer;
	size_t tempBufferSizeInBytes, outputBufferSizeInBytes;

	memset( &accelOptions, 0, sizeof( OptixAccelBuildOptions ) );
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
	accelOptions.motionOptions.numKeys = 0;

	memset( &buildInputs[0], 0, sizeof( OptixBuildInput ) );

	// CUdeviceptr d_vertices;
	// CUdeviceptr d_radii;
	// CHECK_CUDA(cudaMalloc((void **)&d_vertices, sizeof(float3) * P), true);
	// CHECK_CUDA(cudaMalloc((void **)&d_radii, sizeof(float) * P), true);
	// CHECK_CUDA(cudaMemcpy((void *)d_vertices, vertices, sizeof(float3) * P, cudaMemcpyHostToDevice), true);
	// CHECK_CUDA(cudaMemcpy((void *)d_radii, radii, sizeof(float) * P, cudaMemcpyHostToDevice), true);

	// float a[12];
	// CHECK_CUDA(cudaMemcpy((void *)a, (void *)d_aabbBuffer, sizeof(float) * 12, cudaMemcpyDeviceToHost), true);
	// for (int i = 0; i < 12; i++) {
	// 	printf("%f \n", a[i]);
	// }

	// CUdeviceptr d_vertices = (CUdeviceptr)vertices;
	// CUdeviceptr d_radii = (CUdeviceptr)radii;
	CUdeviceptr d_aabbs = (CUdeviceptr)d_aabbBuffer;

	// Setup primitives
	buildInputs[0].type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	OptixBuildInputCustomPrimitiveArray& buildInput = buildInputs[0].customPrimitiveArray;
	buildInput.aabbBuffers = &d_aabbs;
	buildInput.numPrimitives = P;
	// buildInput.vertexBuffers = &d_vertices;
	// buildInput.vertexStrideInBytes = 0; // Default stride is sizeof(float3)
	// buildInput.numVertices = P;
	// buildInput.radiusBuffers = &d_radii;
	// buildInput.radiusStrideInBytes = 0; // Default stride is sizeof(float)
	// buildInput.singleRadius = 0;

	unsigned int flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
	buildInput.flags = flags;
	buildInput.numSbtRecords = 1;
	buildInput.sbtIndexOffsetBuffer = 0;
	buildInput.sbtIndexOffsetSizeInBytes = 0;
	buildInput.sbtIndexOffsetStrideInBytes = 0;
	buildInput.primitiveIndexOffset = 0;

	printf("Compute accel memory usage\n");

	OptixAccelBufferSizes bufferSizes = {};
	OPTIX_CHECK( optixAccelComputeMemoryUsage( optixContext, &accelOptions,
	    buildInputs, 1, &bufferSizes ) );

	CUdeviceptr d_output;
	CUdeviceptr d_temp;

	printf("Output size in bytes %d\n", bufferSizes.outputSizeInBytes);
	printf("Temp size in bytes %d\n", bufferSizes.tempSizeInBytes);

	cudaMalloc( (void **)&d_output, bufferSizes.outputSizeInBytes );
	cudaMalloc( (void **)&d_temp, bufferSizes.tempSizeInBytes );

	printf("Building acceleration structure\n");

	OptixTraversableHandle outputHandle = 0;
	OPTIX_CHECK( optixAccelBuild( optixContext, cuStream,
	     &accelOptions, buildInputs, 1, d_temp,
	     bufferSizes.tempSizeInBytes, d_output,
	     bufferSizes.outputSizeInBytes, &outputHandle, nullptr, 0 ) );

    // Setup Shader Binding Table
    OptixShaderBindingTable        sbt = {};

    CUdeviceptr  d_raygen_record;
    const size_t raygen_record_size = sizeof( RayGenRecord );
    CHECK_CUDA(cudaMalloc( reinterpret_cast< void** >( &d_raygen_record ), raygen_record_size ), true);

    RayGenRecord rg_sbt = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast< void* >( d_raygen_record ),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ), true);

    CUdeviceptr  d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof( HitGroupRecord );
    CHECK_CUDA(cudaMalloc( reinterpret_cast< void** >( &d_hitgroup_records ), hitgroup_record_size ), true);

    HitGroupRecord ah_sbt = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( hit_prog_group, &ah_sbt ) );

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast< void* >( d_hitgroup_records ),
        &ah_sbt,
        hitgroup_record_size,
        cudaMemcpyHostToDevice
    ), true);

    CUdeviceptr  d_missgroup_records;
    const size_t missgroup_record_size = sizeof( MissRecord );
    CHECK_CUDA(cudaMalloc( reinterpret_cast< void** >( &d_missgroup_records ), missgroup_record_size ), true);

    MissRecord ms_sbt = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast< void* >( d_missgroup_records ),
        &ms_sbt,
        missgroup_record_size,
        cudaMemcpyHostToDevice
    ), true);

    CUdeviceptr  d_callablegroup_records;
    const size_t callablegroup_record_size = sizeof( CallablesRecord );
    CHECK_CUDA(cudaMalloc( reinterpret_cast< void** >( &d_callablegroup_records ), callablegroup_record_size ), true);

    CallablesRecord ca_sbt = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( callables_prog_group, &ca_sbt ) );

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast< void* >( d_callablegroup_records ),
        &ca_sbt,
        callablegroup_record_size,
        cudaMemcpyHostToDevice
    ), true);

    sbt.raygenRecord = d_raygen_record;
    sbt.hitgroupRecordBase = d_hitgroup_records;
    sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupRecord );
    sbt.hitgroupRecordCount = 1;
    sbt.missRecordBase = d_missgroup_records;
    sbt.missRecordStrideInBytes = sizeof( MissRecord );
    sbt.missRecordCount = 1;
    sbt.callablesRecordBase = d_callablegroup_records;
    sbt.callablesRecordStrideInBytes = sizeof( CallablesRecord );
    sbt.callablesRecordCount = 1;

	printf("Optix BVH done\n");
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;
	printf("Optix BVH time taken %lf seconds\n", duration.count());

    // Now we launch ray tracing!
	CUdeviceptr d_pipelineParams;

	Params p;

	// Set-up params on CPU side
	CHECK_CUDA(cudaMalloc( reinterpret_cast< void** >( &p.gaussians ), W*H*sizeof(int)*1024 ), true);
	CHECK_CUDA(cudaMalloc( reinterpret_cast< void** >( &p.n_gaussians ), W*H*sizeof(int) ), true);
	CHECK_CUDA(cudaMemset((void *)p.n_gaussians, 0, W * H * sizeof(int)), true);

	p.width = W;
	p.height = H;
	p.tanfovx = tanfovx;
	p.tanfovy = tanfovy;
	CHECK_CUDA(cudaMalloc( reinterpret_cast< void** >( &p.viewmatrix_inv ), 16*sizeof(float) ), true);
	CHECK_CUDA(cudaMemcpy( (void *)p.viewmatrix_inv, (void *)viewmatrix_inv, 16*sizeof(float), cudaMemcpyHostToDevice), true);
	p.viewmatrix_inv = viewmatrix_inv;
	CHECK_CUDA(cudaMalloc( reinterpret_cast< void** >( &p.cam_pos ), 3*sizeof(float) ), true);
	CHECK_CUDA(cudaMemcpy( (void *)p.cam_pos, (void *)cam_pos, 3*sizeof(float), cudaMemcpyHostToDevice), true);
	p.handle = outputHandle;
	p.depths = depths;

	CHECK_CUDA(cudaMalloc(reinterpret_cast< void** >( &d_pipelineParams ), sizeof(Params)), true);
	CHECK_CUDA(cudaMemcpy(reinterpret_cast<void*>(d_pipelineParams), &p, sizeof(Params), cudaMemcpyHostToDevice), true);

	printf("Launching!\n");

	start = std::chrono::high_resolution_clock::now();

	OPTIX_CHECK( optixLaunch( pipeline, cuStream, 
    d_pipelineParams, sizeof(Params),
    &sbt, W, H, 1 ) );

	// Wait for stream to be done
	CHECK_CUDA(cudaStreamSynchronize(cuStream), true);
    CHECK_CUDA(cudaStreamDestroy(cuStream), true);
    CHECK_CUDA(cudaDeviceSynchronize(), true);

    end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	printf("Ray tracing took %lf seconds\n", duration.count());

	// Now we sort and alpha-compose
	start = std::chrono::high_resolution_clock::now();
	int threads_per_block = 256;
	int blocks = (W * H + threads_per_block - 1) / threads_per_block;
	composing<CHANNELS> <<<blocks, threads_per_block>>>(W, H, (int *)p.gaussians, (float *)depths, (int *)p.n_gaussians, means2D, bg_color, colors_precomp, conic_opacity, out_color);
	// composing_3D<CHANNELS> <<<blocks, threads_per_block>>>(W, H, d_aabbBuffer, viewmatrix_inv, cam_pos, (int *)p.gaussians, (int *)p.n_gaussians, (float3 *)p.mean3D, (float *)p.cov3D, bg_color, colors_precomp, conic_opacity, out_color);
	CHECK_CUDA(cudaDeviceSynchronize(), true);
	end = std::chrono::high_resolution_clock::now();
	duration = end - start;
	printf("Sort and compose took %lf seconds\n", duration.count());

}

void FORWARD::ray_render(
	const int P,
	const int W,
	const int H,
	// Information needed by ray tracer
	const float znear,
	const float zfar,
	const float* viewmatrix,
	const float* viewmatrix_inv,
	const float* projmatrix,
	const float tanfovx,
	const float tanfovy,
	const glm::vec3* cam_pos,
	float *aabbs,
	// Information used to compute 2D projection color
	float *means3D,
	float2* means2D,
	const float* bg_color,
	float *depths,
	const float* colors_precomp,
	float4* conic_opacity,
	// Output
	float* out_color)
{
	cudaError_t err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1048576ULL*1024);
	build_optix_bvh<NUM_CHANNELS> (W, H, P, (float *)aabbs, (float)tanfovx, (float)tanfovy, (float *)viewmatrix_inv, (float *)cam_pos, (float *)depths, means2D, bg_color, colors_precomp, conic_opacity, out_color);
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}