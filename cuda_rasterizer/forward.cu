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

#include <tuple>
#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
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
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

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

struct HitInfo {
	 bool hit;
	 float t_near;
	 float t_far;
	 int obj_idx;
};

__device__ struct HitInfo ray_bbox_intersect(
    glm::vec3 ray_pos,
    glm::vec3 ray_dir_inv,
	const struct bvh_aabb* bvh_aabbs,
	int bvh_idx,
    float bound_near,
    float bound_far
) {
    glm::vec3 bbox_min(bvh_aabbs[bvh_idx].x_min, bvh_aabbs[bvh_idx].y_min, bvh_aabbs[bvh_idx].z_min);
    glm::vec3 bbox_max(bvh_aabbs[bvh_idx].x_max, bvh_aabbs[bvh_idx].y_max, bvh_aabbs[bvh_idx].z_max);

	struct HitInfo h;
	h.hit = false;
	h.t_near = FLT_MAX;
	h.obj_idx = bvh_idx;

    glm::vec3 t_min = (bbox_min - ray_pos) * ray_dir_inv;
    glm::vec3 t_max = (bbox_max - ray_pos) * ray_dir_inv;

    glm::vec3 t1 = glm::min(t_min, t_max);
    glm::vec3 t2 = glm::max(t_min, t_max);

    float t_near = fmaxf(fmaxf(t1.x, t1.y), t1.z);
    float t_far = fminf(fminf(t2.x, t2.y), t2.z);
    h.t_near = t_near;
    h.t_far = t_far;

    if (t_far < 0) {
    	return h;
    }
    if (t_near > t_far) {
    	return h;
    }
    if (t_far < bound_near || t_near > bound_far) {
    	return h;
    }
    h.hit = true;
    return h;
}

struct stack_entry {
	int idx;
	float t_near;
	float t_far;
};

template <uint32_t CHANNELS>
__device__ void ray_render_composing (
    // The coordinate of the current point being computed
    int x,
    int y,
    const int H, // height of image
    const int W, // width of image
    // Array of 2D gaussians
    const int N_GAUSSIANS,   	// number of Gaussians
    int *gaussians,          	// index array of Gaussians; should be sorted by depth
    float *depths,            	// depths to sort the Gaussians
    float2 *mean2D,          	// mean which is where it's located in 2D space
    const float *bg_color,      // color of background
    const float* colors_precomp,// computed color 
    float4* conic_opacity,   	// opacity
    float *out_color            // output color
) {
	float accumulated_color[CHANNELS] = { 0.0f };
	float T = 1.0f;  // Start with full transmittance

	// Bounds checking for the arrays
    if (x < 0 || x >= W || y < 0 || y >= H || N_GAUSSIANS <= 0) {
        return;
    }

	// Sort the Gaussians by depth
    thrust::sort_by_key(thrust::seq, depths, depths + N_GAUSSIANS, gaussians);

	// Compute the color of the pixel (cf. "Surface Splatting" by Zwicker et al., 2001)
    for (int i = 0; i < N_GAUSSIANS; i++) {
        int index = gaussians[i];
        float4 con_o = conic_opacity[index];
        float2 gaussian_mean = mean2D[index];

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

    int pix_id = y * W + x;
    for (int ch = 0; ch < CHANNELS; ch++) {
		out_color[ch * H * W + pix_id] = accumulated_color[ch] + bg_color[ch] * T;
    }
}

template <uint32_t CHANNELS>
__global__ void ray_render_cuda(
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
	const int BVH_N,
	const struct bvh_node* bvh_nodes,
	const struct bvh_aabb* bvh_aabbs,
	// Information used to compute 2D projection color
	float2* means2D,
	const float* bg_color,
	float *depths,
	const float* colors_precomp,
	float4* conic_opacity,
	// Output
	float* out_color
) {
	int pixel_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (pixel_id >= W * H) { // Outside number of pixels
		return;
	}
	int pixel_x_coord = pixel_id % W;
	int pixel_y_coord = pixel_id / W;

	// Compute camera frustrum and pixel coordinates in view space
	float top = tanfovy * 1.0;
	float right = tanfovx * 1.0;

	float pixel_view_x = right * 2.0 * ((float)(pixel_x_coord + 0.5) / (float)W) - right;
	float pixel_view_y = top * 2.0 * ((float)(pixel_y_coord + 0.5) / (float)H) - top;
	float3 pixel_v = { pixel_view_x, pixel_view_y, 1.0 };
	float3 pixel_w = transformPoint4x3(pixel_v, viewmatrix_inv);
	glm::vec3 pixel_w_vec(pixel_w.x, pixel_w.y, pixel_w.z);

	// Cast a ray from cam_pos to pixel_w (in world space), and ray trace!
	glm::vec3 ray_pos = *cam_pos;
	glm::vec3 ray_dir = glm::normalize(pixel_w_vec - ray_pos);
	glm::vec3 ray_dir_inv = glm::vec3(1.0f) / ray_dir;

	// printf("Camera position %f %f %f\n", ray_pos.x, ray_pos.y, ray_pos.z);
	// printf("Pixel coord %f %f %f\n", pixel_w_vec.x, pixel_w_vec.y, pixel_w_vec.z);

	const int BVH_STACK_SIZE = 1024;
	const int MAX_GAUSSIANS = 1024;

	// Gaussians
	int gaussians[MAX_GAUSSIANS];
	int gaussian_idx = 0;

	// Data structures for storing intersections
	struct stack_entry stack[BVH_STACK_SIZE];
	int stack_pointer = 0;
	int intersection_idx;
	float intersection_t_near;
	float intersection_t_far;

	float min_t_near = -FLT_MAX;

	intersection_idx = -1;
	intersection_t_near = FLT_MAX;
	intersection_t_far = -FLT_MAX;
	stack[stack_pointer++] = { .idx=0, .t_near=0.0, .t_far=0.0 };
	while (stack_pointer > 0) {
		if (stack_pointer >= BVH_STACK_SIZE) {
			printf("WTF STACK OVERFLOW\n");
		}
		struct stack_entry cur = stack[--stack_pointer];
		int node_addr = cur.idx;
		float node_t_near = cur.t_near;
		float node_t_far = cur.t_far;

		// if (node_t_near >= intersection_t_near) { // Found a closer intersection already, so skip this node
		// 	continue;
		// }

		if (bvh_nodes[node_addr].object_idx != -1) { // Is a leaf node
			// printf("Found leaf. Pixel %d %d. Node %d AABB is (%f, %f, %f) (%f, %f, %f), for object %d at times (%f, %f)\n", pixel_x_coord, pixel_y_coord, node_addr, bvh_aabbs[node_addr].x_min, bvh_aabbs[node_addr].y_min, bvh_aabbs[node_addr].z_min, bvh_aabbs[node_addr].x_max, bvh_aabbs[node_addr].y_max, bvh_aabbs[node_addr].z_max, bvh_nodes[node_addr].object_idx, node_t_near, node_t_far);
			gaussians[gaussian_idx++] = bvh_nodes[node_addr].object_idx;
			if (gaussian_idx == MAX_GAUSSIANS) {
				break;
			}
			if (node_t_near < intersection_t_near) {
				intersection_t_near = node_t_near;
				intersection_t_far = node_t_far;
				intersection_idx = bvh_nodes[node_addr].object_idx;
			}
		} else {
			int left_idx = bvh_nodes[node_addr].left_idx;
			int right_idx = bvh_nodes[node_addr].right_idx;
			HitInfo h1 = ray_bbox_intersect(ray_pos, ray_dir_inv, bvh_aabbs, left_idx, 0.0, FLT_MAX);
			HitInfo h2 = ray_bbox_intersect(ray_pos, ray_dir_inv, bvh_aabbs, right_idx, 0.0, FLT_MAX);
			// int first_idx = (h1.t_near < h2.t_near) ? left_idx : right_idx;
			// int second_idx = (h1.t_near < h2.t_near) ? right_idx : left_idx;
			// bool hit_first = (h1.t_near < h2.t_near) ? h1.hit : h2.hit;
			// bool hit_second = (h1.t_near < h2.t_near) ? h2.hit : h1.hit;
			// float first_t_near = (h1.t_near < h2.t_near) ? h1.t_near : h2.t_near;
			// float second_t_near = (h1.t_near < h2.t_near) ? h2.t_near : h1.t_near;
			// float first_t_far = (h1.t_near < h2.t_near) ? h1.t_far : h2.t_far;
			// float second_t_far = (h1.t_near < h2.t_near) ? h2.t_far : h1.t_far;

			// if (hit_second) {
			// 	stack[stack_pointer++] = { .idx=second_idx, .t_near=second_t_near, .t_far=second_t_far };
			// }
			// if (hit_first) {
			// 	stack[stack_pointer++] = { .idx=first_idx, .t_near=first_t_near, .t_far=first_t_far };
			// }
			if (h1.hit) {
				stack[stack_pointer++] = { .idx=left_idx, .t_near=h1.t_near, .t_far=h1.t_far };
			}
			if (h2.hit) {
				stack[stack_pointer++] = { .idx=right_idx, .t_near=h2.t_near, .t_far=h2.t_far };
			}
		}
	}

	if (gaussian_idx > 0) {
		ray_render_composing<CHANNELS>(
			pixel_x_coord,
			pixel_y_coord,
			H,
			W,
			gaussian_idx,
			gaussians,
			depths,
			means2D,
			bg_color,
			colors_precomp,
			conic_opacity,
			out_color
		);
		printf("Total intersections for pixel (%d, %d): %d. Color: (%f, %f, %f)\n", pixel_x_coord, pixel_y_coord, gaussian_idx, out_color[pixel_id], out_color[H * W + pixel_id], out_color[2 * H * W + pixel_id]);
	}
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
	const int BVH_N,
	const struct bvh_node* bvh_nodes,
	const struct bvh_aabb* bvh_aabbs,
	// Information used to compute 2D projection color
	float2* means2D,
	const float* bg_color,
	float *depths,
	const float* colors_precomp,
	float4* conic_opacity,
	// Output
	float* out_color)
{
	printf("Ray render %d x %d for %d channels\n", W, H, NUM_CHANNELS);
	int threads_per_block = 256;
	int num_blocks = (W * H + threads_per_block - 1) / threads_per_block;
	ray_render_cuda<NUM_CHANNELS> <<<num_blocks, threads_per_block>>> (
		P, W, H,
		znear, zfar,
		viewmatrix,
		viewmatrix_inv,
		projmatrix,
		tanfovx,
		tanfovy,
		cam_pos,
		BVH_N,
		bvh_nodes,
		bvh_aabbs,
		means2D,
		bg_color,
		depths,
		colors_precomp,
		conic_opacity,
		out_color
	);
	printf("Ray render done\n");
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