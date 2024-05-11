#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__device__ float3 ray_render_composing(
    // The coordinate of the current point being computed
    int x,
    int y,
    // Array of 2D gaussians
    const int N_GAUSSIANS,   	// the number of Gaussians
    int *gaussians,          	// the index array of Gaussians; should be sorted by depth
    float *depths,            	// depths to sort the Gaussians
    float2 *mean2D,          	// the mean which is where it's located in 2D space
    float *cov2D,            	// the covariances of the Gaussian
    const float* colors_precomp,// the computed color 
    float4* conic_opacity    	// the opacity
) {
    float3 accumulated_color = {0.0f, 0.0f, 0.0f};
    float accumulated_alpha = 0.0f;

	// Sort the Gaussians by depth
	// quickSort(gaussians, depths, 0, N_GAUSSIANS - 1);

    for (int i = 0; i < N_GAUSSIANS; i++) {
        int index = gaussians[i];
        float3 gaussian_color = {
            colors_precomp[index * 3],     // R
            colors_precomp[index * 3 + 1], // G
            colors_precomp[index * 3 + 2]  // B
        };
        float4 con_o = conic_opacity[index];
        float2 gaussian_mean = mean2D[index];

        // Compute power using conic matrix (cf. "Surface Splatting" by Zwicker et al., 2001)
        float2 d = {x - gaussian_mean.x, y - gaussian_mean.y};
        float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
        if (power > 0.0f)
            continue;

        // Compute alpha using the exponential of power and the Gaussian opacity
        float alpha = min(exp(power) * con_o.w, 0.99f);
        if (alpha < 1.0f / 255.0f)
			continue;	// Skip if alpha is too small
		if (1 - alpha < 0.0001f) {
			break; 		// Break if alpha value saturates
		}

        // Alpha compositing from front to back
        accumulated_color.x = gaussian_color.x * alpha + accumulated_color.x * (1.0f - alpha);
        accumulated_color.y = gaussian_color.y * alpha + accumulated_color.y * (1.0f - alpha);
        accumulated_color.z = gaussian_color.z * alpha + accumulated_color.z * (1.0f - alpha);
        accumulated_alpha += alpha * (1.0f - accumulated_alpha);
    }

    return accumulated_color;
}

// Random state setup kernel
__global__ void setup_kernel(curandState *state, int seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

// Test kernel for ray_render_composing
__global__ void testRayRenderComposing(curandState *states, int *gaussians, float *depths, float2 *mean2D, float *cov2D, float *colors_precomp, float4 *conic_opacity, int N_GAUSSIANS) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N_GAUSSIANS) {
        // Random values setup
        depths[idx] = curand_uniform(&states[idx]) * 100.0f;  // Random depths between 0 and 100
        mean2D[idx] = make_float2(curand_uniform(&states[idx]) * 800, curand_uniform(&states[idx]) * 600);  // Random positions within 800x600 window
        cov2D[idx] = curand_uniform(&states[idx]) * 10.0f + 1.0f;  // Random covariances between 1 and 11
        int colorIndex = idx * 3;
        colors_precomp[colorIndex] = curand_uniform(&states[idx]);  // Random color R
        colors_precomp[colorIndex + 1] = curand_uniform(&states[idx]);  // Random color G
        colors_precomp[colorIndex + 2] = curand_uniform(&states[idx]);  // Random color B
        conic_opacity[idx] = make_float4(1.0f, 0.0f, 1.0f, curand_uniform(&states[idx]));  // Random opacity
    }

    // Ensure all data is generated before testing
    __syncthreads();

    // Run the ray rendering composing test on a specific pixel
    if (idx == 0) {
        int testX = 400, testY = 300;
        float3 result = ray_render_composing(testX, testY, N_GAUSSIANS, gaussians, depths, mean2D, cov2D, colors_precomp, conic_opacity);
        printf("Accumulated color at (%d, %d): R=%f, G=%f, B=%f\n", testX, testY, result.x, result.y, result.z);
    }
}

int main() {
    const int N_GAUSSIANS = 1000;
    int *gaussians;
    float *depths;
    float2 *mean2D;
    float *cov2D;
    float *colors_precomp;
    float4 *conic_opacity;
    curandState *states;

    printf("Test code running");

    // Allocate memory for arrays and states
    cudaMalloc(&gaussians, N_GAUSSIANS * sizeof(int));
    cudaMalloc(&depths, N_GAUSSIANS * sizeof(float));
    cudaMalloc(&mean2D, N_GAUSSIANS * sizeof(float2));
    cudaMalloc(&cov2D, N_GAUSSIANS * sizeof(float));
    cudaMalloc(&colors_precomp, N_GAUSSIANS * 3 * sizeof(float));
    cudaMalloc(&conic_opacity, N_GAUSSIANS * sizeof(float4));
    cudaMalloc(&states, N_GAUSSIANS * sizeof(curandState));

    // Initialize indices
    for (int i = 0; i < N_GAUSSIANS; i++) {
        gaussians[i] = i;
    }

    // Setup random states
    setup_kernel<<<(N_GAUSSIANS + 255) / 256, 256>>>(states, time(NULL));
    cudaDeviceSynchronize();

    // Run test
    testRayRenderComposing<<<(N_GAUSSIANS + 255) / 256, 256>>>(states, gaussians, depths, mean2D, cov2D, colors_precomp, conic_opacity, N_GAUSSIANS);
    cudaDeviceSynchronize();

    // Clean up
    cudaFree(gaussians);
    cudaFree(depths);
    cudaFree(mean2D);
    cudaFree(cov2D);
    cudaFree(colors_precomp);
    cudaFree(conic_opacity);
    cudaFree(states);

    return 0;
}
