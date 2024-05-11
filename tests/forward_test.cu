#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Assuming declaration of ray_render_composing is available
__device__ float3 ray_render_composing(int x, int y, const int N_GAUSSIANS, int *gaussians, float *depths, float2 *mean2D, float *cov2D, const float* colors_precomp, float4* conic_opacity);

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
