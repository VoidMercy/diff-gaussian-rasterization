#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

__device__ float3 ray_render_composing(int x, int y, const int N_GAUSSIANS, int *gaussians, float *depths, float2 *mean2D, float *cov2D, const float* colors_precomp, float4* conic_opacity);

// Kernel to run test cases
__global__ void testRayRenderComposing() {
    const int N_GAUSSIANS = 1000;  // Number of Gaussians
    int *gaussians;   // Indices of the gaussians
    float *depths;    // Array to hold depth values
    float2 *mean2D;   // Mean positions of Gaussians
    float *cov2D;     // Covariances of Gaussians
    float *colors_precomp; // Precomputed colors
    float4 *conic_opacity; // Opacity values

    // Allocate memory for each array
    cudaMallocManaged(&gaussians, N_GAUSSIANS * sizeof(int));
    cudaMallocManaged(&depths, N_GAUSSIANS * sizeof(float));
    cudaMallocManaged(&mean2D, N_GAUSSIANS * sizeof(float2));
    cudaMallocManaged(&cov2D, N_GAUSSIANS * sizeof(float));
    cudaMallocManaged(&colors_precomp, N_GAUSSIANS * 3 * sizeof(float));
    cudaMallocManaged(&conic_opacity, N_GAUSSIANS * sizeof(float4));

    // Initialize data
    for (int i = 0; i < N_GAUSSIANS; i++) {
        gaussians[i] = i;
        depths[i] = rand() / (float)RAND_MAX * 100.0f;  // Random depths
        mean2D[i] = make_float2(rand() / (float)RAND_MAX * 800, rand() / (float)RAND_MAX * 600);  // Random positions within a hypothetical 800x600 window
        cov2D[i] = rand() / (float)RAND_MAX * 10.0f + 1.0f;  // Random positive covariance
        colors_precomp[3 * i] = rand() / (float)RAND_MAX;  // Random color R
        colors_precomp[3 * i + 1] = rand() / (float)RAND_MAX;  // Random color G
        colors_precomp[3 * i + 2] = rand() / (float)RAND_MAX;  // Random color B
        conic_opacity[i] = make_float4(1.0f, 0.0f, 1.0f, rand() / (float)RAND_MAX);  // Random opacity
    }

    // Test the rendering function at a particular pixel
    int testX = 400, testY = 300;
    float3 result = ray_render_composing(testX, testY, N_GAUSSIANS, gaussians, depths, mean2D, cov2D, colors_precomp, conic_opacity);

    // Print the result
    printf("Accumulated color at (%d, %d): R=%f, G=%f, B=%f\n", testX, testY, result.x, result.y, result.z);

    // Clean up
    cudaFree(gaussians);
    cudaFree(depths);
    cudaFree(mean2D);
    cudaFree(cov2D);
    cudaFree(colors_precomp);
    cudaFree(conic_opacity);
}

int main() {
    // Launch the kernel
    testRayRenderComposing<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}
