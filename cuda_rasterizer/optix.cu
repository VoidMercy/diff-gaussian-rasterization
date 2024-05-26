#include <cuda_runtime.h>
#include <optix.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })


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
    float*                 depths;
    OptixTraversableHandle handle;
};

struct t_p0123 {
    unsigned int p0, p1, p2, p3, p4, p5, p6, p7;
};

struct Payload {

    t_p0123 p;

    __forceinline__ __device__ void setAll()
    {
        optixSetPayload_0( p.p0 );
        optixSetPayload_1( p.p1 );
        optixSetPayload_2( p.p2 );
        optixSetPayload_3( p.p3 );
        optixSetPayload_4( p.p4 );
        optixSetPayload_5( p.p5 );
        optixSetPayload_6( p.p6 );
        optixSetPayload_7( p.p7 );
    }
    __forceinline__ __device__ void getAll()
    {
        p.p0 = optixGetPayload_0();
        p.p1 = optixGetPayload_1();
        p.p2 = optixGetPayload_2();
        p.p3 = optixGetPayload_3();
        p.p4 = optixGetPayload_4();
        p.p5 = optixGetPayload_5();
        p.p6 = optixGetPayload_6();
        p.p7 = optixGetPayload_7();
    }
};

extern "C" __constant__ Params params;

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    int pixel_x_coord = idx.x;
    int pixel_y_coord = idx.y;
    float top = params.tanfovy;
    float right = params.tanfovx;

    float pixel_view_x = right * (float)2.0 * (((float)(pixel_x_coord) + (float)0.5) / (float)params.width) - right;
    float pixel_view_y = top * (float)2.0 * (((float)(pixel_y_coord) + (float)0.5) / (float)params.height) - top;

    float3 pixel_v = { pixel_view_x, pixel_view_y, 1.0 };
    float3 pixel_w = {
        params.viewmatrix_inv[0] * pixel_v.x + params.viewmatrix_inv[4] * pixel_v.y + params.viewmatrix_inv[8] * pixel_v.z + params.viewmatrix_inv[12],
        params.viewmatrix_inv[1] * pixel_v.x + params.viewmatrix_inv[5] * pixel_v.y + params.viewmatrix_inv[9] * pixel_v.z + params.viewmatrix_inv[13],
        params.viewmatrix_inv[2] * pixel_v.x + params.viewmatrix_inv[6] * pixel_v.y + params.viewmatrix_inv[10] * pixel_v.z + params.viewmatrix_inv[14],
    };

    // Ray
    float3 ray_pos = { params.cam_pos[0], params.cam_pos[1], params.cam_pos[2] };
    float3 ray_dir = { pixel_w.x - ray_pos.x, pixel_w.y - ray_pos.y, pixel_w.z - ray_pos.z };
    float magnitude = sqrt(ray_dir.x * ray_dir.x + ray_dir.y * ray_dir.y + ray_dir.z * ray_dir.z);
    ray_dir = { ray_dir.x / magnitude, ray_dir.y / magnitude, ray_dir.z / magnitude };

    int num = 0;
    int gaussians[1024];
    float t_hit;

    Payload payload;
    payload.p.p0 = idx.x;
    payload.p.p1 = idx.y;
    payload.p.p2 = ((unsigned long)&num) & 0xffffffff;
    payload.p.p3 = (((unsigned long)&num) >> 32) & 0xffffffff;
    payload.p.p4 = ((unsigned long)gaussians) & 0xffffffff;
    payload.p.p5 = (((unsigned long)gaussians) >> 32) & 0xffffffff;
    payload.p.p6 = ((unsigned long)&t_hit) & 0xffffffff;
    payload.p.p7 = (((unsigned long)&t_hit) >> 32) & 0xffffffff;

    int num_before = num;
    float t_min = 0.00f;
    do {
        num_before = num;
        optixTrace(
            params.handle,
            ray_pos,
            ray_dir,
            t_min,
            1e16f,  // tmax
            0.00f,
            OptixVisibilityMask( 255 ),
            // OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
            // OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            OPTIX_RAY_FLAG_NONE,
            0,
            0,
            0,
            payload.p.p0, payload.p.p1, payload.p.p2, payload.p.p3, payload.p.p4, payload.p.p5, payload.p.p6, payload.p.p7
        );

        t_min = t_hit + 0.0001f;
    } while (num_before != num && num < 1024);

    if (num > 1024) {
        num = 1024;
    }
    int pixel_id = pixel_y_coord * params.width + pixel_x_coord;
    ((int *)params.n_gaussians)[pixel_id] = num;

    for (int i = 0; i < num; i++) {
        ((int *)params.gaussians)[i * params.width * params.height + pixel_id] = gaussians[i];
    }
}

extern "C" __global__ void __anyhit__ah() {
    unsigned int x = optixGetPayload_0();
    unsigned int y = optixGetPayload_1();
    int *num = (int *)((unsigned long)optixGetPayload_2() | ((unsigned long)optixGetPayload_3() << 32));
    int *gaussians = (int *)((unsigned long)optixGetPayload_4() | ((unsigned long)optixGetPayload_5() << 32));
    unsigned int primitive_idx = optixGetPrimitiveIndex();

    // int idx = atomicAdd(num, 1);
    // if (idx < 1024) {
    //     // if (x == 0 && y == 0) {
    //     //     printf("Primitive (0, 0) [%d]=%d\n", idx, primitive_idx);
    //     // }
    //     gaussians[idx] = primitive_idx;
    //     optixIgnoreIntersection();
    // } else {
    //     optixTerminateRay();
    // }
    // optixIgnoreIntersection();
}

extern "C" __global__ void __closesthit__ch() {
    unsigned int x = optixGetPayload_0();
    unsigned int y = optixGetPayload_1();
    int *num = (int *)((unsigned long)optixGetPayload_2() | ((unsigned long)optixGetPayload_3() << 32));
    int *gaussians = (int *)((unsigned long)optixGetPayload_4() | ((unsigned long)optixGetPayload_5() << 32));
    float *t_hit = (float *)((unsigned long)optixGetPayload_6() | ((unsigned long)optixGetPayload_7() << 32));
    unsigned int primitive_idx = optixGetPrimitiveIndex();
    float t = optixGetRayTmax();
    float3 ray_pos = optixGetWorldRayOrigin();
    float3 ray_dir = optixGetWorldRayDirection();

    unsigned int t_near = optixGetAttribute_0();
    unsigned int t_far = optixGetAttribute_1();

    int idx = (*num)++;
    t_hit[0] = *((float *)&t_far);
    gaussians[idx] = primitive_idx;
}

extern "C" __global__ void __intersection__is() {
    const float  ray_tmin = optixGetRayTmin();
    const float  ray_tmax = optixGetRayTmax();
    float *data = *((float**)optixGetSbtDataPointer());
    // float *depths = *((float**)optixGetSbtDataPointer() + 1);
    unsigned int primitive_idx = optixGetPrimitiveIndex();
    float3 ray_pos = optixGetWorldRayOrigin();
    float3 ray_dir = optixGetWorldRayDirection();

    // float depth = depths[primitive_idx];
    float *aabb = &data[6 * primitive_idx];

    float3 *bbox_min_ptr = (float3 *)aabb;
    float3 *bbox_max_ptr = (float3 *)&aabb[3];

    float3 bbox_min = *bbox_min_ptr;
    float3 bbox_max = *bbox_max_ptr;
    float3 t_min = make_float3( (bbox_min.x - ray_pos.x) / ray_dir.x, (bbox_min.y - ray_pos.y) / ray_dir.y, (bbox_min.z - ray_pos.z) / ray_dir.z );
    float3 t_max = make_float3( (bbox_max.x - ray_pos.x) / ray_dir.x, (bbox_max.y - ray_pos.y) / ray_dir.y, (bbox_max.z - ray_pos.z) / ray_dir.z );

    float3 t1 = make_float3( min(t_min.x, t_max.x), min(t_min.y, t_max.y), min(t_min.z, t_max.z) );
    float3 t2 = make_float3( max(t_min.x, t_max.x), max(t_min.y, t_max.y), max(t_min.z, t_max.z) );

    float t_near = max(max(t1.x, t1.y), t1.z);
    float t_far = min(min(t2.x, t2.y), t2.z);

    if (t_far > ray_tmin && t_far < ray_tmax) {
        optixReportIntersection(t_far, 0, *((unsigned int *)&t_near), *((unsigned int *)&t_far));
    }
}

extern "C" __global__ void __miss__ms() {
}