// Glacier.Inference High-Performance CUDA Kernels
// Common block definitions, data structures, and warp reduction primitives
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define QK_K 256
#define WARP_SIZE 32

// Block layouts matching C# structs
struct __align__(4) BlockQ4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct __align__(2) BlockQ6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};

#define QK8_0 32

struct __align__(2) BlockQ8_0 {
    half d;
    int8_t qs[32];
};

// Unpack scales & mins for Q4_K
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Warp reduction helper using hardware shuffles
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max_128(float val, float* s_warp_mem) {
    val = warp_reduce_max(val);
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0) {
        s_warp_mem[warp_id] = val;
    }
    __syncthreads();
    float block_max = (threadIdx.x < 4) ? s_warp_mem[threadIdx.x] : -1e30f;
    block_max = warp_reduce_max(block_max);
    if (threadIdx.x == 0) {
        s_warp_mem[0] = block_max;
    }
    __syncthreads();
    return s_warp_mem[0];
}

__device__ __forceinline__ float block_reduce_sum_128(float val, float* s_warp_mem) {
    val = warp_reduce_sum(val);
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0) {
        s_warp_mem[warp_id] = val;
    }
    __syncthreads();
    float block_sum = (threadIdx.x < 4) ? s_warp_mem[threadIdx.x] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    if (threadIdx.x == 0) {
        s_warp_mem[0] = block_sum;
    }
    __syncthreads();
    return s_warp_mem[0];
}
