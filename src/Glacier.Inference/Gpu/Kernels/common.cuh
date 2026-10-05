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

struct __align__(4) BlockQ5_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

struct __align__(2) BlockQ3_K {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half d;
};

#define QK8_0 32

struct __align__(2) BlockQ8_0 {
    half d;
    int8_t qs[32];
};

// Unpack scales & mins for Q4_K and Q5_K
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Unpack scale for Q3_K
__device__ __forceinline__ int get_scale_k3(int idx, const uint8_t* scales) {
    uint32_t raw_aux0 = *(const uint32_t*)(scales);
    uint32_t raw_aux1 = *(const uint32_t*)(scales + 4);
    uint32_t raw_aux2 = *(const uint32_t*)(scales + 8);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0F0F0F0F;
    uint32_t sc0 = (raw_aux0 & kmask2) | (((raw_aux2 >> 0) & kmask1) << 4);
    uint32_t sc1 = (raw_aux1 & kmask2) | (((raw_aux2 >> 2) & kmask1) << 4);
    uint32_t sc2 = ((raw_aux0 >> 4) & kmask2) | (((raw_aux2 >> 4) & kmask1) << 4);
    uint32_t sc3 = ((raw_aux1 >> 4) & kmask2) | (((raw_aux2 >> 6) & kmask1) << 4);
    uint32_t b;
    if (idx < 4) b = (sc0 >> (idx * 8)) & 0xFF;
    else if (idx < 8) b = (sc1 >> ((idx - 4) * 8)) & 0xFF;
    else if (idx < 12) b = (sc2 >> ((idx - 8) * 8)) & 0xFF;
    else b = (sc3 >> ((idx - 12) * 8)) & 0xFF;
    return (int)b - 32;
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
