// Glacier.Inference NariQ Trellis-Coded Quantization CUDA Kernels (GGML_TYPE_TRL1..TRL6)
// High-throughput bit-exact FP32 16x16 tile decode (0x83DCD12Du unsigned IDP4A.U8 with zero >>2 shift,
// PRMT/shift-sharing bit extraction for TRL2/TRL4, zero-u2 TRL3, dynamic-SMEM warp-shuffle FWHT,
// fused SwiGLU+Pre-RHT, and 4-warp/8-warp contiguous Split-K!)
#pragma once

#include "common.cuh"

extern "C" {

// =========================================================================
// Shared Warp-Shuffle FWHT helper for dynamic SMEM (zero static 16 KB bloat!)
// Runs stages h = 1..16 in registers via __shfl_xor_sync, h = 32..block_len/4 in smem,
// and fuses the final h = block_len/2 butterfly with scale + s16 multiply!
// =========================================================================
__device__ __forceinline__ void trellis_pre_rht_finish(
    float* __restrict__ smem,
    const uint16_t* __restrict__ s16_fp16,
    float* __restrict__ x_rot,
    int b0,
    int block_len
) {
    int tid = threadIdx.x;
    if (block_len == 128) {
        __syncthreads();
        if (tid < 64) {
            int idx0 = ((tid & ~31) << 1) | (tid & 31);
            int idx1 = idx0 | 32;
            float a = smem[idx0], b = smem[idx1];
            smem[idx0] = a + b; smem[idx1] = a - b;
        }
        __syncthreads();
        if (tid < 64) {
            int idx0 = tid;
            int idx1 = tid + 64;
            float a = smem[idx0], b = smem[idx1];
            float c = 0.08838834764831845f; // 1 / sqrt(128)
            float s0 = __half2float(*reinterpret_cast<const __half*>(&s16_fp16[(b0 + idx0) >> 4]));
            float s1 = __half2float(*reinterpret_cast<const __half*>(&s16_fp16[(b0 + idx1) >> 4]));
            x_rot[b0 + idx0] = ((a + b) * c) * s0;
            x_rot[b0 + idx1] = ((a - b) * c) * s1;
        }
        return;
    }

    __syncthreads();
    int half = block_len >> 1;
    int start_h = (block_len >= 32) ? 32 : 1;
    for (int h = start_h; h < half; h <<= 1) {
        int mask = h - 1;
        for (int p = tid; p < half; p += blockDim.x) {
            int idx0 = ((p & ~mask) << 1) | (p & mask);
            int idx1 = idx0 | h;
            float a = smem[idx0];
            float b = smem[idx1];
            smem[idx0] = a + b;
            smem[idx1] = a - b;
        }
        __syncthreads();
    }

    float c = rsqrtf((float)block_len);
    if (block_len >= 64 || block_len < 32) {
        // Fuse final butterfly h = half with scale + s16 store (saves 1 __syncthreads() + smem write/read)
        for (int p = tid; p < half; p += blockDim.x) {
            float a = smem[p];
            float b = smem[p + half];
            int g0 = b0 + p;
            int g1 = b0 + p + half;
            float s0 = __half2float(*reinterpret_cast<const __half*>(&s16_fp16[g0 >> 4]));
            float s1 = __half2float(*reinterpret_cast<const __half*>(&s16_fp16[g1 >> 4]));
            x_rot[g0] = ((a + b) * c) * s0;
            x_rot[g1] = ((a - b) * c) * s1;
        }
    } else {
        // block_len == 32: all 5 stages already done in warp shuffle
        for (int i = tid; i < block_len; i += blockDim.x) {
            int gi = b0 + i;
            float s = __half2float(*reinterpret_cast<const __half*>(&s16_fp16[gi >> 4]));
            x_rot[gi] = (smem[i] * c) * s;
        }
    }
}

// =========================================================================
// 1. Pre-transform with FUSED s16 scaling (dynamic SMEM = block_len * 4 B):
//    x_rot[gi] = T_cols(pre * gather_pin(x))[gi] * s16[gi >> 4]
// =========================================================================
__global__ void trellis_pre_rht(
    const float* __restrict__ x,
    const float* __restrict__ pre,
    const int* __restrict__ pin,
    const uint16_t* __restrict__ s16_fp16,
    float* __restrict__ x_rot,
    int K,
    int kind,
    int block_len,
    int has_pin
) {
    if (kind == 0) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < K) {
            int src = has_pin ? pin[idx] : idx;
            float s = __half2float(*reinterpret_cast<const __half*>(&s16_fp16[idx >> 4]));
            x_rot[idx] = (x[src] * pre[idx]) * s;
        }
        return;
    }

    extern __shared__ float smem[];
    int b0 = blockIdx.x * block_len;

    if (block_len >= 32) {
        int warp = threadIdx.x >> 5;
        int lane = threadIdx.x & 31;
        int nwarps = blockDim.x >> 5;
        for (int base = warp * 32; base < block_len; base += nwarps * 32) {
            int i = base + lane;
            int gi = b0 + i;
            int src = has_pin ? pin[gi] : gi;
            float v = x[src] * pre[gi];
            #pragma unroll
            for (int h = 1; h < 32; h <<= 1) {
                float other = __shfl_xor_sync(0xFFFFFFFFu, v, h);
                v = ((lane & h) == 0) ? (v + other) : (other - v);
            }
            smem[i] = v;
        }
    } else {
        for (int i = threadIdx.x; i < block_len; i += blockDim.x) {
            int gi = b0 + i;
            int src = has_pin ? pin[gi] : gi;
            smem[i] = x[src] * pre[gi];
        }
    }

    trellis_pre_rht_finish(smem, s16_fp16, x_rot, b0, block_len);
}

// =========================================================================
// 1b. Fused SwiGLU + Pre-RHT + s16 scaling (eliminates 112 q35_swiglu kernels/token!):
//     x_rot[gi] = T_cols(pre * gather_pin(silu(gate) * up))[gi] * s16[gi >> 4]
// =========================================================================
__global__ void trellis_pre_rht_swiglu(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ pre,
    const int* __restrict__ pin,
    const uint16_t* __restrict__ s16_fp16,
    float* __restrict__ x_rot,
    int K,
    int kind,
    int block_len,
    int has_pin
) {
    if (kind == 0) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < K) {
            int src = has_pin ? pin[idx] : idx;
            float g = gate[src];
            float act = (g / (1.0f + __expf(-g))) * up[src];
            float s = __half2float(*reinterpret_cast<const __half*>(&s16_fp16[idx >> 4]));
            x_rot[idx] = (act * pre[idx]) * s;
        }
        return;
    }

    extern __shared__ float smem[];
    int b0 = blockIdx.x * block_len;

    if (block_len >= 32) {
        int warp = threadIdx.x >> 5;
        int lane = threadIdx.x & 31;
        int nwarps = blockDim.x >> 5;
        for (int base = warp * 32; base < block_len; base += nwarps * 32) {
            int i = base + lane;
            int gi = b0 + i;
            int src = has_pin ? pin[gi] : gi;
            float g = gate[src];
            float v = ((g / (1.0f + __expf(-g))) * up[src]) * pre[gi];
            #pragma unroll
            for (int h = 1; h < 32; h <<= 1) {
                float other = __shfl_xor_sync(0xFFFFFFFFu, v, h);
                v = ((lane & h) == 0) ? (v + other) : (other - v);
            }
            smem[i] = v;
        }
    } else {
        for (int i = threadIdx.x; i < block_len; i += blockDim.x) {
            int gi = b0 + i;
            int src = has_pin ? pin[gi] : gi;
            float g = gate[src];
            smem[i] = ((g / (1.0f + __expf(-g))) * up[src]) * pre[gi];
        }
    }

    trellis_pre_rht_finish(smem, s16_fp16, x_rot, b0, block_len);
}

// Bit-exact 4x scaled decode via unsigned IDP.4A.U8.U8:
// Masking & 0xFCFCFCFCu yields 4 * (byte >> 2) in every byte (0..252), eliminating >> 2 (SHF.R)
// on every single weight across the entire model! Final acc is multiplied by 0.25f once after K-loop.
__device__ __forceinline__ float trl_decode_4x_f32(uint32_t state) {
    uint32_t xv = state * 0x83DCD12Du;
    uint32_t u4 = __dp4a(xv & 0xFCFCFCFCu, 0x01010101u, 0x4B000000u);
    return __uint_as_float(u4) - 8389112.0f; // 4.0f * (u - 126.0f), 100% bit-exact!
}

// Contiguous K-slice [i_begin, i_end) accumulator in full FP32 (returns 4x-scaled acc0, acc1)
__device__ __forceinline__ void trl_gemv_warp_range(
    const uint32_t* __restrict__ code_words,
    const float* __restrict__ x_rot,
    int J,
    int kt,
    int bits,
    int lane,
    int i_begin,
    int i_end,
    float& acc0,
    float& acc1
) {
    int cb = (lane & 3) << 1;
    size_t tile_row_base = (size_t)J * kt;

    if (bits == 2) {
        int start0 = (16 * lane + 498) & 511;
        int w0 = start0 >> 5;
        int w1 = (w0 + 1) & 15;
        bool odd_lane = (lane & 1) != 0;

        #pragma unroll 4
        for (int i = i_begin; i < i_end; i++) {
            const uint32_t* t = code_words + ((tile_row_base + i) << 4);
            uint32_t u0 = __ldg(&t[w0]);
            uint32_t h0 = odd_lane ? u0 : __byte_perm(__ldg(&t[w1]), u0, 0x5432);

            int xb = (i << 4) + cb;
            float2 x01 = __ldg(reinterpret_cast<const float2*>(&x_rot[xb]));
            float2 x89 = __ldg(reinterpret_cast<const float2*>(&x_rot[xb + 8]));

            // Share 3 left shifts (h2, h4, h6) across pairs (s0,s4), (s1,s5), (s2,s6) via PRMT/SHR;
            // (s3, s7) come directly from h0 with zero shifts! (11 ops instead of 16 ops per tile!)
            uint32_t h2 = h0 << 2;
            acc0 = fmaf(trl_decode_4x_f32(h2 >> 16),                     x01.x, acc0);
            acc1 = fmaf(trl_decode_4x_f32(__byte_perm(h2, 0u, 0x4421)),  x01.x, acc1);

            uint32_t h4 = h0 << 4;
            acc0 = fmaf(trl_decode_4x_f32(h4 >> 16),                     x01.y, acc0);
            acc1 = fmaf(trl_decode_4x_f32(__byte_perm(h4, 0u, 0x4421)),  x01.y, acc1);

            uint32_t h6 = h0 << 6;
            acc0 = fmaf(trl_decode_4x_f32(h6 >> 16),                     x89.x, acc0);
            acc1 = fmaf(trl_decode_4x_f32(__byte_perm(h6, 0u, 0x4421)),  x89.x, acc1);

            acc0 = fmaf(trl_decode_4x_f32(__byte_perm(h0, 0u, 0x4421)),  x89.y, acc0);
            acc1 = fmaf(trl_decode_4x_f32(h0 & 0xFFFFu),                 x89.y, acc1);
        }
    } else if (bits == 3) {
        // For bits == 3, o0 = ((24*lane + 755) & 31) is always in {3, 11, 19, 27} <= 27,
        // so o0 + 21 + 16 <= 64: all 8 states always fit in u0 and u1 (ZERO u2 load!).
        int start0 = (24 * lane + 755) % 768;
        int w0 = start0 >> 5;
        int w1 = (w0 + 1 == 24) ? 0 : (w0 + 1);
        int o0 = start0 & 31;

        #pragma unroll 4
        for (int i = i_begin; i < i_end; i++) {
            const uint32_t* t = code_words + (tile_row_base + i) * 24;
            uint32_t u0 = __ldg(&t[w0]);
            uint32_t u1 = __ldg(&t[w1]);
            uint32_t hi  = __funnelshift_l(u1, u0, o0);
            uint32_t mid = u1 << o0;

            int xb = (i << 4) + cb;
            float2 x01 = __ldg(reinterpret_cast<const float2*>(&x_rot[xb]));
            float2 x89 = __ldg(reinterpret_cast<const float2*>(&x_rot[xb + 8]));

            acc0 = fmaf(trl_decode_4x_f32(hi >> 16),                              x01.x, acc0);
            acc0 = fmaf(trl_decode_4x_f32((hi >> 13) & 0xFFFFu),                  x01.y, acc0);
            acc0 = fmaf(trl_decode_4x_f32((hi >> 10) & 0xFFFFu),                  x89.x, acc0);
            acc0 = fmaf(trl_decode_4x_f32((hi >> 7)  & 0xFFFFu),                  x89.y, acc0);
            acc1 = fmaf(trl_decode_4x_f32((hi >> 4)  & 0xFFFFu),                  x01.x, acc1);
            acc1 = fmaf(trl_decode_4x_f32((hi >> 1)  & 0xFFFFu),                  x01.y, acc1);
            acc1 = fmaf(trl_decode_4x_f32(__funnelshift_l(mid, hi, 18) >> 16),    x89.x, acc1);
            acc1 = fmaf(trl_decode_4x_f32(__funnelshift_l(mid, hi, 21) >> 16),    x89.y, acc1);
        }
    } else if (bits == 4) {
        // For bits == 4, start0 = (32*lane + 1012) & 1023 => o0 == 20 ALWAYS!
        // o0 + 28 + 16 == 64 => zero u2, zero 64-bit shifts, s3/s5/s7 byte-aligned in u1!
        int w0 = (lane + 31) & 31;
        int w1 = lane;

        #pragma unroll 4
        for (int i = i_begin; i < i_end; i++) {
            const uint32_t* t = code_words + ((tile_row_base + i) << 5);
            uint32_t u0 = __ldg(&t[w0]);
            uint32_t u1 = __ldg(&t[w1]);
            uint32_t hi = __funnelshift_l(u1, u0, 20);
            uint32_t u1_4 = u1 << 4;

            int xb = (i << 4) + cb;
            float2 x01 = __ldg(reinterpret_cast<const float2*>(&x_rot[xb]));
            float2 x89 = __ldg(reinterpret_cast<const float2*>(&x_rot[xb + 8]));

            acc0 = fmaf(trl_decode_4x_f32(hi >> 16),                        x01.x, acc0);
            acc1 = fmaf(trl_decode_4x_f32(u1_4 >> 16),                      x01.x, acc1);

            acc0 = fmaf(trl_decode_4x_f32((hi >> 12) & 0xFFFFu),            x01.y, acc0);
            acc1 = fmaf(trl_decode_4x_f32(__byte_perm(u1, 0u, 0x4421)),     x01.y, acc1);

            acc0 = fmaf(trl_decode_4x_f32(__byte_perm(hi, 0u, 0x4421)),     x89.x, acc0);
            acc1 = fmaf(trl_decode_4x_f32(__byte_perm(u1_4, 0u, 0x4421)),   x89.x, acc1);

            acc0 = fmaf(trl_decode_4x_f32(u1 >> 16),                        x89.y, acc0);
            acc1 = fmaf(trl_decode_4x_f32(u1 & 0xFFFFu),                    x89.y, acc1);
        }
    } else {
        int nw = 8 * bits;
        int total = 256 * bits;
        int start0 = ((8 * lane + 1) * bits - 16 + total) % total;
        int w0 = start0 >> 5;
        int w1 = (w0 + 1) % nw;
        int w2 = (w0 + 2) % nw;
        int o0 = start0 & 31;
        bool need_u2 = (o0 + 7 * bits + 16) > 64;

        for (int i = i_begin; i < i_end; i++) {
            const uint32_t* t = code_words + (tile_row_base + i) * nw;
            uint32_t u0 = __ldg(&t[w0]);
            uint32_t u1 = __ldg(&t[w1]);
            uint32_t u2 = need_u2 ? __ldg(&t[w2]) : 0u;
            uint32_t hi  = __funnelshift_l(u1, u0, o0);
            uint32_t mid = __funnelshift_l(u2, u1, o0);
            uint64_t win = ((uint64_t)hi << 32) | (uint64_t)mid;

            int xb = (i << 4) + cb;
            float2 x01 = __ldg(reinterpret_cast<const float2*>(&x_rot[xb]));
            float2 x89 = __ldg(reinterpret_cast<const float2*>(&x_rot[xb + 8]));

            acc0 = fmaf(trl_decode_4x_f32((uint32_t)(win >> 48)), x01.x, acc0); win <<= bits;
            acc0 = fmaf(trl_decode_4x_f32((uint32_t)(win >> 48)), x01.y, acc0); win <<= bits;
            acc0 = fmaf(trl_decode_4x_f32((uint32_t)(win >> 48)), x89.x, acc0); win <<= bits;
            acc0 = fmaf(trl_decode_4x_f32((uint32_t)(win >> 48)), x89.y, acc0); win <<= bits;

            acc1 = fmaf(trl_decode_4x_f32((uint32_t)(win >> 48)), x01.x, acc1); win <<= bits;
            acc1 = fmaf(trl_decode_4x_f32((uint32_t)(win >> 48)), x01.y, acc1); win <<= bits;
            acc1 = fmaf(trl_decode_4x_f32((uint32_t)(win >> 48)), x89.x, acc1); win <<= bits;
            acc1 = fmaf(trl_decode_4x_f32((uint32_t)(win >> 48)), x89.y, acc1);
        }
    }
}

__global__ void trellis_gemv(
    const uint32_t* __restrict__ code_words,
    const float* __restrict__ x_rot,
    float* __restrict__ y_rot,
    int K,
    int N,
    int bits
) {
    int tid = threadIdx.x; // 0..127
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int J = blockIdx.x * 4 + warp_id;
    if (J * 16 >= N) return;

    float acc0 = 0.0f, acc1 = 0.0f;
    trl_gemv_warp_range(code_words, x_rot, J, K >> 4, bits, lane, 0, K >> 4, acc0, acc1);

    acc0 += __shfl_xor_sync(0xFFFFFFFFu, acc0, 1);
    acc0 += __shfl_xor_sync(0xFFFFFFFFu, acc0, 2);
    acc1 += __shfl_xor_sync(0xFFFFFFFFu, acc1, 1);
    acc1 += __shfl_xor_sync(0xFFFFFFFFu, acc1, 2);

    if ((lane & 3) == 0) {
        int r = lane >> 2;
        y_rot[(J << 4) + r]     = acc0 * 0.25f;
        y_rot[(J << 4) + r + 8] = acc1 * 0.25f;
    }
}

__global__ void trellis_gemv_splitk(
    const uint32_t* __restrict__ code_words,
    const float* __restrict__ x_rot,
    float* __restrict__ y_rot,
    int K,
    int N,
    int bits
) {
    int J = blockIdx.x;
    if (J * 16 >= N) return;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int kt = K >> 4;
    int chunk = (kt + 3) >> 2;
    int i_begin = warp_id * chunk;
    int i_end = min(kt, i_begin + chunk);

    float acc0 = 0.0f, acc1 = 0.0f;
    if (i_begin < i_end) {
        trl_gemv_warp_range(code_words, x_rot, J, kt, bits, lane, i_begin, i_end, acc0, acc1);
    }

    acc0 += __shfl_xor_sync(0xFFFFFFFFu, acc0, 1);
    acc0 += __shfl_xor_sync(0xFFFFFFFFu, acc0, 2);
    acc1 += __shfl_xor_sync(0xFFFFFFFFu, acc1, 1);
    acc1 += __shfl_xor_sync(0xFFFFFFFFu, acc1, 2);

    __shared__ float s_part[4][16];
    if ((lane & 3) == 0) {
        int r = lane >> 2;
        s_part[warp_id][r] = acc0;
        s_part[warp_id][r + 8] = acc1;
    }
    __syncthreads();

    if (tid < 16) {
        y_rot[(J << 4) + tid] = (s_part[0][tid] + s_part[1][tid] + s_part[2][tid] + s_part[3][tid]) * 0.25f;
    }
}

// 8-warp Split-K (256 threads per 16-row tile): doubles SM saturation for small-N matrices (N <= 2048, e.g. KW/VW)!
__global__ void trellis_gemv_splitk8(
    const uint32_t* __restrict__ code_words,
    const float* __restrict__ x_rot,
    float* __restrict__ y_rot,
    int K,
    int N,
    int bits
) {
    int J = blockIdx.x;
    if (J * 16 >= N) return;

    int tid = threadIdx.x; // 0..255
    int warp_id = tid >> 5; // 0..7
    int lane = tid & 31;
    int kt = K >> 4;
    int chunk = (kt + 7) >> 3;
    int i_begin = warp_id * chunk;
    int i_end = min(kt, i_begin + chunk);

    float acc0 = 0.0f, acc1 = 0.0f;
    if (i_begin < i_end) {
        trl_gemv_warp_range(code_words, x_rot, J, kt, bits, lane, i_begin, i_end, acc0, acc1);
    }

    acc0 += __shfl_xor_sync(0xFFFFFFFFu, acc0, 1);
    acc0 += __shfl_xor_sync(0xFFFFFFFFu, acc0, 2);
    acc1 += __shfl_xor_sync(0xFFFFFFFFu, acc1, 1);
    acc1 += __shfl_xor_sync(0xFFFFFFFFu, acc1, 2);

    __shared__ float s_part[8][16];
    if ((lane & 3) == 0) {
        int r = lane >> 2;
        s_part[warp_id][r] = acc0;
        s_part[warp_id][r + 8] = acc1;
    }
    __syncthreads();

    if (tid < 16) {
        float s = (s_part[0][tid] + s_part[1][tid] + s_part[2][tid] + s_part[3][tid])
                + (s_part[4][tid] + s_part[5][tid] + s_part[6][tid] + s_part[7][tid]);
        y_rot[(J << 4) + tid] = s * 0.25f;
    }
}

// =========================================================================
// 3. Post-RHT with Warp-Shuffle FWHT (h = 1..16) + Fused Final Butterfly + Dynamic SMEM!
//    Eliminates 6 __syncthreads() per call AND drops static 16 KB SMEM to block_len * 4 B!
// =========================================================================
__global__ void trellis_post_rht(
    const float* __restrict__ y_rot,
    const float* __restrict__ post,
    float* __restrict__ y_scaled,
    int N,
    int kind,
    int block_len,
    int out_offset
) {
    if (kind == 0) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            y_scaled[out_offset + idx] = y_rot[idx] * post[idx];
        }
        return;
    }

    extern __shared__ float smem[];
    int b0 = blockIdx.x * block_len;
    int tid = threadIdx.x;

    if (block_len >= 32) {
        int warp = tid >> 5;
        int lane = tid & 31;
        int nwarps = blockDim.x >> 5;
        for (int base = warp * 32; base < block_len; base += nwarps * 32) {
            int i = base + lane;
            float v = y_rot[b0 + i];
            #pragma unroll
            for (int h = 1; h < 32; h <<= 1) {
                float other = __shfl_xor_sync(0xFFFFFFFFu, v, h);
                v = ((lane & h) == 0) ? (v + other) : (other - v);
            }
            smem[i] = v;
        }
    } else {
        for (int i = tid; i < block_len; i += blockDim.x) {
            smem[i] = y_rot[b0 + i];
        }
    }

    if (block_len == 128) {
        __syncthreads();
        if (tid < 64) {
            int idx0 = ((tid & ~31) << 1) | (tid & 31);
            int idx1 = idx0 | 32;
            float a = smem[idx0], b = smem[idx1];
            smem[idx0] = a + b; smem[idx1] = a - b;
        }
        __syncthreads();
        if (tid < 64) {
            float a = smem[tid], b = smem[tid + 64];
            float c = 0.08838834764831845f; // 1 / sqrt(128)
            int g0 = b0 + tid;
            int g1 = b0 + tid + 64;
            y_scaled[out_offset + g0] = ((a + b) * c) * post[g0];
            y_scaled[out_offset + g1] = ((a - b) * c) * post[g1];
        }
        return;
    }

    __syncthreads();
    int half = block_len >> 1;
    int start_h = (block_len >= 32) ? 32 : 1;
    for (int h = start_h; h < half; h <<= 1) {
        int mask = h - 1;
        for (int p = tid; p < half; p += blockDim.x) {
            int idx0 = ((p & ~mask) << 1) | (p & mask);
            int idx1 = idx0 | h;
            float a = smem[idx0];
            float b = smem[idx1];
            smem[idx0] = a + b;
            smem[idx1] = a - b;
        }
        __syncthreads();
    }

    float c = rsqrtf((float)block_len);
    if (block_len >= 64 || block_len < 32) {
        for (int p = tid; p < half; p += blockDim.x) {
            float a = smem[p];
            float b = smem[p + half];
            int g0 = b0 + p;
            int g1 = b0 + p + half;
            y_scaled[out_offset + g0] = ((a + b) * c) * post[g0];
            y_scaled[out_offset + g1] = ((a - b) * c) * post[g1];
        }
    } else {
        for (int i = tid; i < block_len; i += blockDim.x) {
            int gi = b0 + i;
            y_scaled[out_offset + gi] = (smem[i] * c) * post[gi];
        }
    }
}

__global__ void trellis_gather_f32(
    const float* __restrict__ src,
    const int* __restrict__ gather_idx,
    float* __restrict__ dst,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = src[gather_idx[idx]];
    }
}

} // extern "C"
