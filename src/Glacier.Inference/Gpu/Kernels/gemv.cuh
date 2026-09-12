// Glacier.Inference High-Performance CUDA Kernels
// Single-token Quantized GEMV Kernels (Q4_K, Q6_K)
// Preserved exact implementation delivering 41.92 tok/s on RTX 4060 (sm_89) and 42.7 tok/s on RTX 3060 (sm_86)
#pragma once

#include "common.cuh"

extern "C" {

// =========================================================================
// 1. GEMV Q4_K: Matrix-Vector Multiplication (y = W * x)
// Each warp of 32 threads computes 1 row of output y
// =========================================================================
__global__ void gemv_q4_k(
    float* __restrict__ y,
    const float* __restrict__ x,
    const BlockQ4_K* __restrict__ W,
    int k_cols,
    int m_rows
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE; // 0..31

    if (warp_id >= m_rows) return;

    int nb = k_cols / QK_K;
    const BlockQ4_K* row_w = W + (size_t)warp_id * nb;
    float row_sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const BlockQ4_K* blk = &row_w[b];
        float d = __half2float(blk->d);
        float min_val = __half2float(blk->dmin);
        const float* x_blk = x + b * QK_K;

        int is_idx = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(is_idx + 0, blk->scales, &sc0, &m0);
            get_scale_min_k4(is_idx + 1, blk->scales, &sc1, &m1);

            float d1 = d * sc0;
            float min1 = min_val * m0;
            float d2 = d * sc1;
            float min2 = min_val * m1;

            // Lane_id (0..31) loads exactly 1 byte from qs (containing two 4-bit nibbles)
            uint8_t q_byte = blk->qs[(j / 2) + lane_id];
            float val_x1 = x_blk[j + lane_id];
            float val_x2 = x_blk[j + 32 + lane_id];

            float w1 = d1 * (q_byte & 0x0F) - min1;
            float w2 = d2 * (q_byte >> 4) - min2;

            row_sum += (w1 * val_x1) + (w2 * val_x2);

            is_idx += 2;
        }
    }

    // Warp-level reduction
    row_sum = warp_reduce_sum(row_sum);

    if (lane_id == 0) {
        y[warp_id] = row_sum;
    }
}

// =========================================================================
// 1b. GEMV Q4_K Fast: Vectorized 128-byte coalesced loads + float4 + factored scales
// Each warp of 32 threads computes 1 row of output y
// Processes all 256 weights of BlockQ4_K in a single step with zero chunk loops
// =========================================================================
__global__ void gemv_q4_k_fast(
    float* __restrict__ y,
    const float* __restrict__ x,
    const BlockQ4_K* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE; // 0..31

    if (warp_id >= m_rows) return;

    int nb = k_cols / QK_K;
    const BlockQ4_K* row_w = W + (size_t)warp_id * nb;
    float row_sum = 0.0f;

    int chunk = lane_id / 8;        // 0..3
    int chunk_lane = lane_id % 8;   // 0..7
    int is_idx = chunk * 2;         // 0, 2, 4, 6
    int x_offset1 = chunk * 64 + chunk_lane * 4;
    int x_offset2 = x_offset1 + 32;

    for (int b = 0; b < nb; b++) {
        const BlockQ4_K* blk = &row_w[b];
        float d = __half2float(blk->d);
        float min_val = __half2float(blk->dmin);
        const float* x_blk = x + b * QK_K;

        // Unpack scales for this lane's chunk
        uint8_t sc0, m0, sc1, m1;
        get_scale_min_k4(is_idx + 0, blk->scales, &sc0, &m0);
        get_scale_min_k4(is_idx + 1, blk->scales, &sc1, &m1);

        float d1 = d * sc0;
        float min1 = min_val * m0;
        float d2 = d * sc1;
        float min2 = min_val * m1;

        // Single 32-bit load per thread: 32 threads load exactly 128 bytes (all 256 nibbles)
        uint32_t q = ((const uint32_t*)blk->qs)[lane_id];

        // 16-byte aligned float4 loads for low nibbles and high nibbles
        float4 x1 = *((const float4*)(x_blk + x_offset1));
        float4 x2 = *((const float4*)(x_blk + x_offset2));

        uint8_t q0 = (uint8_t)(q & 0xFF);
        uint8_t q1 = (uint8_t)((q >> 8) & 0xFF);
        uint8_t q2 = (uint8_t)((q >> 16) & 0xFF);
        uint8_t q3 = (uint8_t)(q >> 24);

        float dot1 = (float)(q0 & 0x0F) * x1.x + (float)(q1 & 0x0F) * x1.y + (float)(q2 & 0x0F) * x1.z + (float)(q3 & 0x0F) * x1.w;
        float sum_x1 = x1.x + x1.y + x1.z + x1.w;

        float dot2 = (float)(q0 >> 4) * x2.x + (float)(q1 >> 4) * x2.y + (float)(q2 >> 4) * x2.z + (float)(q3 >> 4) * x2.w;
        float sum_x2 = x2.x + x2.y + x2.z + x2.w;

        row_sum += (d1 * dot1 - min1 * sum_x1) + (d2 * dot2 - min2 * sum_x2);
    }

    row_sum = warp_reduce_sum(row_sum);

    if (lane_id == 0) {
        if (bias != nullptr) {
            row_sum += bias[warp_id];
        }
        if (residual != nullptr) {
            residual[warp_id] += row_sum;
        }
        if (y != nullptr) {
            y[warp_id] = row_sum;
        }
    }
}

// =========================================================================
// 1c. GEMV Q4_K Fused SwiGLU: Computes Gate and Up GEMV simultaneously in registers
// Writes directly: dst[r] = SiLU(Gate[r]) * Up[r]
// Eliminates 2 intermediate VRAM buffers and 1 extra kernel launch per layer!
// =========================================================================
__global__ void gemv_q4_k_swiglu_fused(
    float* __restrict__ dst,
    const float* __restrict__ x,
    const BlockQ4_K* __restrict__ W_gate,
    const BlockQ4_K* __restrict__ W_up,
    int k_cols,
    int m_rows
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE; // 0..31

    if (warp_id >= m_rows) return;

    int nb = k_cols / QK_K;
    const BlockQ4_K* row_gate = W_gate + (size_t)warp_id * nb;
    const BlockQ4_K* row_up = W_up + (size_t)warp_id * nb;
    float sum_gate = 0.0f;
    float sum_up = 0.0f;

    int chunk = lane_id / 8;
    int chunk_lane = lane_id % 8;
    int is_idx = chunk * 2;
    int x_offset1 = chunk * 64 + chunk_lane * 4;
    int x_offset2 = x_offset1 + 32;

    for (int b = 0; b < nb; b++) {
        const float* x_blk = x + b * QK_K;
        float4 x1 = *((const float4*)(x_blk + x_offset1));
        float4 x2 = *((const float4*)(x_blk + x_offset2));
        float sum_x1 = x1.x + x1.y + x1.z + x1.w;
        float sum_x2 = x2.x + x2.y + x2.z + x2.w;

        const BlockQ4_K* blk_g = &row_gate[b];
        const BlockQ4_K* blk_u = &row_up[b];

        // Dual-issue both loads concurrently to hide memory latency
        uint32_t q_g = ((const uint32_t*)blk_g->qs)[lane_id];
        uint32_t q_u = ((const uint32_t*)blk_u->qs)[lane_id];

        // 1. Gate row computation
        {
            float d = __half2float(blk_g->d);
            float min_val = __half2float(blk_g->dmin);

            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(is_idx + 0, blk_g->scales, &sc0, &m0);
            get_scale_min_k4(is_idx + 1, blk_g->scales, &sc1, &m1);

            float d1 = d * sc0;
            float min1 = min_val * m0;
            float d2 = d * sc1;
            float min2 = min_val * m1;

            uint8_t q0 = (uint8_t)(q_g & 0xFF);
            uint8_t q1 = (uint8_t)((q_g >> 8) & 0xFF);
            uint8_t q2 = (uint8_t)((q_g >> 16) & 0xFF);
            uint8_t q3 = (uint8_t)(q_g >> 24);

            float dot1 = (float)(q0 & 0x0F) * x1.x + (float)(q1 & 0x0F) * x1.y + (float)(q2 & 0x0F) * x1.z + (float)(q3 & 0x0F) * x1.w;
            float dot2 = (float)(q0 >> 4) * x2.x + (float)(q1 >> 4) * x2.y + (float)(q2 >> 4) * x2.z + (float)(q3 >> 4) * x2.w;

            sum_gate += (d1 * dot1 - min1 * sum_x1) + (d2 * dot2 - min2 * sum_x2);
        }

        // 2. Up row computation
        {
            float d = __half2float(blk_u->d);
            float min_val = __half2float(blk_u->dmin);

            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(is_idx + 0, blk_u->scales, &sc0, &m0);
            get_scale_min_k4(is_idx + 1, blk_u->scales, &sc1, &m1);

            float d1 = d * sc0;
            float min1 = min_val * m0;
            float d2 = d * sc1;
            float min2 = min_val * m1;

            uint8_t q0 = (uint8_t)(q_u & 0xFF);
            uint8_t q1 = (uint8_t)((q_u >> 8) & 0xFF);
            uint8_t q2 = (uint8_t)((q_u >> 16) & 0xFF);
            uint8_t q3 = (uint8_t)(q_u >> 24);

            float dot1 = (float)(q0 & 0x0F) * x1.x + (float)(q1 & 0x0F) * x1.y + (float)(q2 & 0x0F) * x1.z + (float)(q3 & 0x0F) * x1.w;
            float dot2 = (float)(q0 >> 4) * x2.x + (float)(q1 >> 4) * x2.y + (float)(q2 >> 4) * x2.z + (float)(q3 >> 4) * x2.w;

            sum_up += (d1 * dot1 - min1 * sum_x1) + (d2 * dot2 - min2 * sum_x2);
        }
    }

    sum_gate = warp_reduce_sum(sum_gate);
    sum_up = warp_reduce_sum(sum_up);

    if (lane_id == 0) {
        float silu = sum_gate / (1.0f + __expf(-sum_gate));
        dst[warp_id] = silu * sum_up;
    }
}

// =========================================================================
// 1d. GEMV Q4_K Aligned: 128-byte hardware cache coalesced layout
// W_qs is 128-byte aligned contiguous array of nibbles
// W_scales is pre-unpacked FP16 scales (d * sc0, dmin * m0, etc.) as half2
// =========================================================================
__global__ void gemv_q4_k_aligned(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ W_qs,
    const half2* __restrict__ W_scales,
    int k_cols,
    int m_rows
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= m_rows) return;

    int nb = k_cols / QK_K;
    const uint8_t* row_qs = W_qs + (size_t)warp_id * nb * 128;
    const half2* row_scales = W_scales + (size_t)warp_id * nb * 8;
    float row_sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const uint8_t* blk_qs = row_qs + b * 128;
        const half2* blk_scales = row_scales + b * 8;
        const float* x_blk = x + b * QK_K;

        #pragma unroll
        for (int s = 0; s < 4; s++) {
            half2 sc0 = blk_scales[s * 2 + 0];
            half2 sc1 = blk_scales[s * 2 + 1];

            float d1 = __low2float(sc0);
            float min1 = __high2float(sc0);
            float d2 = __low2float(sc1);
            float min2 = __high2float(sc1);

            int j = s * 64;
            uint8_t q_byte = blk_qs[s * 32 + lane_id];
            float val_x1 = x_blk[j + lane_id];
            float val_x2 = x_blk[j + 32 + lane_id];

            float w1 = d1 * (float)(q_byte & 0x0F) - min1;
            float w2 = d2 * (float)(q_byte >> 4) - min2;

            row_sum += (w1 * val_x1) + (w2 * val_x2);
        }
    }

    row_sum = warp_reduce_sum(row_sum);

    if (lane_id == 0) {
        y[warp_id] = row_sum;
    }
}

// =========================================================================
// 2. GEMV Q6_K: Matrix-Vector Multiplication (y = W * x)
// Each warp of 32 threads computes 1 row of output y
// =========================================================================
__global__ void gemv_q6_k(
    float* __restrict__ y,
    const float* __restrict__ x,
    const BlockQ6_K* __restrict__ W,
    int k_cols,
    int m_rows
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE; // 0..31

    if (warp_id >= m_rows) return;

    int nb = k_cols / QK_K;
    const BlockQ6_K* row_w = W + (size_t)warp_id * nb;
    float row_sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const BlockQ6_K* blk = &row_w[b];
        float d = __half2float(blk->d);
        const float* x_blk = x + b * QK_K;

        const uint8_t* ql = blk->ql;
        const uint8_t* qh = blk->qh;
        const int8_t* sc = blk->scales;

        for (int n = 0; n < QK_K; n += 128) {
            int is_idx = lane_id / 16;
            int8_t q1 = (int8_t)((ql[lane_id + 0] & 0x0F) | (((qh[lane_id] >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql[lane_id + 32] & 0x0F) | (((qh[lane_id] >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql[lane_id + 0] >> 4) | (((qh[lane_id] >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((ql[lane_id + 32] >> 4) | (((qh[lane_id] >> 6) & 3) << 4)) - 32;

            float w1 = d * sc[is_idx + 0] * q1;
            float w2 = d * sc[is_idx + 2] * q2;
            float w3 = d * sc[is_idx + 4] * q3;
            float w4 = d * sc[is_idx + 6] * q4;

            row_sum += w1 * x_blk[n + lane_id + 0];
            row_sum += w2 * x_blk[n + lane_id + 32];
            row_sum += w3 * x_blk[n + lane_id + 64];
            row_sum += w4 * x_blk[n + lane_id + 96];

            ql += 64;
            qh += 32;
            sc += 8;
        }
    }

    row_sum = warp_reduce_sum(row_sum);

    if (lane_id == 0) {
        y[warp_id] = row_sum;
    }
}

// =========================================================================
// 2b. GEMV Q6_K Fast: Unrolled with factored scales & vectorized lane access
// =========================================================================
__global__ void gemv_q6_k_fast(
    float* __restrict__ y,
    const float* __restrict__ x,
    const BlockQ6_K* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE; // 0..31

    if (warp_id >= m_rows) return;

    int nb = k_cols / QK_K;
    const BlockQ6_K* row_w = W + (size_t)warp_id * nb;
    float row_sum = 0.0f;
    int is_idx = lane_id / 16;

    for (int b = 0; b < nb; b++) {
        const BlockQ6_K* blk = &row_w[b];
        float d = __half2float(blk->d);
        const float* x_blk = x + b * QK_K;
        float blk_sum = 0.0f;

        #pragma unroll
        for (int step = 0; step < 2; step++) {
            int n = step * 128;
            const uint8_t* ql = blk->ql + step * 64;
            const uint8_t* qh = blk->qh + step * 32;
            const int8_t* sc = blk->scales + step * 8;

            uint8_t qh_val = qh[lane_id];
            uint8_t ql_low = ql[lane_id + 0];
            uint8_t ql_high = ql[lane_id + 32];

            int8_t q1 = (int8_t)((ql_low & 0x0F) | (((qh_val >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql_high & 0x0F) | (((qh_val >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql_low >> 4) | (((qh_val >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((ql_high >> 4) | (((qh_val >> 6) & 3) << 4)) - 32;

            float s1 = (float)sc[is_idx + 0];
            float s2 = (float)sc[is_idx + 2];
            float s3 = (float)sc[is_idx + 4];
            float s4 = (float)sc[is_idx + 6];

            blk_sum += (s1 * (float)q1) * x_blk[n + lane_id + 0]
                     + (s2 * (float)q2) * x_blk[n + lane_id + 32]
                     + (s3 * (float)q3) * x_blk[n + lane_id + 64]
                     + (s4 * (float)q4) * x_blk[n + lane_id + 96];
        }

        row_sum += d * blk_sum;
    }

    row_sum = warp_reduce_sum(row_sum);

    if (lane_id == 0) {
        if (bias != nullptr) {
            row_sum += bias[warp_id];
        }
        if (residual != nullptr) {
            residual[warp_id] += row_sum;
        }
        if (y != nullptr) {
            y[warp_id] = row_sum;
        }
    }
}

// =========================================================================
// 3. GEMV Q8_0: Matrix-Vector Multiplication (y = W * x)
// Each warp of 32 threads computes 1 row of output y
// Lane_id (0..31) loads exactly 1 signed byte from qs and 1 float from x
// =========================================================================
__global__ void gemv_q8_0(
    float* __restrict__ y,
    const float* __restrict__ x,
    const BlockQ8_0* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE; // 0..31

    if (warp_id >= m_rows) return;

    int nb = k_cols / QK8_0;
    const BlockQ8_0* row_w = W + (size_t)warp_id * nb;
    float row_sum = 0.0f;

    int b = 0;
    for (; b + 1 < nb; b += 2) {
        const BlockQ8_0* blk0 = &row_w[b];
        const BlockQ8_0* blk1 = &row_w[b + 1];

        float d0 = __half2float(blk0->d);
        float d1 = __half2float(blk1->d);

        float q0 = (float)blk0->qs[lane_id];
        float q1 = (float)blk1->qs[lane_id];

        float x0 = x[b * QK8_0 + lane_id];
        float x1 = x[(b + 1) * QK8_0 + lane_id];

        row_sum += d0 * (q0 * x0) + d1 * (q1 * x1);
    }
    if (b < nb) {
        const BlockQ8_0* blk = &row_w[b];
        float d = __half2float(blk->d);
        float q = (float)blk->qs[lane_id];
        float xv = x[b * QK8_0 + lane_id];
        row_sum += d * (q * xv);
    }

    row_sum = warp_reduce_sum(row_sum);

    if (lane_id == 0) {
        if (bias != nullptr) {
            row_sum += bias[warp_id];
        }
        if (residual != nullptr) {
            residual[warp_id] += row_sum;
        }
        if (y != nullptr) {
            y[warp_id] = row_sum;
        }
    }
}

} // extern "C"
