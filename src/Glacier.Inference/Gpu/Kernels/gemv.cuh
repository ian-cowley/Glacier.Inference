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

// =========================================================================
// GEMV Q5_K: Matrix-Vector Multiplication (y = W * x)
// =========================================================================
__global__ void gemv_q5_k(
    float* __restrict__ y,
    const float* __restrict__ x,
    const BlockQ5_K* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE; // 0..31

    if (warp_id >= m_rows) return;

    int nb = k_cols / QK_K;
    const BlockQ5_K* row_w = W + (size_t)warp_id * nb;
    float row_sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const BlockQ5_K* blk = &row_w[b];
        float d = __half2float(blk->d);
        float min_val = __half2float(blk->dmin);
        const float* x_blk = x + b * QK_K;

        uint8_t qh_val = blk->qh[lane_id];

        int is_idx = 0;
        for (int c = 0; c < 4; c++) {
            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(is_idx + 0, blk->scales, &sc0, &m0);
            get_scale_min_k4(is_idx + 1, blk->scales, &sc1, &m1);

            float d1 = d * sc0;
            float min1 = min_val * m0;
            float d2 = d * sc1;
            float min2 = min_val * m1;

            uint8_t q_byte = blk->qs[c * 32 + lane_id];
            uint8_t bit0 = (qh_val >> (c * 2 + 0)) & 1;
            uint8_t bit1 = (qh_val >> (c * 2 + 1)) & 1;

            int q0 = (int)((q_byte & 0x0F) | (bit0 << 4));
            int q1 = (int)((q_byte >> 4)   | (bit1 << 4));

            float x0 = x_blk[c * 64 + lane_id];
            float x1 = x_blk[c * 64 + 32 + lane_id];

            row_sum += (d1 * (float)q0 - min1) * x0 + (d2 * (float)q1 - min2) * x1;
            is_idx += 2;
        }
    }

    row_sum = warp_reduce_sum(row_sum);

    if (lane_id == 0) {
        if (bias != nullptr) row_sum += bias[warp_id];
        if (residual != nullptr) residual[warp_id] += row_sum;
        if (y != nullptr) y[warp_id] = row_sum;
    }
}

// =========================================================================
// GEMV Q3_K: Matrix-Vector Multiplication (y = W * x)
// =========================================================================
__global__ void gemv_q3_k(
    float* __restrict__ y,
    const float* __restrict__ x,
    const BlockQ3_K* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE; // 0..31

    if (warp_id >= m_rows) return;

    int nb = k_cols / QK_K;
    const BlockQ3_K* row_w = W + (size_t)warp_id * nb;
    float row_sum = 0.0f;

    for (int b = 0; b < nb; b++) {
        const BlockQ3_K* blk = &row_w[b];
        float d_all = __half2float(blk->d);
        const float* x_blk = x + b * QK_K;

        uint8_t hm_byte = blk->hmask[lane_id];

        #pragma unroll
        for (int step = 0; step < 2; step++) {
            int n = step * 128;
            uint8_t q_byte = blk->qs[step * 32 + lane_id];

            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int shift = j * 2;
                uint8_t m = 1 << (step * 4 + j);

                int is_idx = step * 8 + j * 2 + ((lane_id < 16) ? 0 : 1);
                int s = get_scale_k3(is_idx, blk->scales);
                float dl = d_all * (float)s;

                int q = (int)((q_byte >> shift) & 3) - (((hm_byte & m) != 0) ? 0 : 4);
                float val_x = x_blk[n + j * 32 + lane_id];

                row_sum += (dl * (float)q) * val_x;
            }
        }
    }

    row_sum = warp_reduce_sum(row_sum);

    if (lane_id == 0) {
        if (bias != nullptr) row_sum += bias[warp_id];
        if (residual != nullptr) residual[warp_id] += row_sum;
        if (y != nullptr) y[warp_id] = row_sum;
    }
}

// =========================================================================
// GEMV FP32: Unquantized Matrix-Vector Multiplication (y = W * x)
// =========================================================================
__global__ void gemv_fp32(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= m_rows) return;

    const float* row_w = W + (size_t)warp_id * k_cols;
    float row_sum = 0.0f;

    for (int col = lane_id; col < k_cols; col += WARP_SIZE) {
        row_sum += row_w[col] * x[col];
    }

    row_sum = warp_reduce_sum(row_sum);

    if (lane_id == 0) {
        if (bias != nullptr) row_sum += bias[warp_id];
        if (residual != nullptr) residual[warp_id] += row_sum;
        if (y != nullptr) y[warp_id] = row_sum;
    }
}

__global__ void gemv_iq4_xs(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    static constexpr int kvalues_iq4nl[16] = { -127,-104,-83,-65,-49,-35,-22,-10, 1, 13, 25, 38, 53, 69, 89, 113 };
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (warp_id >= m_rows) return;

    int nb = k_cols >> 8;
    const uint8_t* row_w = W + (size_t)warp_id * nb * 136;
    float row_sum = 0.0f;

    int sub = lane_id >> 2;   // 0..7 (8 sub-blocks of 32)
    int qid = lane_id & 3;    // 0..3 (4 lanes per sub-block -> 4 bytes = 8 weights each)
    for (int blk = 0; blk < nb; blk++) {
        const uint8_t* bo = row_w + (size_t)blk * 136;
        uint16_t dh = *(const uint16_t*)bo;
        float d = __half2float(*(const __half*)&dh);
        uint16_t sh = *(const uint16_t*)(bo + 2);
        uint32_t sl = *(const uint32_t*)(bo + 4);
        int ls = (int)(((sl >> (4 * sub)) & 0xFu) | (((sh >> (2 * sub)) & 3u) << 4));
        float dl = d * (float)(ls - 32);

        const uint8_t* qs = bo + 8 + sub * 16 + qid * 4;
        const float* xb = x + blk * 256 + sub * 32 + qid * 4;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint8_t q = qs[j];
            row_sum += dl * (kvalues_iq4nl[q & 0xF] * xb[j] + kvalues_iq4nl[q >> 4] * xb[j + 16]);
        }
    }

    row_sum = warp_reduce_sum(row_sum);
    if (lane_id == 0) {
        if (bias != nullptr) row_sum += bias[warp_id];
        if (residual != nullptr) residual[warp_id] += row_sum;
        if (y != nullptr) y[warp_id] = row_sum;
    }
}

__global__ void gemv_mxfp4(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    static constexpr float kvalues_mxfp4[16] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (warp_id >= m_rows) return;

    int nb = k_cols >> 5;
    const uint8_t* row_w = W + (size_t)warp_id * nb * 17;
    float row_sum = 0.0f;

    int blk_lane = lane_id & 3; // 0..3 (4 lanes per 32-weight block -> 4 bytes = 8 weights each)
    for (int blk = lane_id >> 2; blk < nb; blk += 8) {
        const uint8_t* bo = row_w + (size_t)blk * 17;
        uint32_t e = bo[0];
        uint32_t fbits = (e == 0u) ? 0u : (e << 23);
        float d = __uint_as_float(fbits) * 0.5f;
        const uint8_t* qs = bo + 1 + blk_lane * 4;
        const float* xb = x + blk * 32 + blk_lane * 4;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint8_t q = qs[j];
            row_sum += d * (kvalues_mxfp4[q & 0xF] * xb[j] + kvalues_mxfp4[q >> 4] * xb[j + 16]);
        }
    }

    row_sum = warp_reduce_sum(row_sum);
    if (lane_id == 0) {
        if (bias != nullptr) row_sum += bias[warp_id];
        if (residual != nullptr) residual[warp_id] += row_sum;
        if (y != nullptr) y[warp_id] = row_sum;
    }
}

// =========================================================================
// Unsloth Dynamic 3.1 Mixed-Tensor GEMV Kernels (Alignment-Safe Byte Loads)
// =========================================================================

__device__ uint64_t g_iq2xxs_grid[256];
__device__ uint64_t g_iq2xs_grid[512];
__device__ uint64_t g_iq2s_grid[1024];

__device__ __forceinline__ float load_f16_bytes(const uint8_t* p) {
    uint16_t raw = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    __half h;
    memcpy(&h, &raw, sizeof(uint16_t));
    return __half2float(h);
}

__device__ __forceinline__ uint8_t ksign_iq2xs(uint32_t idx) {
    return (uint8_t)((idx & 127u) | ((__popc(idx & 127u) & 1u) << 7));
}

__device__ __forceinline__ uint32_t decode_iq3s_grid_cu(uint32_t idx) {
    uint32_t b0 = 1u + 2u * ((idx >> 0) & 3u);
    uint32_t b1 = 1u + 2u * ((idx >> 2) & 3u);
    uint32_t b2 = 1u + 2u * ((idx >> 4) & 3u);
    uint32_t b3 = 1u + 2u * ((idx >> 6) & 3u);
    if ((idx & 256u) != 0u) {
        b0 += 8u;
        b1 += ((idx >> 1) & 1u) * 8u;
        b2 += ((idx >> 3) & 1u) * 8u;
        b3 += ((idx >> 5) & 1u) * 8u;
    }
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

__global__ void gemv_iq2_xxs(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (warp_id >= m_rows) return;

    int nb = k_cols >> 8;
    const uint8_t* row_w = W + (size_t)warp_id * nb * 66;
    float row_sum = 0.0f;

    int ib32 = lane_id >> 2;
    int l = lane_id & 3;
    int x_offset = lane_id * 8;

    for (int blk = 0; blk < nb; blk++) {
        const uint8_t* bo = row_w + (size_t)blk * 66;
        float d = load_f16_bytes(bo);
        const uint8_t* q_base = bo + 2 + ib32 * 8;
        uint32_t aux0 = (uint32_t)q_base[0] | ((uint32_t)q_base[1] << 8) | ((uint32_t)q_base[2] << 16) | ((uint32_t)q_base[3] << 24);
        uint32_t aux1 = (uint32_t)q_base[4] | ((uint32_t)q_base[5] << 8) | ((uint32_t)q_base[6] << 16) | ((uint32_t)q_base[7] << 24);

        float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
        uint32_t grid_idx = (aux0 >> (l * 8)) & 0xFFu;
        uint8_t signs = ksign_iq2xs((aux1 >> (7 * l)) & 127u);
        uint64_t grid = g_iq2xxs_grid[grid_idx];

        const float* xb = x + blk * 256 + x_offset;
        float sub = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float g = (float)((uint8_t)((grid >> (j * 8)) & 0xFFu));
            float s = ((signs >> j) & 1) ? -1.0f : 1.0f;
            sub += xb[j] * (g * s);
        }
        row_sum += db * sub;
    }

    row_sum = warp_reduce_sum(row_sum);
    if (lane_id == 0) {
        if (bias != nullptr) row_sum += bias[warp_id];
        if (residual != nullptr) residual[warp_id] += row_sum;
        if (y != nullptr) y[warp_id] = row_sum;
    }
}

__global__ void gemv_iq2_xs(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (warp_id >= m_rows) return;

    int nb = k_cols >> 8;
    const uint8_t* row_w = W + (size_t)warp_id * nb * 74;
    float row_sum = 0.0f;

    int ib32 = lane_id >> 2;
    int l = lane_id & 3;
    int q_off = 2 + (ib32 * 4 + l) * 2;
    int x_offset = lane_id * 8;

    for (int blk = 0; blk < nb; blk++) {
        const uint8_t* bo = row_w + (size_t)blk * 74;
        float d = load_f16_bytes(bo);
        uint16_t q = (uint16_t)bo[q_off] | ((uint16_t)bo[q_off + 1] << 8);
        uint8_t sc = bo[66 + ib32];
        uint32_t ls = (l < 2) ? (sc & 0x0Fu) : (sc >> 4);
        float db = d * (0.5f + (float)ls) * 0.25f;

        uint64_t grid = g_iq2xs_grid[q & 511u];
        uint8_t signs = ksign_iq2xs(q >> 9);

        const float* xb = x + blk * 256 + x_offset;
        float sub = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float g = (float)((uint8_t)((grid >> (j * 8)) & 0xFFu));
            float s = ((signs >> j) & 1) ? -1.0f : 1.0f;
            sub += xb[j] * (g * s);
        }
        row_sum += db * sub;
    }

    row_sum = warp_reduce_sum(row_sum);
    if (lane_id == 0) {
        if (bias != nullptr) row_sum += bias[warp_id];
        if (residual != nullptr) residual[warp_id] += row_sum;
        if (y != nullptr) y[warp_id] = row_sum;
    }
}

__global__ void gemv_iq2_s(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (warp_id >= m_rows) return;

    int nb = k_cols >> 8;
    const uint8_t* row_w = W + (size_t)warp_id * nb * 82;
    float row_sum = 0.0f;

    int ib32 = lane_id >> 2;
    int l = lane_id & 3;
    int x_offset = lane_id * 8;

    for (int blk = 0; blk < nb; blk++) {
        const uint8_t* bo = row_w + (size_t)blk * 82;
        float d = load_f16_bytes(bo);
        uint8_t q_lo = bo[2 + 4 * ib32 + l];
        uint8_t signs = bo[34 + 4 * ib32 + l];
        uint8_t qh = bo[66 + ib32];
        uint32_t grid_idx = (uint32_t)q_lo | (((uint32_t)((qh >> (2 * l)) & 3u)) << 8);
        uint8_t sc = bo[74 + ib32];
        uint32_t ls = (l < 2) ? (sc & 0x0Fu) : (sc >> 4);
        float db = d * (0.5f + (float)ls) * 0.25f;

        uint64_t grid = g_iq2s_grid[grid_idx];
        const float* xb = x + blk * 256 + x_offset;
        float sub = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float g = (float)((uint8_t)((grid >> (j * 8)) & 0xFFu));
            float s = ((signs >> j) & 1) ? -1.0f : 1.0f;
            sub += xb[j] * (g * s);
        }
        row_sum += db * sub;
    }

    row_sum = warp_reduce_sum(row_sum);
    if (lane_id == 0) {
        if (bias != nullptr) row_sum += bias[warp_id];
        if (residual != nullptr) residual[warp_id] += row_sum;
        if (y != nullptr) y[warp_id] = row_sum;
    }
}

__global__ void gemv_iq3_xxs(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (warp_id >= m_rows) return;

    int nb = k_cols >> 8;
    const uint8_t* row_w = W + (size_t)warp_id * nb * 98;
    float row_sum = 0.0f;

    int ib32 = lane_id >> 2;
    int l = lane_id & 3;
    int x_offset = lane_id * 8;

    for (int blk = 0; blk < nb; blk++) {
        const uint8_t* bo = row_w + (size_t)blk * 98;
        float d = load_f16_bytes(bo);
        const uint8_t* ap = bo + 66 + 4 * ib32;
        uint32_t aux32 = (uint32_t)ap[0] | ((uint32_t)ap[1] << 8) | ((uint32_t)ap[2] << 16) | ((uint32_t)ap[3] << 24);
        float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;

        uint8_t signs = ksign_iq2xs((aux32 >> (7 * l)) & 127u);
        uint32_t g0 = decode_iq3s_grid_cu(bo[2 + 8 * ib32 + 2 * l]);
        uint32_t g1 = decode_iq3s_grid_cu(bo[2 + 8 * ib32 + 2 * l + 1]);

        const float* xb = x + blk * 256 + x_offset;
        float sub = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float v0 = (float)((g0 >> (j * 8)) & 0xFFu);
            float v1 = (float)((g1 >> (j * 8)) & 0xFFu);
            sub += xb[j] * (v0 * (((signs >> j) & 1) ? -1.0f : 1.0f));
            sub += xb[4 + j] * (v1 * (((signs >> (4 + j)) & 1) ? -1.0f : 1.0f));
        }
        row_sum += db * sub;
    }

    row_sum = warp_reduce_sum(row_sum);
    if (lane_id == 0) {
        if (bias != nullptr) row_sum += bias[warp_id];
        if (residual != nullptr) residual[warp_id] += row_sum;
        if (y != nullptr) y[warp_id] = row_sum;
    }
}

__global__ void gemv_iq3_s(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (warp_id >= m_rows) return;

    int nb = k_cols >> 8;
    const uint8_t* row_w = W + (size_t)warp_id * nb * 110;
    float row_sum = 0.0f;

    int ib = lane_id >> 2;
    int gp = lane_id & 3;
    int g0_idx = gp * 2;
    int g1_idx = g0_idx + 1;
    int x_offset = lane_id * 8;

    for (int blk = 0; blk < nb; blk++) {
        const uint8_t* bo = row_w + (size_t)blk * 110;
        float d = load_f16_bytes(bo);
        uint8_t sc_byte = bo[106 + (ib >> 1)];
        uint32_t ls = ((ib & 1) == 0) ? (sc_byte & 0x0Fu) : (sc_byte >> 4);
        float db = d * (float)(1u + 2u * ls);

        uint8_t qh = bo[66 + ib];
        uint32_t idx0 = bo[2 + ib * 8 + g0_idx] | (((uint32_t)((qh >> g0_idx) & 1u)) << 8);
        uint32_t idx1 = bo[2 + ib * 8 + g1_idx] | (((uint32_t)((qh >> g1_idx) & 1u)) << 8);
        uint32_t grid0 = decode_iq3s_grid_cu(idx0);
        uint32_t grid1 = decode_iq3s_grid_cu(idx1);

        uint8_t sign_byte = bo[74 + ib * 4 + gp];
        const float* xb = x + blk * 256 + x_offset;
        float sub = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float v0 = (float)((grid0 >> (j * 8)) & 0xFFu);
            float v1 = (float)((grid1 >> (j * 8)) & 0xFFu);
            sub += xb[j] * (v0 * (((sign_byte >> j) & 1) ? -1.0f : 1.0f));
            sub += xb[4 + j] * (v1 * (((sign_byte >> (4 + j)) & 1) ? -1.0f : 1.0f));
        }
        row_sum += db * sub;
    }

    row_sum = warp_reduce_sum(row_sum);
    if (lane_id == 0) {
        if (bias != nullptr) row_sum += bias[warp_id];
        if (residual != nullptr) residual[warp_id] += row_sum;
        if (y != nullptr) y[warp_id] = row_sum;
    }
}

__global__ void gemv_iq4_nl(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    static constexpr int kvalues_iq4nl[16] = { -127,-104,-83,-65,-49,-35,-22,-10, 1, 13, 25, 38, 53, 69, 89, 113 };
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (warp_id >= m_rows) return;

    int nb = k_cols >> 5;
    const uint8_t* row_w = W + (size_t)warp_id * nb * 18;
    float row_sum = 0.0f;
    int l = lane_id & 15;
    int is_high = lane_id >> 4;

    for (int blk = 0; blk < nb; blk++) {
        const uint8_t* bo = row_w + (size_t)blk * 18;
        float d = load_f16_bytes(bo);
        uint8_t q = bo[2 + l];
        int nib = is_high ? (q >> 4) : (q & 0x0F);
        row_sum += d * (float)kvalues_iq4nl[nib] * x[blk * 32 + lane_id];
    }

    row_sum = warp_reduce_sum(row_sum);
    if (lane_id == 0) {
        if (bias != nullptr) row_sum += bias[warp_id];
        if (residual != nullptr) residual[warp_id] += row_sum;
        if (y != nullptr) y[warp_id] = row_sum;
    }
}

__global__ void gemv_q2_k(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (warp_id >= m_rows) return;

    int nb = k_cols >> 8;
    const uint8_t* row_w = W + (size_t)warp_id * nb * 84;
    float row_sum = 0.0f;

    int n_half = lane_id >> 4;
    int l = lane_id & 15;

    for (int blk = 0; blk < nb; blk++) {
        const uint8_t* bo = row_w + (size_t)blk * 84;
        float d = load_f16_bytes(bo + 80);
        float dmin = load_f16_bytes(bo + 82);
        const uint8_t* qs = bo + 16 + n_half * 32;
        const float* xb = x + blk * 256 + n_half * 128;

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int shift = j * 2;
            int is_idx = n_half * 8 + j * 2;
            uint8_t sb0 = bo[is_idx];
            uint8_t sb1 = bo[is_idx + 1];
            float dl0 = d * (float)(sb0 & 0x0F);
            float ml0 = dmin * (float)(sb0 >> 4);
            float dl1 = d * (float)(sb1 & 0x0F);
            float ml1 = dmin * (float)(sb1 >> 4);

            int q0 = (qs[l] >> shift) & 3;
            int q1 = (qs[16 + l] >> shift) & 3;
            row_sum += (dl0 * q0 - ml0) * xb[j * 32 + l]
                     + (dl1 * q1 - ml1) * xb[j * 32 + 16 + l];
        }
    }

    row_sum = warp_reduce_sum(row_sum);
    if (lane_id == 0) {
        if (bias != nullptr) row_sum += bias[warp_id];
        if (residual != nullptr) residual[warp_id] += row_sum;
        if (y != nullptr) y[warp_id] = row_sum;
    }
}

__global__ void gemv_q4_0(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (warp_id >= m_rows) return;

    int nb = k_cols >> 5;
    const uint8_t* row_w = W + (size_t)warp_id * nb * 18;
    float row_sum = 0.0f;
    int l = lane_id & 15;
    int is_high = lane_id >> 4;

    for (int blk = 0; blk < nb; blk++) {
        const uint8_t* bo = row_w + (size_t)blk * 18;
        float d = load_f16_bytes(bo);
        uint8_t q = bo[2 + l];
        int qv = (is_high ? (q >> 4) : (q & 0x0F)) - 8;
        row_sum += d * (float)qv * x[blk * 32 + lane_id];
    }

    row_sum = warp_reduce_sum(row_sum);
    if (lane_id == 0) {
        if (bias != nullptr) row_sum += bias[warp_id];
        if (residual != nullptr) residual[warp_id] += row_sum;
        if (y != nullptr) y[warp_id] = row_sum;
    }
}

__global__ void gemv_q5_0(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (warp_id >= m_rows) return;

    int nb = k_cols >> 5;
    const uint8_t* row_w = W + (size_t)warp_id * nb * 22;
    float row_sum = 0.0f;
    int l = lane_id & 15;
    int is_high = lane_id >> 4;

    for (int blk = 0; blk < nb; blk++) {
        const uint8_t* bo = row_w + (size_t)blk * 22;
        float d = load_f16_bytes(bo);
        uint32_t qh = (uint32_t)bo[2] | ((uint32_t)bo[3] << 8) | ((uint32_t)bo[4] << 16) | ((uint32_t)bo[5] << 24);
        uint8_t q = bo[6 + l];
        int nib = is_high ? (q >> 4) : (q & 0x0F);
        int hb = ((qh >> lane_id) & 1u) << 4;
        int qv = (nib | hb) - 16;
        row_sum += d * (float)qv * x[blk * 32 + lane_id];
    }

    row_sum = warp_reduce_sum(row_sum);
    if (lane_id == 0) {
        if (bias != nullptr) row_sum += bias[warp_id];
        if (residual != nullptr) residual[warp_id] += row_sum;
        if (y != nullptr) y[warp_id] = row_sum;
    }
}

__global__ void gemv_f16(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (warp_id >= m_rows) return;

    const __half* row_w = (const __half*)W + (size_t)warp_id * k_cols;
    float row_sum = 0.0f;
    for (int col = lane_id; col < k_cols; col += WARP_SIZE) {
        row_sum += __half2float(row_w[col]) * x[col];
    }

    row_sum = warp_reduce_sum(row_sum);
    if (lane_id == 0) {
        if (bias != nullptr) row_sum += bias[warp_id];
        if (residual != nullptr) residual[warp_id] += row_sum;
        if (y != nullptr) y[warp_id] = row_sum;
    }
}

__global__ void gemv_bf16(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint8_t* __restrict__ W,
    int k_cols,
    int m_rows,
    const float* __restrict__ bias,
    float* __restrict__ residual
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (warp_id >= m_rows) return;

    const uint16_t* row_w = (const uint16_t*)W + (size_t)warp_id * k_cols;
    float row_sum = 0.0f;
    for (int col = lane_id; col < k_cols; col += WARP_SIZE) {
        uint32_t bits = ((uint32_t)row_w[col]) << 16;
        row_sum += __uint_as_float(bits) * x[col];
    }

    row_sum = warp_reduce_sum(row_sum);
    if (lane_id == 0) {
        if (bias != nullptr) row_sum += bias[warp_id];
        if (residual != nullptr) residual[warp_id] += row_sum;
        if (y != nullptr) y[warp_id] = row_sum;
    }
}

} // extern "C"
